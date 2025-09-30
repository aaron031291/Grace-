#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
event_bus.py
EventBus — Central event routing system for Grace governance kernel (production).

Features
- Typed topics (EventType) + "*" wildcard
- Async + sync handlers supported
- Per-handler timeout & retries with jitter backoff
- Optional bounded queue (backpressure) per topic
- Dead-letter capture with reasons
- Correlation tracking
- Middleware hooks (before_publish, after_publish, before_deliver, after_deliver)
- Metrics snapshot (publish/deliver counts, in-flight, queue depth)
- Graceful shutdown & draining
- UTC-safe timestamps

Public API (async unless noted):
    subscribe(event_type|str, handler)
    unsubscribe(event_type|str, handler)
    use(hook, middleware)
    publish(event_type|str, payload, correlation_id=None) -> str
    publish_sync(event_type|str, payload, correlation_id=None) -> str   # sync wrapper
    get_events_by_correlation(correlation_id) -> list[dict]
    get_recent_events(limit=100) -> list[dict]
    get_dead_letters(limit=100) -> list[dict]
    metrics() -> dict
    clear_history(keep_recent=1000)
    close() / await aclose()
    async with EventBus(...) as bus: ...

Notes
- Handlers and middleware receive a dict event: {id, type, payload, correlation_id, timestamp}.
- EventType is optional; str topics work equally (kept JSON-friendly).
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple, Union

try:
    # Optional: import EventType from your contracts (not required for str topics)
    from .contracts import EventType, generate_correlation_id
except Exception:  # pragma: no cover - keep standalone-friendly
    class EventType:  # minimal shim
        value: str
        def __init__(self, v: str): self.value = v  # type: ignore
    def generate_correlation_id() -> str:
        import uuid
        return f"corr_{uuid.uuid4().hex[:12]}"

logger = logging.getLogger(__name__)

# ---- Types ----

EventPayload = Mapping[str, Any]
EventDict = Dict[str, Any]
Handler = Callable[[Mapping[str, Any]], Union[None, Awaitable[None]]]
Middleware = Callable[[Mapping[str, Any]], Union[None, Awaitable[None]]]

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()

@dataclass(frozen=True)
class PublishedEvent:
    id: str                 # evt_000123
    type: str               # EventType value or literal topic
    payload: EventPayload
    correlation_id: str
    timestamp: str          # ISO8601 UTC


class EventBus:
    """
    Event-driven messaging system for governance components.
    Thread-safe for asyncio; not thread-safe across Python threads.
    """

    def __init__(
        self,
        *,
        handler_timeout_s: float = 10.0,
        handler_retries: int = 0,
        per_topic_queue_size: Optional[int] = None,  # None = immediate dispatch
        jitter_ratio: float = 0.1,                   # retry jitter ratio (0..1)
        deliver_parallelism: int = 0,                # 0 = unbounded per publish call
    ) -> None:
        self._subs: Dict[str, List[Handler]] = {}
        self._wildcard_subs: List[Handler] = []
        self._history: List[PublishedEvent] = []
        self._corr_index: Dict[str, List[str]] = {}
        self._dead_letter: List[Tuple[PublishedEvent, str]] = []
        self._middlewares: Dict[str, List[Middleware]] = {
            "before_publish": [],
            "after_publish": [],
            "before_deliver": [],
            "after_deliver": [],
        }
        self._lock = asyncio.Lock()
        self._seq = 0

        # delivery config
        self._timeout_s = float(handler_timeout_s)
        self._retries = max(0, int(handler_retries))
        self._jitter_ratio = max(0.0, min(1.0, jitter_ratio))
        self._deliver_parallelism = max(0, int(deliver_parallelism))
        self._sema = None if self._deliver_parallelism == 0 else asyncio.Semaphore(self._deliver_parallelism)

        # optional per-topic queues (backpressure)
        self._queue_size = per_topic_queue_size
        self._topic_queues: Dict[str, asyncio.Queue[PublishedEvent]] = {}
        self._queue_tasks: Dict[str, asyncio.Task] = {}

        # lifecycle
        self._closing = False
        self._closed = False

        # metrics
        self._published = 0
        self._delivered = 0
        self._delivery_errors = 0
        self._inflight_deliveries = 0

    # --------------- Subscription API ---------------

    async def subscribe(self, event_type: Union[str, EventType], handler: Handler) -> None:
        """Subscribe a handler to a specific event type or '*' for all."""
        topic = event_type.value if hasattr(event_type, "value") else str(event_type)
        async with self._lock:
            if topic == "*":
                self._wildcard_subs.append(handler)
            else:
                self._subs.setdefault(topic, []).append(handler)
                # spin up queue worker if queueing is enabled
                if self._queue_size is not None and topic not in self._topic_queues:
                    q: asyncio.Queue[PublishedEvent] = asyncio.Queue(maxsize=self._queue_size)
                    self._topic_queues[topic] = q
                    self._queue_tasks[topic] = asyncio.create_task(self._queue_worker(topic, q))
        logger.info("Subscribed handler to %s", topic)

    async def subscribe_once(self, event_type: Union[str, EventType], handler: Handler) -> None:
        """Subscribe a handler that will be auto-unsubscribed after first call."""
        topic = event_type.value if hasattr(event_type, "value") else str(event_type)

        async def wrapper(evt: Mapping[str, Any]):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(evt)
                else:
                    handler(evt)
            finally:
                await self.unsubscribe(topic, wrapper)

        await self.subscribe(topic, wrapper)

    async def unsubscribe(self, event_type: Union[str, EventType], handler: Handler) -> None:
        """Unsubscribe a handler."""
        topic = event_type.value if hasattr(event_type, "value") else str(event_type)
        async with self._lock:
            if topic == "*":
                try:
                    self._wildcard_subs.remove(handler)
                except ValueError:
                    logger.warning("Handler not found for wildcard")
            else:
                if topic in self._subs:
                    try:
                        self._subs[topic].remove(handler)
                        if not self._subs[topic]:
                            del self._subs[topic]
                    except ValueError:
                        logger.warning("Handler not found for %s", topic)
        logger.info("Unsubscribed handler from %s", topic)

    # --------------- Middleware ---------------

    def use(self, hook: str, middleware: Middleware) -> None:
        """
        Register a middleware: hook ∈ {"before_publish","after_publish","before_deliver","after_deliver"}.
        Middleware signature: async|sync (event_dict) -> None
        """
        if hook not in self._middlewares:
            raise ValueError(f"Unknown hook '{hook}'")
        self._middlewares[hook].append(middleware)

    async def _run_middlewares(self, hook: str, event_dict: Mapping[str, Any]) -> None:
        for mw in self._middlewares.get(hook, []):
            try:
                if asyncio.iscoroutinefunction(mw):
                    await mw(event_dict)
                else:
                    mw(event_dict)
            except Exception as e:
                logger.exception("Middleware '%s' failed: %s", hook, e)

    # --------------- Publish API ---------------

    async def publish(
        self,
        event_type: Union[str, EventType],
        payload: EventPayload,
        *,
        correlation_id: Optional[str] = None,
    ) -> str:
        """
        Publish an event to all subscribers. Returns the correlation_id used.
        If per_topic_queue_size is set, delivery is enqueued; otherwise dispatched immediately.
        """
        if self._closing or self._closed:
            raise RuntimeError("EventBus is closing/closed; publish rejected")

        topic = event_type.value if hasattr(event_type, "value") else str(event_type)
        if correlation_id is None:
            correlation_id = generate_correlation_id()

        async with self._lock:
            evt_id = f"evt_{self._seq:06d}"
            self._seq += 1
            self._published += 1

        event = PublishedEvent(
            id=evt_id,
            type=topic,
            payload=payload,
            correlation_id=correlation_id,
            timestamp=_utcnow(),
        )

        # Record tracking before delivery
        async with self._lock:
            self._history.append(event)
            self._corr_index.setdefault(correlation_id, []).append(event.id)

        # Middlewares (pre-publish)
        await self._run_middlewares("before_publish", self._as_dict(event))

        # Delivery
        if self._queue_size is not None:
            q = self._topic_queues.get(topic)
            if q is None:
                logger.debug("No queue for topic %s; no subscribers yet", topic)
            else:
                await q.put(event)
        else:
            await self._dispatch(event)

        # Middlewares (post-publish)
        await self._run_middlewares("after_publish", self._as_dict(event))
        logger.info("Published %s with correlation_id=%s (id=%s)", topic, correlation_id, event.id)
        return correlation_id

    def publish_sync(
        self,
        event_type: Union[str, EventType],
        payload: EventPayload,
        *,
        correlation_id: Optional[str] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> str:
        """Synchronous wrapper for publish (for legacy code / scripts)."""
        loop = loop or asyncio.get_event_loop()
        return loop.run_until_complete(self.publish(event_type, payload, correlation_id=correlation_id))

    async def _queue_worker(self, topic: str, q: asyncio.Queue[PublishedEvent]) -> None:
        """Background worker delivering queued events for a topic."""
        try:
            while True:
                event = await q.get()
                try:
                    await self._dispatch(event)
                except Exception:
                    # _dispatch already logs and dead-letters
                    pass
                finally:
                    q.task_done()
                if self._closing and q.empty():
                    # allow worker to end after draining
                    break
        except asyncio.CancelledError:  # shutdown
            pass
        finally:
            logger.info("Queue worker for %s stopped", topic)

    async def _dispatch(self, event: PublishedEvent) -> None:
        """Dispatch to specific and wildcard subscribers with safety, timeout, retries."""
        await self._run_middlewares("before_deliver", self._as_dict(event))

        async with self._lock:
            handlers = list(self._subs.get(event.type, [])) + list(self._wildcard_subs)

        if not handlers:
            logger.debug("No subscribers for %s", event.type)
            return

        async def _deliver(h: Handler):
            if self._sema:
                async with self._sema:
                    await self._deliver_with_retry(h, event)
            else:
                await self._deliver_with_retry(h, event)

        tasks = [asyncio.create_task(_deliver(h)) for h in handlers]
        self._inflight_deliveries += len(tasks)
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self._inflight_deliveries -= len(tasks)

        await self._run_middlewares("after_deliver", self._as_dict(event))

    async def _deliver_with_retry(self, handler: Handler, event: PublishedEvent) -> None:
        """Deliver to handler with timeout and retries; dead-letter on failure."""
        attempt = 0
        while True:
            try:
                await asyncio.wait_for(self._call(handler, self._as_dict(event)), timeout=self._timeout_s)
                async with self._lock:
                    self._delivered += 1
                return
            except asyncio.TimeoutError:
                reason = f"timeout after {self._timeout_s}s"
            except Exception as e:
                reason = f"handler exception: {e!r}"

            attempt += 1
            if attempt > self._retries:
                logger.error("Delivery to handler failed (%s). Dead-lettering %s", reason, event.id)
                async with self._lock:
                    self._dead_letter.append((event, reason))
                    self._delivery_errors += 1
                return
            else:
                # exponential backoff + jitter
                base = min(2 ** (attempt - 1), 8.0)
                jitter = base * self._jitter_ratio * random.random()
                delay = base + jitter
                logger.warning(
                    "Delivery attempt %d failed (%s) for %s; retrying in %.2fs...",
                    attempt, reason, event.id, delay
                )
                await asyncio.sleep(delay)

    @staticmethod
    async def _call(handler: Handler, event_dict: Mapping[str, Any]) -> None:
        if asyncio.iscoroutinefunction(handler):
            await handler(event_dict)
        else:
            handler(event_dict)

    # --------------- Queries & maintenance ---------------

    def get_events_by_correlation(self, correlation_id: str) -> List[Mapping[str, Any]]:
        """Return all events for a correlation_id (in publish order)."""
        ids = self._corr_index.get(correlation_id, [])
        if not ids:
            return []
        idset = set(ids)
        return [self._as_dict(e) for e in self._history if e.id in idset]

    def get_recent_events(self, limit: int = 100) -> List[Mapping[str, Any]]:
        """Return most recent N events."""
        return [self._as_dict(e) for e in self._history[-int(limit):]]

    def get_dead_letters(self, limit: int = 100) -> List[Mapping[str, Any]]:
        """Return most recent dead-lettered events with reasons."""
        out: List[Dict[str, Any]] = []
        for ev, reason in self._dead_letter[-int(limit):]:
            d = self._as_dict(ev)
            d["dead_letter_reason"] = reason
            out.append(d)
        return out

    def metrics(self) -> Dict[str, Any]:
        """Return a snapshot of bus metrics."""
        queues = {
            topic: q.qsize()
            for topic, q in self._topic_queues.items()
        }
        return {
            "published": self._published,
            "delivered": self._delivered,
            "delivery_errors": self._delivery_errors,
            "inflight_deliveries": self._inflight_deliveries,
            "queues": queues,
            "subscribers": {t: len(hs) for t, hs in self._subs.items()},
            "wildcard_subscribers": len(self._wildcard_subs),
            "closing": self._closing,
            "closed": self._closed,
        }

    async def clear_history(self, keep_recent: int = 1000) -> None:
        """Trim in-memory history and dead letters (keeps correlation index consistent for kept items)."""
        async with self._lock:
            if len(self._history) > keep_recent:
                self._history = self._history[-int(keep_recent):]
                # rebuild corr index from kept events
                self._corr_index.clear()
                for e in self._history:
                    self._corr_index.setdefault(e.correlation_id, []).append(e.id)
                logger.info("Cleared event history; kept last %d events", keep_recent)
            if len(self._dead_letter) > 10_000:
                self._dead_letter = self._dead_letter[-10_000:]

    # --------------- Lifecycle ---------------

    async def aclose(self, drain: bool = True, drain_timeout_s: float = 15.0) -> None:
        """
        Gracefully close the bus.
        - If queueing: stop accepting new publishes, drain queues, stop workers.
        - Wait for in-flight deliveries (bounded by drain_timeout_s).
        """
        if self._closed:
            return
        self._closing = True

        # Wait for queues to drain
        if self._queue_size is not None:
            # Signal workers via _closing flag; they exit when queues empty
            try:
                await asyncio.wait_for(self._wait_queues_empty(), timeout=drain_timeout_s)
            except asyncio.TimeoutError:
                logger.warning("Timed out draining queues; cancelling workers")
            finally:
                # Cancel any remaining workers
                for task in self._queue_tasks.values():
                    task.cancel()
                await asyncio.gather(*self._queue_tasks.values(), return_exceptions=True)
                self._topic_queues.clear()
                self._queue_tasks.clear()

        # Optionally wait for in-flight deliveries
        if drain:
            try:
                await asyncio.wait_for(self._wait_inflight_zero(), timeout=drain_timeout_s)
            except asyncio.TimeoutError:
                logger.warning("Timed out waiting for inflight deliveries to complete")

        self._closed = True
        logger.info("EventBus closed")

    def close(self) -> None:
        """Synchronous close (best-effort)."""
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            pass
        if loop and loop.is_running():
            # When inside an event loop (e.g., FastAPI), schedule async close
            loop.create_task(self.aclose())
        else:
            asyncio.run(self.aclose())

    async def _wait_queues_empty(self) -> None:
        if not self._topic_queues:
            return
        await asyncio.gather(*(q.join() for q in self._topic_queues.values()))

    async def _wait_inflight_zero(self) -> None:
        while self._inflight_deliveries > 0:
            await asyncio.sleep(0.01)

    # --------------- Utils ---------------

    @staticmethod
    def _as_dict(e: PublishedEvent) -> Dict[str, Any]:
        return {
            "id": e.id,
            "type": e.type,
            "payload": dict(e.payload),
            "correlation_id": e.correlation_id,
            "timestamp": e.timestamp,
        }

    # --------------- Context manager ---------------

    async def __aenter__(self) -> "EventBus":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()
