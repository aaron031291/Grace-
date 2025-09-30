# parliament.py
# -*- coding: utf-8 -*-
"""
Parliament — Democratic review system for major governance decisions (production).

Highlights
- Strong typing & UTC timestamps
- Configurable voting thresholds per proposal type
- Weighted voting with participation floors
- Async-safe (internal lock) + EventBus integration
- Clear status lifecycle & serialization helpers
- Emits telemetry Experience records to MemoryCore
- Robust handler subscriptions with async init() hook
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..core.contracts import Experience
from ..core.contracts import generate_correlation_id  # for notifications
from ..core import EventBus, MemoryCore

logger = logging.getLogger(__name__)


# --------------------------- Utilities ---------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    # Always serialize timestamps as ISO-8601 UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


# --------------------------- Enums ---------------------------

class ReviewStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class VoteType(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    NEEDS_INFO = "needs_info"


# --------------------------- Data Models ---------------------------

@dataclass
class VoteRecord:
    vote: VoteType
    rationale: str
    weight: float
    at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["vote"] = self.vote.value
        d["at"] = _iso(self.at)
        return d


@dataclass
class ParliamentMember:
    """Represents a member of the governance parliament."""
    member_id: str
    name: str
    expertise: List[str]
    weight: float = 1.0
    active: bool = True
    vote_history: List[Tuple[str, VoteRecord]] = field(default_factory=list)

    def record_vote(self, proposal_id: str, vr: VoteRecord) -> None:
        self.vote_history.append((proposal_id, vr))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "member_id": self.member_id,
            "name": self.name,
            "expertise": list(self.expertise),
            "weight": self.weight,
            "active": self.active,
            "vote_count": len(self.vote_history),
        }


@dataclass
class ReviewProposal:
    """Represents a proposal for parliamentary review."""
    proposal_id: str
    title: str
    description: str
    proposal_type: str  # "policy" | "constitutional" | "operational"
    urgency: str = "normal"  # "low","normal","high","critical"
    submitted_at: datetime = field(default_factory=_utcnow)
    status: ReviewStatus = ReviewStatus.PENDING
    deadline: datetime = field(init=False)
    votes: Dict[str, VoteRecord] = field(default_factory=dict)
    discussion: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        days_map = {"critical": 1, "high": 3, "normal": 7, "low": 14}
        self.deadline = self.submitted_at + timedelta(days=days_map.get(self.urgency, 7))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "title": self.title,
            "description": self.description,
            "proposal_type": self.proposal_type,
            "urgency": self.urgency,
            "submitted_at": _iso(self.submitted_at),
            "status": self.status.value,
            "deadline": _iso(self.deadline),
            "vote_count": len(self.votes),
            "discussion_count": len(self.discussion),
        }


# --------------------------- Parliament ---------------------------

class Parliament:
    """
    Democratic review system for major governance decisions and policy updates.
    Manages committee reviews, voting processes, and democratic validation.

    Events consumed:
      - "GOVERNANCE_NEEDS_REVIEW" (payload: {decision_id, rationale, ...})
      - "PARLIAMENT_VOTE_CAST" (payload: {proposal_id, member_id, vote, rationale})

    Events produced:
      - "PARLIAMENT_NOTIFICATION" (review assigned)
      - "PARLIAMENT_DECISION" (finalized outcome)
    """

    # Default thresholds & floors; override via ctor args if needed
    DEFAULT_VOTING_THRESHOLDS: Mapping[str, float] = {
        "policy": 0.60,
        "constitutional": 0.75,
        "operational": 0.50,
    }
    PARTICIPATION_FLOOR: float = 0.60  # 60% of active members must vote to finalize early
    REJECTION_FLOOR: float = 0.40      # direct rejection if ≥ 40% weighted reject with floor met

    def __init__(
        self,
        event_bus: EventBus,
        memory_core: MemoryCore,
        *,
        voting_thresholds: Optional[Mapping[str, float]] = None,
        participation_floor: Optional[float] = None,
        rejection_floor: Optional[float] = None,
    ) -> None:
        self.event_bus = event_bus
        self.memory_core = memory_core

        self._members: Dict[str, ParliamentMember] = self._default_members()
        self._active: Dict[str, ReviewProposal] = {}
        self._history: List[Dict[str, Any]] = []

        self._thresholds: Dict[str, float] = dict(self.DEFAULT_VOTING_THRESHOLDS)
        if voting_thresholds:
            self._thresholds.update({k: float(v) for k, v in voting_thresholds.items()})

        self._participation_floor = float(participation_floor) if participation_floor is not None else self.PARTICIPATION_FLOOR
        self._rejection_floor = float(rejection_floor) if rejection_floor is not None else self.REJECTION_FLOOR

        self._lock = asyncio.Lock()
        # use async init to attach subscriptions so __init__ stays sync-safe

    async def async_init(self) -> None:
        """Attach event subscriptions (call once during kernel startup)."""
        await self.event_bus.subscribe("GOVERNANCE_NEEDS_REVIEW", self._on_review_request)
        await self.event_bus.subscribe("PARLIAMENT_VOTE_CAST", self._on_vote_cast)
        logger.info("Parliament subscriptions attached")

    # --------------------------- Members ---------------------------

    def _default_members(self) -> Dict[str, ParliamentMember]:
        defaults = [
            ParliamentMember("ethics_chair", "Ethics Committee Chair", ["ethics", "fairness", "harm_prevention"], 1.2),
            ParliamentMember("tech_lead", "Technical Lead", ["security", "privacy", "technical"], 1.1),
            ParliamentMember("transparency_officer", "Transparency Officer", ["transparency", "accountability", "audit"], 1.1),
            ParliamentMember("user_advocate", "User Advocate", ["user_rights", "accessibility", "usability"], 1.0),
            ParliamentMember("legal_counsel", "Legal Counsel", ["legal", "compliance", "constitutional"], 1.2),
            ParliamentMember("domain_expert_1", "Domain Expert (AI/ML)", ["machine_learning", "ai_safety"], 1.0),
            ParliamentMember("domain_expert_2", "Domain Expert (Governance)", ["governance", "policy", "process"], 1.0),
        ]
        return {m.member_id: m for m in defaults}

    def add_member(self, member_id: str, name: str, expertise: List[str], *, weight: float = 1.0) -> bool:
        if member_id in self._members:
            logger.warning("Member %s already exists", member_id)
            return False
        self._members[member_id] = ParliamentMember(member_id, name, expertise, weight)
        logger.info("Added parliament member: %s (%s)", name, member_id)
        return True

    def remove_member(self, member_id: str) -> bool:
        m = self._members.get(member_id)
        if not m:
            logger.warning("Member %s not found", member_id)
            return False
        m.active = False
        logger.info("Deactivated parliament member: %s", member_id)
        return True

    # --------------------------- Proposals ---------------------------

    async def submit_for_review(
        self,
        title: str,
        description: str,
        proposal_type: str,
        *,
        urgency: str = "normal",
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Submit a proposal for parliamentary review.
        Returns the proposal_id.
        """
        proposal_id = f"prop_{_utcnow().strftime('%Y%m%d_%H%M%S')}_{len(self._active)}"
        proposal = ReviewProposal(
            proposal_id=proposal_id,
            title=title,
            description=description,
            proposal_type=proposal_type,
            urgency=urgency,
        )

        async with self._lock:
            self._active[proposal_id] = proposal

        assigned = await self._assign_reviewers(proposal, context)
        await self._notify_reviewers(proposal, assigned)
        logger.info("Submitted proposal %s (%s) for review", proposal_id, proposal_type)
        return proposal_id

    async def _assign_reviewers(self, proposal: ReviewProposal, context: Optional[Dict[str, Any]]) -> List[str]:
        assigned: List[str] = []

        # Chairs / heavier weights for constitutional & policy
        if proposal.proposal_type in ("constitutional", "policy"):
            for mid, m in self._members.items():
                if "chair" in mid or m.weight > 1.1:
                    if m.active and mid not in assigned:
                        assigned.append(mid)

        # Domain expertise
        if context:
            for need in context.get("required_expertise", []):
                for mid, m in self._members.items():
                    if m.active and need in m.expertise and mid not in assigned:
                        assigned.append(mid)

        # Ensure minimum headcount
        min_req = {"constitutional": 5, "policy": 4, "operational": 3}.get(proposal.proposal_type, 3)
        if len(assigned) < min_req:
            for mid, m in self._members.items():
                if m.active and mid not in assigned:
                    assigned.append(mid)
                    if len(assigned) >= min_req:
                        break
        return assigned

    async def _notify_reviewers(self, proposal: ReviewProposal, assigned: List[str]) -> None:
        payload = {
            "type": "PARLIAMENT_REVIEW_ASSIGNED",
            "proposal": proposal.to_dict(),
            "assigned_reviewers": assigned,
            "deadline": _iso(proposal.deadline),
        }
        await self.event_bus.publish("PARLIAMENT_NOTIFICATION", payload, correlation_id=generate_correlation_id())

    # --------------------------- Voting ---------------------------

    async def cast_vote(self, proposal_id: str, member_id: str, vote: VoteType, *, rationale: str = "") -> bool:
        async with self._lock:
            proposal = self._active.get(proposal_id)
            if not proposal:
                logger.error("Proposal %s not found", proposal_id)
                return False

            member = self._members.get(member_id)
            if not member or not member.active:
                logger.error("Member %s not found or inactive", member_id)
                return False

            vr = VoteRecord(vote=vote, rationale=rationale, weight=member.weight)
            proposal.votes[member_id] = vr
            member.record_vote(proposal_id, vr)

        logger.info("Member %s voted %s on proposal %s", member_id, vote.value, proposal_id)
        await self._maybe_finalize(proposal_id)
        return True

    async def _maybe_finalize(self, proposal_id: str) -> None:
        async with self._lock:
            proposal = self._active.get(proposal_id)
            if not proposal:
                return

            if not proposal.votes:
                return

            approve_w = sum(v.weight for v in proposal.votes.values() if v.vote is VoteType.APPROVE)
            reject_w = sum(v.weight for v in proposal.votes.values() if v.vote is VoteType.REJECT)
            total_w = sum(v.weight for v in proposal.votes.values())
            if total_w <= 0.0:
                return

            approval_ratio = approve_w / total_w
            rejection_ratio = reject_w / total_w
            threshold = self._thresholds.get(proposal.proposal_type, 0.60)

            active_members = sum(1 for m in self._members.values() if m.active)
            participation_ratio = (len(proposal.votes) / active_members) if active_members else 0.0

            now = _utcnow()
            reached_deadline = now >= proposal.deadline

            # Decision criteria
            outcome: Optional[str] = None
            final_ratio: float = 0.0

            if participation_ratio >= self._participation_floor:
                if approval_ratio >= threshold:
                    outcome, final_ratio = "approved", approval_ratio
                elif rejection_ratio >= self._rejection_floor:
                    outcome, final_ratio = "rejected", rejection_ratio

            if outcome is None and reached_deadline:
                # On deadline, prefer the side with higher weight; if tie, needs_revision
                if approval_ratio > rejection_ratio:
                    outcome, final_ratio = "approved_by_deadline", approval_ratio
                elif rejection_ratio > approval_ratio:
                    outcome, final_ratio = "rejected_by_deadline", rejection_ratio
                else:
                    outcome, final_ratio = "needs_revision", rejection_ratio

        if outcome:
            await self._finalize(proposal_id, outcome, final_ratio)

    async def _finalize(self, proposal_id: str, outcome: str, final_ratio: float) -> None:
        async with self._lock:
            proposal = self._active.pop(proposal_id, None)
            if not proposal:
                return

            proposal.status = (
                ReviewStatus.APPROVED
                if outcome.startswith("approved")
                else (ReviewStatus.REJECTED if outcome.startswith("rejected") else ReviewStatus.NEEDS_REVISION)
            )

            record = {
                "proposal": proposal.to_dict(),
                "outcome": outcome,
                "final_ratio": final_ratio,
                "finalized_at": _iso(_utcnow()),
                "votes": {mid: vr.to_dict() for mid, vr in proposal.votes.items()},
            }
            self._history.append(record)

        # Telemetry: Experience for MLT
        exp = Experience(
            type="DEMOCRATIC_REVIEW",
            component_id="parliament",
            context={
                "proposal_type": proposal.proposal_type,
                "urgency": proposal.urgency,
                "participation_ratio": min(1.0, len(proposal.votes) / max(1, sum(1 for m in self._members.values() if m.active))),
            },
            outcome={"decision": outcome, "approval_ratio": final_ratio, "vote_count": len(proposal.votes)},
            success_score=final_ratio if outcome.startswith("approved") else (1.0 - final_ratio),
            timestamp=_utcnow(),
        )
        self.memory_core.store_experience(exp)

        # Notify decision
        await self.event_bus.publish(
            "PARLIAMENT_DECISION",
            {
                "proposal_id": proposal.proposal_id,
                "outcome": outcome,
                "final_ratio": final_ratio,
                "votes": {mid: vr.to_dict() for mid, vr in proposal.votes.items()},
            },
            correlation_id=generate_correlation_id(),
        )
        logger.info("Finalized proposal %s: %s (ratio=%.3f)", proposal.proposal_id, outcome, final_ratio)

    # --------------------------- Event Handlers ---------------------------

    async def _on_review_request(self, event: Mapping[str, Any]) -> None:
        """Handle 'GOVERNANCE_NEEDS_REVIEW' events."""
        payload = event.get("payload", {}) if isinstance(event, Mapping) else {}
        decision_id = str(payload.get("decision_id", "unknown"))
        rationale = str(payload.get("rationale", ""))[:500]

        await self.submit_for_review(
            title=f"Review Required: {decision_id}",
            description=f"Governance decision requires parliamentary review. {rationale}",
            proposal_type="policy",
            urgency="normal",
        )

    async def _on_vote_cast(self, event: Mapping[str, Any]) -> None:
        """Handle 'PARLIAMENT_VOTE_CAST' events."""
        payload = event.get("payload", {}) if isinstance(event, Mapping) else {}
        proposal_id = str(payload.get("proposal_id", ""))
        member_id = str(payload.get("member_id", ""))
        vote_str = str(payload.get("vote", ""))  # expect one of VoteType values
        rationale = str(payload.get("rationale", ""))

        try:
            vote = VoteType(vote_str)
        except Exception:
            logger.error("Invalid vote type: %s", vote_str)
            return

        await self.cast_vote(proposal_id, member_id, vote, rationale=rationale)

    # --------------------------- Queries ---------------------------

    def get_active_proposals(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self._active.values()]

    def get_proposal_status(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        p = self._active.get(proposal_id)
        if not p:
            return None
        return {
            **p.to_dict(),
            "votes": {mid: vr.to_dict() for mid, vr in p.votes.items()},
            "discussion": list(p.discussion),
        }

    def get_member_stats(self) -> Dict[str, Any]:
        active_count = sum(1 for m in self._members.values() if m.active)
        total_votes = sum(len(m.vote_history) for m in self._members.values())
        return {
            "total_members": len(self._members),
            "active_members": active_count,
            "total_votes_cast": total_votes,
            "active_proposals": len(self._active),
            "completed_reviews": len(self._history),
        }

    def get_members(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self._members.values()]
