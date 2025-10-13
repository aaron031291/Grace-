"""
DLQ and Replay CLI for Grace Worker Queues
"""

import argparse
import asyncio
import redis.asyncio as redis
import json

parser = argparse.ArgumentParser(description="Grace DLQ Replay CLI")
parser.add_argument(
    "--queue", required=True, help="Queue name (e.g. ingestion, embeddings, media)"
)
parser.add_argument(
    "--action", choices=["list", "replay"], required=True, help="Action: list or replay"
)
parser.add_argument(
    "--redis-url", default="redis://localhost:6379", help="Redis connection URL"
)


async def main():
    args = parser.parse_args()
    r = redis.from_url(args.redis_url, decode_responses=True)
    dlq_key = f"grace:dlq:{args.queue}"
    queue_key = f"grace:queue:{args.queue}"

    if args.action == "list":
        items = await r.lrange(dlq_key, 0, -1)
        print(f"DLQ ({dlq_key}) contains {len(items)} items:")
        for i, item in enumerate(items):
            try:
                task = json.loads(item)
            except Exception:
                task = item
            print(f"[{i}] {task}")
    elif args.action == "replay":
        items = await r.lrange(dlq_key, 0, -1)
        print(f"Replaying {len(items)} items from DLQ to {queue_key}")
        for item in items:
            await r.rpush(queue_key, item)
        await r.delete(dlq_key)
        print("Replay complete. DLQ cleared.")


if __name__ == "__main__":
    asyncio.run(main())
