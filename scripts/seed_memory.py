"""Seed the graph memory from past Telegram conversations stored in MongoDB.

Walks the `conversations` collection, groups consecutive user→assistant turn
pairs into text episodes, and ingests each via MemoryService.add_episode.

Usage:
    # Show what would be ingested, don't write to the graph
    python -m scripts.seed_memory --dry-run --limit 20

    # Actually seed, cap at 100 conversations
    python -m scripts.seed_memory --limit 100

    # Seed everything
    python -m scripts.seed_memory

Run from project root: `python -m scripts.seed_memory ...`
Requires GRAPHITI_MEMORY_ENABLED=true to actually ingest; in --dry-run it
only reads MongoDB and prints episodes, so the memory service stays off.
"""
import argparse
import asyncio
import datetime
import hashlib
import os
import sys
from typing import Iterator, List, Optional, Tuple

# Make `src.*` importable when running via `python -m scripts.seed_memory`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.bot.utils.config import config  # noqa: E402
from src.bot.utils.mongodb import MongoDB  # noqa: E402
from src.bot.services.memory_service import MemoryService  # noqa: E402
from src.bot.models.conversation import Conversation, Message  # noqa: E402


def iter_conversations(limit: Optional[int]) -> Iterator[Conversation]:
    """Stream all conversations from MongoDB, newest first."""
    coll = MongoDB.get_collection("conversations")
    cursor = coll.find({}).sort("updated_at", -1)
    if limit:
        cursor = cursor.limit(limit)
    for doc in cursor:
        try:
            yield Conversation.from_dict(doc)
        except Exception as e:
            print(f"  ! skipping malformed conversation doc: {e}", file=sys.stderr)


def extract_episodes(conversation: Conversation) -> List[Tuple[str, datetime.datetime, str]]:
    """Group consecutive (user, assistant) turns into episode blobs.

    Returns list of (episode_text, reference_time, source_description).
    Skips tool-only assistant turns, system messages, and empty content.
    """
    episodes: List[Tuple[str, datetime.datetime, str]] = []
    msgs: List[Message] = conversation.messages

    i = 0
    while i < len(msgs):
        msg = msgs[i]
        if msg.role != "user" or not (msg.content or "").strip():
            i += 1
            continue
        user_text = msg.content.strip()
        user_ts = msg.timestamp
        # Find the next assistant text message within the next few messages
        assistant_text = ""
        assistant_ts = user_ts
        j = i + 1
        while j < len(msgs) and j - i <= 5:
            am = msgs[j]
            if am.role == "assistant" and (am.content or "").strip():
                assistant_text = am.content.strip()
                assistant_ts = am.timestamp
                break
            j += 1
        if assistant_text:
            episode = f"User asked: {user_text}\nAssistant answered: {assistant_text}"
            source = (
                f"telegram-seed user_id={conversation.user_id} "
                f"chat_id={conversation.chat_id} topic_id={conversation.topic_id}"
            )
            episodes.append((episode, assistant_ts, source))
            i = j + 1
        else:
            i += 1

    return episodes


async def seed(dry_run: bool, limit: Optional[int], throttle_s: float) -> None:
    memory = MemoryService()

    if not dry_run:
        if not config.graphiti_memory_enabled:
            print("ERROR: GRAPHITI_MEMORY_ENABLED is not true. Use --dry-run or set the env var.",
                  file=sys.stderr)
            sys.exit(2)
        await memory.initialize()

    coll_count_doc = MongoDB.get_collection("conversations").count_documents({})
    print(f"Conversations in MongoDB: {coll_count_doc}")

    total_eps = 0
    conv_seen = 0
    for conv in iter_conversations(limit):
        conv_seen += 1
        episodes = extract_episodes(conv)
        if not episodes:
            continue
        for ep_text, ref_time, source in episodes:
            total_eps += 1
            ep_preview = ep_text.replace("\n", " ")[:120]
            if dry_run:
                print(f"[dry-run] {source}  ({len(ep_text)} chars)")
                print(f"   {ep_preview}{'...' if len(ep_text) > 120 else ''}")
            else:
                name = f"seed-{hashlib.md5(ep_text.encode('utf-8')).hexdigest()[:8]}"
                result = await memory.add_episode(
                    text=ep_text,
                    source_description=source,
                    reference_time=ref_time,
                    name=name,
                )
                print(f"[ingested] id={result}  {source}  ({len(ep_text)} chars)")
                await asyncio.sleep(throttle_s)

    if not dry_run:
        await memory.close()

    print(f"\nDone. conversations scanned: {conv_seen}, episodes: {total_eps}"
          f"{' (dry run)' if dry_run else ''}.'")


def main():
    parser = argparse.ArgumentParser(description="Seed graph memory from past conversations.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be ingested; do not write to the graph.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max conversations to scan (default: all).")
    parser.add_argument("--throttle", type=float, default=0.2,
                        help="Seconds to sleep between ingests (default: 0.2).")
    args = parser.parse_args()
    asyncio.run(seed(args.dry_run, args.limit, args.throttle))


if __name__ == "__main__":
    main()
