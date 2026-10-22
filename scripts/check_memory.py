"""Diagnostic script: list what's actually stored in the Graphiti memory.

Usage (run inside the bot container):
    docker exec -it telegram-bot python3 -m scripts.check_memory
    docker exec -it telegram-bot python3 -m scripts.check_memory "pod crash"
    docker exec -it telegram-bot python3 -m scripts.check_memory --count

Without args: print episode count + last N episodes + a sample search.
With a query: print top-k search results for that query (k=5).

This lets you verify whether /addmemory actually wrote data into FalkorDB
and whether the search pipeline returns anything for a known-good query.
"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.bot.services.memory_service import MemoryService


async def print_stats(memory: MemoryService) -> None:
    """Show episode count + recent episodes."""
    print("=== Stats ===")
    if not memory._initialized or memory._graphiti is None:
        print("memory not initialized — add_episode can't have written anything")
        return

    # Graphiti exposes the driver; query episodes through it.
    driver = memory._graphiti.driver
    try:
        # Use the FalkorDriver session to query the Graphiti schema — _all
        # episodes live as nodes with label='Episode'. The exact attribute
        # name may vary by graphiti version; we use the fact that the
        # FalkorDriver exposes a `session()` async context manager.
        async with driver.session() as sess:
            # Count episodes
            result = await sess.run(
                "MATCH (n:EpisodeEntity) RETURN count(n) AS c"
            )
            records = await result.to_list()
            for r in records:
                print(f"EpisodeEntity count: {r['c']}")

            # Last 10 episodes by createdAt if available, else by name
            try:
                result = await sess.run(
                    "MATCH (n:EpisodeEntity) "
                    "RETURN n.name AS name, n.group_id AS gid, n.source_description AS src "
                    "ORDER BY n.created_at DESC LIMIT 10"
                )
                records = await result.to_list()
                if records:
                    print("\nLast 10 episodes:")
                    for r in records:
                        print(f"  - {r.get('name')!r} gid={r.get('gid')} src={r.get('src')}")
                else:
                    print("No episode records returned (label may differ).")
            except Exception as e:
                print(f"could not list recent episodes: {e}")
    except Exception as e:
        print(f"stat query failed: {type(e).__name__}: {e}")


async def search(memory: MemoryService, query: str, k: int) -> None:
    """Run a search and print results."""
    print(f"\n=== Search: {query!r} (k={k}) ===")
    facts = await memory.retrieve_relevant(query, k=k)
    if not facts:
        print("No facts returned. Possibilities:")
        print("  1. /addmemory never actually wrote anything (check add_episode logs)")
        print("  2. Embedder connection is broken (check memory_retrieval_failed log)")
        print("  3. Vector mismatch: dim or model changed since ingest")
        print("  4. The query just doesn't match anything kept")
    else:
        print(f"Got {len(facts)} fact(s):")
        for i, f in enumerate(facts, 1):
            print(f"  {i}. {f}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Graphiti memory.")
    parser.add_argument("query", nargs="?", default=None,
                        help="Optional search query; without it, print stats only.")
    parser.add_argument("--count", action="store_true",
                        help="Just print episode count and exit.")
    parser.add_argument("-k", type=int, default=5,
                        help="Max search results (default 5).")
    args = parser.parse_args()

    memory = MemoryService()
    try:
        await memory.initialize()
    except Exception as e:
        print(f"INIT FAILED: {type(e).__name__}: {e}")
        return

    if not memory._initialized:
        print("memory did NOT initialize — /addmemory cannot have written anything.")
        print("Check 'memory_init_failed_at_step' logs.")
        return

    print("memory initialized OK")
    if args.count:
        await print_stats(memory)
        return

    if args.query:
        await search(memory, args.query, args.k)
    else:
        await print_stats(memory)
        await search(memory, "test", args.k)


if __name__ == "__main__":
    asyncio.run(main())
