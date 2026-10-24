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

    # Graphiti writes go to the graph named after group_id (default: "global"),
    # NOT the driver's _database ("default_db"). Clone the driver to point at
    # the same graph graphiti writes to, otherwise counts come back empty.
    from src.bot.utils.config import config as bot_config
    group_id = bot_config.graphiti_group_id
    driver = memory._graphiti.driver
    if driver._database != group_id:
        driver = driver.clone(database=group_id)

    async def _run(cypher):
        """FalkorDriver.execute_query returns (records_list, columns, metadata).
        Normalize to just the records list. Returns [] when execute_query
        returns None (e.g. "Index already exists" suppression path)."""
        result = await driver.execute_query(cypher)
        if result is None:
            return []
        if isinstance(result, tuple):
            return result[0]
        return result

    try:
        # Graphiti 0.31 uses label "Episodic" (not "EpisodeEntity")
        records = await _run("MATCH (n:Episodic) RETURN count(n) AS c")
        for r in records:
            c = r['c'] if isinstance(r, dict) else r.get('c')
            print(f"Episodic count: {c}")

        records = await _run("MATCH (n:Entity) RETURN count(n) AS c")
        for r in records:
            c = r['c'] if isinstance(r, dict) else r.get('c')
            print(f"Entity count: {c}")

        records = await _run("MATCH ()-[r]->() RETURN count(r) AS c")
        for r in records:
            c = r['c'] if isinstance(r, dict) else r.get('c')
            print(f"Edge count: {c}")

        # Last 10 episodes
        try:
            records = await _run(
                "MATCH (n:Episodic) "
                "RETURN n.name AS name, n.group_id AS gid, n.source_description AS src "
                "ORDER BY n.created_at DESC LIMIT 10"
            )
            if records:
                print("\nLast 10 episodes:")
                for r in records:
                    get = (r.get if isinstance(r, dict) else lambda k: getattr(r, k, None))
                    print(f"  - {get('name')!r} gid={get('gid')} src={get('src')}")
            else:
                print("No episode records returned.")
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
