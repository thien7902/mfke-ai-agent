"""Ingest one or more text/markdown files into the graph memory.

Usage:
    python -m scripts.ingest_file /path/to/runbook.md
    python -m scripts.ingest_file /path/to/docs/*.md
    python -m scripts.ingest_file a.txt b.md c.txt
    python -m scripts.ingest_file --dir /root/.holmes/runbook-scripts

Each file becomes one episode. Skips binary/unreadable files. Requires
GRAPHITI_MEMORY_ENABLED=true (default) so MemoryService actually ingests.

Run from project root: `python -m scripts.ingest_file <paths...>`
"""
import argparse
import asyncio
import glob
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.bot.utils.config import config  # noqa: E402
from src.bot.services.memory_service import MemoryService  # noqa: E402

# Extensions we'll attempt to read as plain text. Anything else is skipped
# (use /addmemory via Telegram for pdf/docx/xlsx — those need extractors).
_TEXT_EXT = {".txt", ".md", ".markdown", ".rst", ".log", ".yaml", ".yml", ".json"}


def resolve_files(args) -> list[Path]:
    """Expand globs, dirs, and explicit paths into a flat list of files."""
    files: list[Path] = []
    for entry in args.paths:
        # Expand ~ and globs
        expanded = os.path.expanduser(entry)
        matches = glob.glob(expanded, recursive=True)
        if not matches:
            # User passed an explicit path that doesn't match a glob —
            # surface the error rather than silently skip.
            print(f"  ! no match for: {entry}", file=sys.stderr)
            continue
        for m in matches:
            p = Path(m)
            if p.is_dir():
                for sub in p.rglob("*"):
                    if sub.is_file() and sub.suffix.lower() in _TEXT_EXT:
                        files.append(sub)
            elif p.is_file() and p.suffix.lower() in _TEXT_EXT:
                files.append(p)
            elif p.is_file():
                print(f"  ? skipping non-text file: {p}", file=sys.stderr)
    # De-dup while preserving order
    seen = set()
    unique = []
    for f in files:
        rp = f.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(f)
    return unique


def read_file(path: Path) -> str:
    """Read file content as UTF-8. Returns empty string on failure."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  ! read failed for {path}: {e}", file=sys.stderr)
        return ""


async def ingest(files: list[Path]) -> None:
    if not config.graphiti_memory_enabled:
        print("ERROR: GRAPHITI_MEMORY_ENABLED is false. Set to true to ingest.",
              file=sys.stderr)
        sys.exit(2)

    memory = MemoryService()
    await memory.initialize()
    count_ok = 0
    count_fail = 0
    try:
        for f in files:
            content = read_file(f)
            if not content.strip():
                print(f"  ? skipping empty file: {f}")
                continue
            name = f"ingest-{hashlib.md5(content.encode('utf-8')).hexdigest()[:8]}"
            result = await memory.add_episode(
                text=content,
                source_description=f"ingest_file path={f.resolve()}",
                name=name,
            )
            if result:
                count_ok += 1
                print(f"[ok] {f}  ({len(content)} chars)  id={result}")
            else:
                count_fail += 1
                print(f"[fail] {f}  ({len(content)} chars)")
    finally:
        await memory.close()

    print(f"\nDone. ok={count_ok} fail={count_fail}")


def main():
    parser = argparse.ArgumentParser(description="Ingest text/markdown files into graph memory.")
    parser.add_argument("paths", nargs="+", help="Files or glob patterns to ingest.")
    parser.add_argument("--dir", action="store_true",
                        help="Treat each path as a directory and ingest all text files within.")
    args = parser.parse_args()

    files = resolve_files(args)
    if not files:
        print("No files to ingest.", file=sys.stderr)
        sys.exit(1)
    print(f"Ingesting {len(files)} file(s)...")
    asyncio.run(ingest(files))


if __name__ == "__main__":
    main()
