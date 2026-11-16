"""Run inside the bot container to measure real distilled facts token cost.

Usage (pretty quick if you're already running the bot):
    docker exec -it <bot_container> python3 /app/scripts/measure_distilled_tokens.py

It:
1. Pulls the injectable facts list (post-cosine-dedup) the same way the live
   system prompt builder does, so the count matches what the LLM actually sees.
2. Builds the exact block HolmesService._build_holmes_messages would append
   (header + bulleted facts) and writes it to a temp file.
3. Counts tokens two ways:
   - LiteLLM's count_tokens (provider-agnostic via the model the bot uses)
   - Approx char/3.5 (Vietnamese heuristic, for sanity)
4. Prints total chars, total tokens, header overhead, average per fact.

No network calls, no LLM API calls, no writes outside /tmp. Idempotent.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

# --- Imports kept tiny so the script starts fast even inside container --- #
def _import_modules():
    from src.bot.services.distilled_fact_store import DistilledFactStore
    from src.bot.services.holmes_service import HolmesService  # only to read header
    from src.bot.utils.mongodb import MongoDB
    return DistilledFactStore, HolmesService, MongoDB


def main():
    DistilledFactStore, HolmesService, MongoDB = _import_modules()

    store = DistilledFactStore()
    facts = store.get_injectable_facts()
    count = len(facts)
    print(f"== Distilled facts (post-dedup, injectable) ==")
    print(f"  count: {count}")

    texts = [f.get("text", "") for f in facts if f.get("text")]
    total_chars = sum(len(t) for t in texts)
    bytes_utf8 = sum(len(t.encode("utf-8")) for t in texts)
    bullet_overhead = count * 2  # "- "
    header_len = len(HolmesService._DISTILLED_FACTS_HEADER)
    print(f"  total chars (fact bodies): {total_chars}")
    print(f"  total bytes (utf-8 bodies): {bytes_utf8}")
    print(f"  bullets overhead (chars): {bullet_overhead}")
    print(f"  header length (chars): {header_len}")
    print(f"  block total chars (header + bullets + bodies): "
          f"{header_len + bullet_overhead + total_chars}")
    print(f"  avg chars/fact: {total_chars // max(count, 1)}")
    print(f"  max chars/fact: {max((len(t) for t in texts), default=0)}")

    # Build block exactly like HolmesService does.
    bullet_lines = "\n".join(
        f"- {f.get('text', '')}".strip() for f in facts if f.get("text")
    )
    block = (HolmesService._DISTILLED_FACTS_HEADER + bullet_lines).strip() if bullet_lines else ""

    # Write block + raw facts JSON to /tmp for offline inspection.
    with tempfile.NamedTemporaryFile(
        "w", delete=False, suffix=".txt", prefix="distilled_block_"
    ) as f:
        f.write(block)
        block_path = f.name
    with tempfile.NamedTemporaryFile(
        "w", delete=False, suffix=".json", prefix="distilled_facts_"
    ) as f:
        json.dump(
            [{"text": t} for t in texts], f, ensure_ascii=False, indent=2,
        )
        facts_path = f.name

    # Token count via litellm, which is what the bot ends up calling.
    try:
        import litellm
        # Force a clean count without invoking the real provider.
        from holmes.core.llm import LLM
        # Use the model the bot is wired to.
        model = os.environ.get("OPENAI_MODEL") or os.environ.get("LITELLM_MODEL")
        if not model:
            print("\n== Token count (litellm) ==")
            print("  no OPENAI_MODEL/LITELLM_MODEL env, skipping litellm count")
            print("  raw dump of block saved at:")
            print(f"    {block_path}")
            print(f"  raw dump of facts saved at:")
            print(f"    {facts_path}")
            return

        # litellm.token_counter takes a model + messages. Use a single user
        # message carrying the full block so it counts tokens we control.
        tokens = litellm.token_counter(
            model=f"openai/{model}" if not model.startswith("openai/") else model,
            messages=[{"role": "system", "content": block}],
        )
        print(f"\n== Token count (litellm, model={model}) ==")
        print(f"  block tokens (system role): {tokens}")

        # Also count as user-role message to compare — Anthropic/OpenAI
        # tokenize Vietnamese slightly differently per role in some routes.
        tokens_user = litellm.token_counter(
            model=f"openai/{model}" if not model.startswith("openai/") else model,
            messages=[{"role": "user", "content": block}],
        )
        print(f"  block tokens (user role): {tokens_user}")

        # Estimate tokens per individual fact for variance.
        per_fact_tokens = []
        for t in texts:
            tk = litellm.token_counter(
                model=f"openai/{model}" if not model.startswith("openai/") else model,
                messages=[{"role": "user", "content": t}],
            )
            per_fact_tokens.append(tk)
        if per_fact_tokens:
            print(f"  tokens: min={min(per_fact_tokens)} "
                  f"max={max(per_fact_tokens)} "
                  f"sum={sum(per_fact_tokens)} "
                  f"avg={sum(per_fact_tokens) // len(per_fact_tokens)}")

        print("\n  raw dump of block saved at:")
        print(f"    {block_path}")
        print(f"  raw dump of facts saved at:")
        print(f"    {facts_path}")
    except Exception as e:
        print(f"\n== Token count failed: {type(e).__name__}: {e}")
        print("  files saved:")
        print(f"    block: {block_path}")
        print(f"    facts: {facts_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
