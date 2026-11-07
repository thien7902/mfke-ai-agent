"""One-shot script: ingest a FPT Container KB docx into MongoDB `kb_rules`.

Pipeline (offline, idempotent):
  1. Parse the docx (stdlib zipfile + ElementTree, recursive into w:tbl
     for code-block cells). Walk drawing/a:blip refs to find inline images.
  2. For each image: write zip-bytes to a tmp file and call
     `vision_service.describe_image` → captions interleaved as
     `[IMG: <caption>]` inline markers in the article text.
  3. Group paragraphs under Heading1 (`^[KB\\d{8}-\\d{3}\\]`). Inside an
     article, split on `^## ` markdown markers → sections
     (overview, environment, symptoms, resolution, code blocks).
  4. For each article, call the distill LLM (KB_DISTILL_MODEL →
     CONSOLIDATOR_MODEL fallback) with a strict-JSON prompt → 3-5
     if-then rules.
  5. Embed each rule via the OpenAI-compatible endpoint
     (Vietnamese_Embedding, 1024-dim — same as the bot's query path).
  6. Upsert into `kb_rules` by `rule_id = f"{issue_id}#{idx}"`.
     `--purge` drops the collection first (use when the docx structure
     changed enough that stale rule_ids would linger).

Usage:
    python -m scripts.ingest_kb_docx "C:/path/to/FPT Container.docx"
    python -m scripts.ingest_kb_docx --purge path.docx

Run from project root.
"""
import argparse
import json
import os
import re
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
from xml.etree import ElementTree as ET

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import openai
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    retry_if_exception_type,
)

from src.bot.services.embedder import embed_text
from src.bot.utils.config import config as bot_config
from src.bot.utils.mongodb import MongoDB
from src.bot.services import vision_service

logger = structlog.get_logger(__name__)

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
_NS = {"w": _W_NS, "a": _A_NS, "r": _R_NS, "pic": _PIC_NS}

_KB_HEADING_RE = re.compile(r"^\[KB\d{8}-\d{3}\]")
# Sub-section header inside the article body (raw text — not Word Heading2
# style, because the docx was Aspose-exported from Confluence and the
# sections are markdown text, per Explore verification).
_SECTION_RE = re.compile(r"^(#{1,4})\s+(.*)")
# "Nhờ KT/team..." chatter between KB headings — skip.
_CHATTER_RE = re.compile(r"^\s*[+]\s*Nh[oờ]\s", re.IGNORECASE)
CODE_KEYWORDS = ("kubectl", "apiVersion", "kind:", "spec:", "docker", "FROM", "helm")


# --------------------------------------------------------------------- #
# Data shapes
# --------------------------------------------------------------------- #

@dataclass
class KBArticle:
    issue_id: str
    title: str
    overview: str = ""
    environment: str = ""
    symptoms: str = ""
    resolution: str = ""
    code_blocks: List[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """What the distill LLM sees. Skip overview/environment metadata — they
        are noise for the troubleshooting-rule extractor, but the symptoms,
        resolution, and code blocks (with [IMG: ...] captions where images
        appeared inline) carry the signal.
        """
        parts = [f"# {self.issue_id} — {self.title}"]
        if self.symptoms:
            parts.append(f"## Symptoms\n{self.symptoms}")
        if self.resolution:
            parts.append(f"## Resolution\n{self.resolution}")
        if self.code_blocks:
            parts.append("## Code/Commands\n" + "\n\n".join(self.code_blocks))
        return "\n\n".join(parts)


# --------------------------------------------------------------------- #
# Docx parsing
# --------------------------------------------------------------------- #

def parse_docx(docx_path: Path, with_images: bool = True) -> List[KBArticle]:
    """Parse the docx into a list of KBArticle dicts. Walks tables
    recursively so code blocks inside table cells flow back into the
    linear text stream. Images are interleaved as `[IMG: <caption>]`
    placeholders by calling `vision_service.describe_image` per image.
    """
    with zipfile.ZipFile(docx_path) as z:
        # Build rId → media path map (relationships file).
        rels = _load_rels(z)
        media_bytes_cache: dict[str, bytes] = {
            n: z.read(n) for n in z.namelist() if n.startswith("word/media/")
        }

        with z.open("word/document.xml") as f:
            tree = ET.parse(f)
        root = tree.getroot()

        # Walk body children in order, producing a flat list of
        # (kind, text) "tokens" where kind is "text" or "image".
        body = root.find("w:body", _NS)
        if body is None:
            return []
        tokens: List[Tuple[str, str]] = []
        for child in body.iter():
            # iter() flattens recursion; we want top-level walking only.
            pass
        # Use a manual recursion so we preserve document order incl. tables.
        tokens = _walk_body(body, rels, media_bytes_cache, with_images=with_images)

    # Group tokens into articles by KB heading.
    raw_articles: List[KBArticle] = []
    current: Optional[KBArticle] = None
    current_section: str = "preamble"  # one of preamble|overview|environment|symptoms|resolution|code
    for kind, text in tokens:
        if kind == "image":
            if current is not None:
                # Inject image caption into whatever section we're in.
                _append_to_section(current, current_section, text)
            continue
        stripped = text.strip()
        if not stripped:
            continue
        if _CHATTER_RE.match(stripped):
            continue
        m = _KB_HEADING_RE.match(stripped)
        if m:
            # Flush current.
            if current is not None:
                raw_articles.append(current)
            issue_id, title = _parse_kb_heading(stripped)
            current = KBArticle(issue_id=issue_id, title=title)
            current_section = "preamble"
            continue
        if current is None:
            # Pre-article metadata (TOC, revision log) — skip.
            continue
        # Section header?
        sm = _SECTION_RE.match(stripped)
        if sm:
            heading_text = sm.group(2).lower()
            if "overview" in heading_text or "purpose" in heading_text:
                current_section = "overview"
            elif "environment" in heading_text or "scope" in heading_text:
                current_section = "environment"
            elif "symptom" in heading_text or "problem" in heading_text:
                current_section = "symptoms"
            elif "resolution" in heading_text or "step" in heading_text:
                current_section = "resolution"
            else:
                current_section = "overview"
            continue
        # `---` separator between sections → keep current_section.
        if stripped == "---":
            continue
        # Code-block heuristic.
        if _looks_like_code(stripped):
            current.code_blocks.append(stripped)
        _append_to_section(current, current_section, stripped)
    if current is not None:
        raw_articles.append(current)

    # The docx's TOC produces empty-article stubs that share the KB ID
    # AND title with the real body article. Dedup by (issue_id, title):
    # keep the one with strictly more content. Distinct titles under the
    # same ID are kept separate (per Explore's analysis — some IDs are
    # reused across genuinely different articles, often by mistake).
    # A final filter drops any article that has zero content (likely
    # a TOC stub whose body never followed).
    articles: List[KBArticle] = []
    seen: dict = {}  # (issue_id, title) → index in articles
    for art in raw_articles:
        content_size = (
            len(art.overview) + len(art.environment) + len(art.symptoms)
            + len(art.resolution) + sum(len(c) for c in art.code_blocks)
        )
        if content_size == 0:
            # Skip empty stubs outright — a TOC entry or section header
            # that produced no body. If a real article follows, it wins.
            continue
        key = (art.issue_id, _normalize_title(art.title))
        if key in seen:
            idx = seen[key]
            prev = articles[idx]
            prev_size = (
                len(prev.overview) + len(prev.environment) + len(prev.symptoms)
                + len(prev.resolution) + sum(len(c) for c in prev.code_blocks)
            )
            if content_size > prev_size:
                articles[idx] = art
        else:
            seen[key] = len(articles)
            articles.append(art)
    return articles


def _walk_body(body_el, rels: dict, media_bytes_cache: dict, with_images: bool = True) -> List[Tuple[str, str]]:
    """Walk w:body children in document order, recursing into w:tbl so code
    blocks inside table cells flow back into the linear token stream.
    """
    out: List[Tuple[str, str]] = []
    for el in body_el:
        tag = el.tag.split("}", 1)[-1]
        if tag == "p":
            out.extend(_walk_p(el, rels, media_bytes_cache, with_images=with_images))
        elif tag == "tbl":
            out.extend(_walk_tbl(el, rels, media_bytes_cache, with_images=with_images))
        # Other top-level elements (sectPr, etc.) are skipped silently.
    return out


def _walk_tbl(tbl_el, rels, media_bytes_cache, with_images: bool = True) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for tr in tbl_el.findall("w:tr", _NS):
        for tc in tr.findall("w:tc", _NS):
            for el in tc:
                tag = el.tag.split("}", 1)[-1]
                if tag == "p":
                    out.extend(_walk_p(el, rels, media_bytes_cache, with_images=with_images))
                elif tag == "tbl":
                    # Nested table — recurse.
                    out.extend(_walk_tbl(el, rels, media_bytes_cache, with_images=with_images))
    return out


def _walk_p(p_el, rels, media_bytes_cache, with_images: bool = True) -> List[Tuple[str, str]]:
    """Walk one paragraph's runs. Concatenate w:t text. Detect inline
    drawings and emit a synthetic image token at the position the image
    appears in the run order.
    """
    out: List[Tuple[str, str]] = []
    text_parts: List[str] = []
    for child in p_el.iter():
        tag = child.tag.split("}", 1)[-1]
        if tag == "t":
            if child.text:
                text_parts.append(child.text)
        elif tag == "blip" and with_images:
            embed = child.get(f"{{{_R_NS}}}embed")
            if embed and embed in rels:
                media_path = rels[embed]
                media_bytes = media_bytes_cache.get(media_path)
                if media_bytes:
                    caption = _describe_image_bytes(media_bytes, media_path)
                    if caption:
                        out.append(("image", f"[IMG: {caption}]"))
    joined = "".join(text_parts).strip()
    if joined:
        out.append(("text", joined))
    return out


def _describe_image_bytes(image_bytes: bytes, media_name: str) -> str:
    """Write image bytes to a tmp file and call vision_service.describe_image.
    Failures return "" — the parser sees empty caption and skips the
    placeholder, so a transient vision outage doesn't block ingest.
    """
    suffix = ".png"
    if media_name.lower().endswith((".jpeg", ".jpg")):
        suffix = ".jpg"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            tf.write(image_bytes)
            tmp_path = Path(tf.name)
        try:
            return vision_service.describe_image(tmp_path).strip()
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
    except Exception as e:
        logger.warning("kb_image_describe_failed", media=media_name, error=str(e))
        return ""


def _load_rels(z: zipfile.ZipFile) -> dict:
    """Return {rId: 'word/media/imageNN.png'} map."""
    try:
        with z.open("word/_rels/document.xml.rels") as f:
            tree = ET.parse(f)
    except KeyError:
        return {}
    out: dict = {}
    for rel in tree.getroot():
        rid = rel.get("Id")
        target = rel.get("Target", "")
        if rid and "media" in target:
            # rels targets are relative to word/ — normalize to word/media/...
            if not target.startswith("word/"):
                target = "word/" + target
            out[rid] = target
    return out


def _parse_kb_heading(text: str) -> Tuple[str, str]:
    """`[KB20250929-002] FKE [Known-Issue] ...` → ('KB20250929-002', full text)."""
    m = re.match(r"^\[(KB\d{8}-\d{3})\]\s*(.*)", text)
    if not m:
        return text.strip(), text.strip()
    return m.group(1), text.strip()


def _looks_like_code(text: str) -> bool:
    """Heuristic: a paragraph containing one of the kubectl/apiVersion/...
    keywords is treated as a code block. Conservative — false positives
    (prose mentioning 'kubectl') are acceptable since the distill LLM
    handles them as context.
    """
    lowered = text.lower()
    return any(kw.lower() in lowered for kw in CODE_KEYWORDS)


def _normalize_title(title: str) -> str:
    """Collapse NBSP and other Unicode whitespace to a single ASCII space so
    the TOC entry (which may use NBSP) dedups against the real body article.
    """
    if not title:
        return ""
    # \xa0 (NBSP) and other Unicode spaces → normal space, then collapse runs.
    out = []
    for ch in title:
        if ch in "\xa0           ":
            out.append(" ")
        else:
            out.append(ch)
    return " ".join("".join(out).split())


def _append_to_section(article: KBArticle, section: str, text: str) -> None:
    if section == "overview":
        article.overview = (article.overview + "\n" + text).strip() if article.overview else text
    elif section == "environment":
        article.environment = (article.environment + "\n" + text).strip() if article.environment else text
    elif section == "symptoms":
        article.symptoms = (article.symptoms + "\n" + text).strip() if article.symptoms else text
    elif section == "resolution":
        article.resolution = (article.resolution + "\n" + text).strip() if article.resolution else text
    elif section == "preamble":
        # Pre-section prose under the title (rarer in this docx, but still).
        article.symptoms = (article.symptoms + "\n" + text).strip() if article.symptoms else text
    # code_blocks are appended directly in parse_docx, not here.


# --------------------------------------------------------------------- #
# LLM distill
# --------------------------------------------------------------------- #

_DISTILL_PROMPT = (
    "You are a knowledge extractor for a Kubernetes operations knowledge base. "
    "Read the KB article title, symptoms, resolution, and code/commands below "
    "and extract 3 to 5 structured if-then rules that would help a senior "
    "engineer diagnose the same issue in the future. Each rule must be one "
    "self-contained line in this exact format:\n"
    "IF <symptom + signal patterns> THEN <root cause> -> <fix or check step>\n"
    "\n"
    "Where:\n"
    "- <symptom + signal> is concrete observables: pod states (CrashLoopBackOff, "
    "ImagePullBackOff, Pending, Evicted), event reasons (FailedScheduling, "
    "SystemOOM, DiskPressure), shoot conditions (ReconciliationFailed, "
    "ControlPlaneNotReady), log keywords. Cover the different ways this issue "
    "can manifest.\n"
    "- <root cause> is one short clause.\n"
    "- <fix or check step> is the diagnostic or remediation action.\n"
    "\n"
    "Skip purely informational Q&A about service offerings or architecture — "
    "return {\"rules\":[]} for those.\n"
    "\n"
    "Output STRICT JSON only — no prose, no markdown fences:\n"
    "{\"rules\":[{\"text\":\"IF ... THEN ... -> ...\"}]}\n"
)


@retry(
    stop=stop_after_attempt(2),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _call_distill_llm(article_text: str) -> List[str]:
    """Return list of rule text strings. Caller wraps in try/except."""
    model = bot_config.kb_distill_model or bot_config.consolidator_model
    if not model:
        return []
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return []
    api_base = os.environ.get("OPENAI_API_BASE")
    client = (
        openai.OpenAI(api_key=api_key, base_url=api_base)
        if api_base
        else openai.OpenAI(api_key=api_key)
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _DISTILL_PROMPT},
            {"role": "user", "content": article_text},
        ],
        temperature=0,
    )
    raw = (resp.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        # Strip ```json...``` / ```...``` fences.
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("kb_distill_invalid_json", error=str(e), raw_preview=raw[:200])
        return []
    if not isinstance(data, dict):
        return []
    rules = data.get("rules", [])
    if not isinstance(rules, list):
        return []
    out = []
    for r in rules:
        if isinstance(r, dict):
            t = (r.get("text") or "").strip()
            if t:
                out.append(t)
        elif isinstance(r, str):
            if r.strip():
                out.append(r.strip())
    return out


# --------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------- #

def ingest(articles: List[KBArticle], purge: bool = False) -> Tuple[int, int, int]:
    """Upsert distilled rules into kb_rules. Returns (upserted, skipped, errors)."""
    from pymongo import ASCENDING, DESCENDING
    coll = MongoDB.get_collection("kb_rules")
    try:
        coll.create_index([("rule_id", ASCENDING)], unique=True)
        coll.create_index([("issue_id", ASCENDING)])
        coll.create_index([("match_count", DESCENDING)])
    except Exception as e:
        logger.warning("kb_rules_index_failed", error=str(e))
    if purge:
        try:
            coll.delete_many({})
            logger.info("kb_rules_purged")
        except Exception as e:
            logger.warning("kb_rules_purge_failed", error=str(e))

    upserted = 0
    skipped = 0
    errors = 0
    from datetime import datetime
    for art in articles:
        try:
            rules = _call_distill_llm(art.full_text)
        except Exception as e:
            logger.warning("kb_distill_failed", issue_id=art.issue_id, error=str(e))
            errors += 1
            continue
        if not rules:
            skipped += 1
            print(f"[skip] {art.issue_id} — {art.title[:80]} (no reusable pattern)")
            continue
        for idx, rule_text in enumerate(rules, start=1):
            rule_id = f"{art.issue_id}#{idx}"
            try:
                embedding = embed_text(rule_text)
            except Exception as e:
                logger.warning("kb_embed_failed", rule_id=rule_id, error=str(e))
                errors += 1
                continue
            if not embedding:
                errors += 1
                continue
            doc = {
                "rule_id": rule_id,
                "issue_id": art.issue_id,
                "issue_title": art.title,
                "rule_text": rule_text,
                "embedding": embedding,
                "match_count": 0,
                "created_at": datetime.utcnow(),
                "last_matched_at": None,
            }
            try:
                coll.update_one(
                    {"rule_id": rule_id},
                    {"$set": doc},
                    upsert=True,
                )
                upserted += 1
            except Exception as e:
                logger.warning("kb_upsert_failed", rule_id=rule_id, error=str(e))
                errors += 1
        print(f"[ok] {art.issue_id} — {art.title[:80]} rules={len(rules)}")
    return upserted, skipped, errors


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest FPT Container KB docx into kb_rules.")
    parser.add_argument("docx", help="Path to the .docx file")
    parser.add_argument("--purge", action="store_true",
                        help="Drop kb_rules collection before re-inserting.")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip image captioning (text-only distill). Use when "
                             "VISION_MODEL endpoint is unavailable.")
    args = parser.parse_args()

    docx_path = Path(args.docx)
    if not docx_path.is_file():
        print(f"ERROR: file not found: {docx_path}", file=sys.stderr)
        return 1

    if not bot_config.kb_distill_model and not bot_config.consolidator_model:
        print("ERROR: KB_DISTILL_MODEL (or CONSOLIDATOR_MODEL fallback) is not set.", file=sys.stderr)
        return 1

    print(f"Parsing {docx_path}...")
    articles = parse_docx(docx_path, with_images=not args.no_images)
    print(f"Parsed {len(articles)} KB articles. Distilling + embedding...")

    upserted, skipped, errors = ingest(articles, purge=args.purge)
    print()
    print(f"Done. upserted={upserted} rules across {len(articles)} articles skipped={skipped} errors={errors}")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
