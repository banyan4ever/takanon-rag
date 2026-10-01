"""Load data/takanon.pdf into the Neon `sections` table: one row per chunk, with a Gemini summary.

Run: .venv/bin/python load_takanon.py   (safe to re-run: existing rows are updated, never duplicated;
Gemini is only called for new or changed chunks)
"""
import json
import logging
import re
import sys
import time
from pathlib import Path

import pymupdf
from google import genai
from google.genai import errors, types

import config
import db

HERE = Path(__file__).parent
PDF_PATH = HERE / "data" / "takanon.pdf"
LOG_PATH = HERE / "load_takanon.log"
MAX_CHARS = 800
MODEL = "gemini-flash-lite-latest"

CHAPTER_RE = re.compile(r"פרק (\d+) — (.+)")
SECTION_RE = re.compile(r"סעיף (\d+(?:\.\d+)*)")
APPENDIX_RE = re.compile(r"נספח [א-ת]")
SENTENCE_END_RE = re.compile(r"(?<=[.!?;])\s+")
MIRRORED_NUMBER_RE = re.compile(r"([.,]*)(\d(?:[\d.:/-]*\d)?)(-?)")  # e.g. ".10.1" -> "10.1.", "24-" -> "-24"
PUNCT = set(",.;:!?)")

log = logging.getLogger("load_takanon")


# ---------- PDF -> lines ----------

def join_words(words):
    """Rebuild a Hebrew line from right-to-left words, re-attaching punctuation the PDF stored as separate words."""
    # ponytail: covers the cases seen in this PDF; a few rare ones (e.g. ".)") stay slightly off, harmless for search
    out = []
    for w in words:
        if out and set(w) <= PUNCT:
            out[-1] += w
            continue
        m = MIRRORED_NUMBER_RE.fullmatch(w)
        if m and (m[1] or m[3]):
            w = m[2] + m[1]
            if m[3] and out:
                out[-1] += "-" + w
                continue
        if out and out[-1] == "(":
            out[-1] += w
        else:
            out.append(w)
    return " ".join(out)


def read_pdf(path):
    """Return (page_count, [(page_number, line), ...]) in reading order.

    The PDF stores Hebrew in visual order, so plain text extraction scrambles numbers and punctuation.
    Instead, each line is rebuilt from its words sorted right to left by position.
    """
    lines = []
    with pymupdf.open(path) as doc:
        for page_no, page in enumerate(doc, 1):
            rows = {}
            for x0, y0, x1, y1, word, block, line, _ in page.get_text("words"):
                rows.setdefault((block, line), []).append((x1, y0, word))
            for words in sorted(rows.values(), key=lambda ws: min(y for _, y, _ in ws)):
                text = join_words(w for _, _, w in sorted(words, reverse=True))
                if text.strip():
                    lines.append((page_no, text.strip()))
        return doc.page_count, lines


# ---------- lines -> sections -> chunks ----------

def parse_sections(lines):
    """Group lines into sections by the regulation's numbering ("פרק N — title", "סעיף N.M").

    Appendices ("נספח א") have no number and come out with number=None, so validation skips and logs them.
    """
    sections, current, chapter_no, chapter = [], None, 0, ""
    for page, line in lines:
        m = CHAPTER_RE.fullmatch(line)
        # a real chapter heading is the next chapter in sequence; anything else is a wrapped sentence like "...בפרק 14 — פטור"
        if m and int(m[1]) == chapter_no + 1:
            chapter_no, chapter, current = int(m[1]), m[2].strip(), None
        elif m := SECTION_RE.fullmatch(line):
            current = {"number": m[1], "chapter": chapter, "lines": []}
            sections.append(current)
        elif line == "נספחים":
            chapter, current = "נספחים", None
        elif APPENDIX_RE.fullmatch(line):
            current = {"number": None, "chapter": chapter, "label": line, "lines": []}
            sections.append(current)
        elif current is not None:  # lines before the first section (document title) are ignored
            current["lines"].append((page, line))
    return sections


def chunk_text(text, max_chars=MAX_CHARS):
    """Split on sentence ends into chunks of <= max_chars, repeating the last sentence as overlap.

    A single sentence longer than max_chars becomes its own chunk rather than being cut.
    """
    if len(text) <= max_chars:
        return [text]
    chunks, current = [], []
    for sentence in SENTENCE_END_RE.split(text):
        if current and len(" ".join(current + [sentence])) > max_chars:
            chunks.append(" ".join(current))
            current = current[-1:]  # overlap
            if len(" ".join(current + [sentence])) > max_chars:
                current = []
        current.append(sentence)
    chunks.append(" ".join(current))
    return chunks


def make_chunks(sections):
    chunks = []
    for s in sections:
        text, starts = "", []  # starts: (offset in text, page) for each line
        for page, line in s["lines"]:
            starts.append((len(text) + 1 if text else 0, page))
            text = f"{text} {line}" if text else line
        text = " ".join(text.split())
        for i, chunk in enumerate(chunk_text(text) if text else [""]):
            offset = text.find(chunk)
            page = next((p for o, p in reversed(starts) if o <= offset), starts[0][1] if starts else None)
            chunks.append({"number": s["number"], "chapter": s["chapter"], "text": chunk, "page": page,
                           "chunk_index": i, "label": s.get("label")})
    return chunks


def validation_error(chunk, seen):
    if not chunk["number"]:
        return "אין מספר סעיף"
    if not chunk["text"].strip():
        return "טקסט ריק"
    if (chunk["number"], chunk["chunk_index"]) in seen:
        return "צמד סעיף+מקטע כפול"
    return None


# ---------- LLM ----------

PROMPT = """You receive one chunk of a college regulations document (Hebrew).
Return ONLY a JSON object with exactly these keys:
  "section_number": the section number as given,
  "chapter_title": the chapter title as given,
  "summary": a one-sentence summary of the chunk, in Hebrew.
No prose, no markdown fences, nothing before or after the JSON."""

FIELDS = {"section_number", "chapter_title", "summary"}


def summarize(client, chunk):
    """Return the model's JSON as a dict; raise ValueError if the reply isn't exactly the JSON object we asked for."""
    contents = f"פרק: {chunk['chapter']}\nסעיף: {chunk['number']}\n\n{chunk['text']}"
    for attempt in range(1, 6):
        try:
            reply = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=PROMPT,
                    response_mime_type="application/json",
                    temperature=0,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            break
        except errors.APIError as e:  # rate limit / server hiccup: back off and retry
            if attempt == 5:
                raise ValueError(f"שגיאת API: {e}") from e
            time.sleep(5 * attempt)
    raw = reply.text or ""
    try:
        data = json.loads(raw)  # deliberately strict: fences or prose around the JSON fail here
    except json.JSONDecodeError:
        raise ValueError(f"התשובה אינה JSON תקין: {raw[:200]!r}")
    if not isinstance(data, dict) or set(data) != FIELDS:
        raise ValueError(f"שדות שגויים בתשובה: {raw[:200]!r}")
    if not all(isinstance(data[f], str) and data[f].strip() for f in FIELDS):
        raise ValueError(f"שדה ריק או לא טקסט: {raw[:200]!r}")
    return data


# ---------- main ----------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per Gemini request otherwise
    if not PDF_PATH.exists():
        sys.exit(f"❌ הקובץ {PDF_PATH} לא נמצא")
    if not config.GEMINI_API_KEY:
        sys.exit("❌ GEMINI_API_KEY חסר ב-.env")
    try:
        conn = db.connect()
    except Exception as e:
        sys.exit(f"❌ החיבור למסד הנתונים נכשל: {e}")

    pages, lines = read_pdf(PDF_PATH)
    chunks = make_chunks(parse_sections(lines))
    print(f"📄 נקראו {pages} עמודים, נוצרו {len(chunks)} מקטעים. מעבד...")

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    counts = {"inserted": 0, "updated": 0, "unchanged": 0, "skipped": 0}
    seen = set()
    # ponytail: rows for chunks that disappear from the PDF are left in place; add a cleanup pass if sections ever get removed
    with conn:
        # summaries already in Neon, reused while a chunk's text is unchanged (saves Gemini calls and avoids re-embedding)
        existing = {(r["number"], r["chunk_index"]): r for r in
                    conn.execute("SELECT number, chunk_index, text, summary FROM sections").fetchall()}
        for i, chunk in enumerate(chunks, 1):
            name = f"סעיף {chunk['number'] or chunk['label']} מקטע {chunk['chunk_index']} (עמוד {chunk['page']})"
            if reason := validation_error(chunk, seen):
                log.warning("דילוג על %s: %s", name, reason)
                counts["skipped"] += 1
                continue
            seen.add((chunk["number"], chunk["chunk_index"]))
            old = existing.get((chunk["number"], chunk["chunk_index"]))
            if old and old["text"] == chunk["text"] and old["summary"]:
                summary = old["summary"]
            else:
                try:
                    meta = summarize(client, chunk)
                except ValueError as e:
                    log.warning("דילוג על %s: %s", name, e)
                    counts["skipped"] += 1
                    continue
                if meta["section_number"] != chunk["number"]:
                    log.info("%s: המודל החזיר מספר סעיף %r, נשמר המספר מהמסמך", name, meta["section_number"])
                summary = meta["summary"]
            row = db.upsert_section(conn, chunk["number"], chunk["chapter"], chunk["text"],
                                    chunk_index=chunk["chunk_index"], page=chunk["page"], summary=summary)
            counts["unchanged" if row is None else "inserted" if row["inserted"] else "updated"] += 1
            print(f"  {i}/{len(chunks)}", end="\r", flush=True)

    print()
    print(f"📄 עמודים שנקראו:   {pages}")
    print(f"✂️  מקטעים שנוצרו:   {len(chunks)}")
    print(f"➕ נוספו:           {counts['inserted']}")
    print(f"🔄 עודכנו:          {counts['updated']}")
    print(f"⏸️  ללא שינוי:       {counts['unchanged']}")
    print(f"⏭️  דולגו:           {counts['skipped']}  (פירוט ב-{LOG_PATH.name})")


if __name__ == "__main__":
    main()
