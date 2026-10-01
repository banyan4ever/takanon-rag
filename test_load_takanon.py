"""Run: .venv/bin/python test_load_takanon.py"""
from load_takanon import chunk_text, join_words, make_chunks, parse_sections, validation_error

# punctuation the PDF stores as separate / mirrored words
assert join_words(["מכלול", "ההיבטים", ",", "מתוך", "הלמידה", "."]) == "מכלול ההיבטים, מתוך הלמידה."
assert join_words(["כמפורט", "בסעיף", ".10.1"]) == "כמפורט בסעיף 10.1."
assert join_words(["ליותר", "מ", "24-", "נקודות"]) == "ליותר מ-24 נקודות"
assert join_words(["השעות", "9:00-15:00"]) == "השעות 9:00-15:00"

# long text: chunks stay <= 800, end on a sentence, and overlap by one sentence
sentences = [f"משפט מספר {i} עם קצת טקסט נוסף כדי להאריך אותו מעט." for i in range(40)]
chunks = chunk_text(" ".join(sentences))
assert len(chunks) > 1 and all(len(c) <= 800 for c in chunks)
assert all(c.endswith(".") for c in chunks)
assert all(a.split(". ")[-1] in b for a, b in zip(chunks, chunks[1:])), "missing overlap"
assert " ".join(sentences) == " ".join(chunks[0:1] + [c.split(". ", 1)[1] for c in chunks[1:]])  # nothing lost
assert chunk_text("קצר.") == ["קצר."]
assert chunk_text("א" * 900) == ["א" * 900]  # one long sentence is never cut

# headings: a wrapped "פרק 14 — ..." mid-chapter-10 is body text; appendices get no number
lines = [(1, "תקנון לימודים"), (1, "פרק 1 — כללי"), (1, "סעיף 1.1"), (1, "שורה ראשונה"),
         (2, "פרק 14 — פטור מאגרה."), (2, "פרק 2 — קבלה"), (2, "סעיף 2.1"), (3, "טקסט"),
         (3, "סעיף 2.2"), (3, "נספחים"), (3, "נספח א"), (4, "תוכן נספח")]
secs = parse_sections(lines)
assert [(s["number"], s["chapter"]) for s in secs] == [("1.1", "כללי"), ("2.1", "קבלה"), ("2.2", "קבלה"), (None, "נספחים")]
assert [l for _, l in secs[0]["lines"]] == ["שורה ראשונה", "פרק 14 — פטור מאגרה."]

chunks = make_chunks(secs)
assert [(c["number"], c["page"], c["text"]) for c in chunks][:2] == [("1.1", 1, "שורה ראשונה פרק 14 — פטור מאגרה."), ("2.1", 3, "טקסט")]
seen = set()
assert validation_error(chunks[2], seen) == "טקסט ריק"
assert validation_error(chunks[3], seen) == "אין מספר סעיף"
assert validation_error(chunks[0], {("1.1", 0)}) == "צמד סעיף+מקטע כפול"
assert validation_error(chunks[0], seen) is None

print("OK")
