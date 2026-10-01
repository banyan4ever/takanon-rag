"""Run: .venv/bin/python test_retrieval_eval.py"""
from retrieval_eval import BM25Retriever, load_gold, score, section_id

# MRR / Recall
assert score(["a", "b", "c"], ["b"], 3) == (2, 0.5, 1.0)
assert score(["a", "b", "c"], ["c", "x"], 3) == (3, 1 / 3, 0.5)
assert score(["a", "b", "c"], ["c"], 2) == (None, 0.0, 0.0)  # relevant hit beyond k doesn't count
assert score(["x", "y"], ["x", "y"], 5) == (1, 1.0, 1.0)

# ids: section numbers or Pinecone vector ids
assert section_id("neon-10.8-0") == "10.8" and section_id(" 7.1 ") == "7.1"
assert load_gold('[{"query": "q", "relevant_section_ids": ["neon-5.4-0", 7.1]}]') == [
    {"query": "q", "relevant_section_ids": ["5.4", "7.1"]}
]
for bad in ("{}", "[{}]", '[{"query": "q", "relevant_section_ids": []}]', "not json"):
    try:
        load_gold(bad)
        raise AssertionError(f"accepted {bad}")
    except ValueError:
        pass

# BM25: best match first; two chunks of one section collapse into one hit
bm = BM25Retriever([
    {"number": "1.1", "text": "שכר לימוד ותשלומים"},
    {"number": "2.1", "text": "בחינות ומועדים מיוחדים"},
    {"number": "2.1", "text": "מועד מיוחד לבחינה"},
    {"number": "3.1", "text": "ספרייה"},
])
hits = bm.search("מועד מיוחד", 5)
assert hits[0][0] == "2.1" and [sid for sid, _ in hits].count("2.1") == 1

print("OK")
