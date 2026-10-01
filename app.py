import json

import pandas as pd
import streamlit as st

import config
import db
import qa
import vectors
from retrieval_eval import BM25_MODEL, GOLD_PATH, BM25Retriever, dense_search, load_gold, score, section_id

st.set_page_config(page_title="מערכת התקנון", layout="wide")
st.markdown(
    "<style>.stApp, .stApp * { direction: rtl; text-align: right; } .stApp code, .stApp pre { direction: ltr; text-align: left; }</style>",
    unsafe_allow_html=True,
)
st.title("מערכת התקנון")


def check_neon():
    with db.connect() as conn:
        return f"מחובר · {db.count_sections(conn)} סעיפים שמורים"


def check_pinecone():
    pc = vectors.connect()
    if not pc.has_index(config.PINECONE_INDEX):
        raise RuntimeError(f"האינדקס '{config.PINECONE_INDEX}' לא קיים")
    return f"מחובר · {vectors.count_vectors(pc.Index(config.PINECONE_INDEX))} וקטורים באינדקס"


def check_gemini():
    if not config.GEMINI_API_KEY:
        raise RuntimeError("המפתח חסר")
    return "המפתח קיים"


def status_line(label, check):
    try:
        st.markdown(f"✅ **{label}:** {check()}")
    except Exception as e:
        st.markdown(f"❌ **{label}:**")
        st.code(str(e) or type(e).__name__, language=None)


# ---------- Retrieval Eval ----------

DENSE_MODEL = f"Pinecone/{config.PINECONE_INDEX} · {vectors.EMBED_MODEL} ({vectors.DIMENSION}d)"


@st.cache_resource(ttl=600, show_spinner=False)
def load_corpus():
    """BM25 index over all Neon chunks + section texts for the drill-down (refreshed every 10 minutes)."""
    with db.connect() as conn:
        rows = conn.execute("SELECT number, chunk_index, text FROM sections ORDER BY number, chunk_index").fetchall()
    return BM25Retriever(rows), {r["number"]: r["text"] for r in rows if r["chunk_index"] == 0}


@st.cache_resource(show_spinner=False)
def pinecone_index():
    return vectors.get_index(vectors.connect(), create=False)


@st.cache_data(show_spinner=False)
def embed_query(query, model):
    # `model` is part of the cache key, so switching embedding model never serves stale vectors
    return vectors.embed([query], task_type="RETRIEVAL_QUERY")[0]


def gold_to_df(gold):
    return pd.DataFrame(
        [{"query": g["query"], "relevant_section_ids": ", ".join(g["relevant_section_ids"])} for g in gold],
        columns=["query", "relevant_section_ids"],
    )


def df_to_gold(df):
    items = [
        {"query": str(r["query"] or "").strip(), "relevant_section_ids": [x for x in str(r["relevant_section_ids"] or "").split(",") if x.strip()]}
        for _, r in df.iterrows()
        if str(r["query"] or "").strip()
    ]
    return load_gold(json.dumps(items, ensure_ascii=False))  # same validation as the file


def gold_editor():
    if "gold_df" not in st.session_state:
        gold = load_gold(GOLD_PATH.read_text(encoding="utf-8")) if GOLD_PATH.exists() else []
        st.session_state.gold_df, st.session_state.gold_version = gold_to_df(gold), 0

    uploaded = st.file_uploader("העלאת סט שאילתות (JSON)", type="json")
    if uploaded and uploaded.file_id != st.session_state.get("gold_upload_id"):
        st.session_state.gold_upload_id = uploaded.file_id
        try:
            st.session_state.gold_df = gold_to_df(load_gold(uploaded.getvalue().decode("utf-8")))
            st.session_state.gold_version += 1  # new editor key, so it shows the uploaded rows
        except (ValueError, UnicodeDecodeError) as e:
            st.error(f"הקובץ לא נטען: {e}")

    st.caption("מזהי סעיפים מופרדים בפסיקים, למשל: 5.4, 14.2")
    edited = st.data_editor(
        st.session_state.gold_df, num_rows="dynamic", width="stretch",
        key=f"gold_editor_{st.session_state.gold_version}",
    )
    try:
        gold = df_to_gold(edited)
    except ValueError as e:
        st.error(f"סט השאילתות לא תקין: {e}")
        return None

    c1, c2 = st.columns(2)
    if c1.button(f"💾 שמירה ל-{GOLD_PATH.parent.name}/{GOLD_PATH.name}"):
        GOLD_PATH.parent.mkdir(exist_ok=True)
        GOLD_PATH.write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")
        st.success(f"נשמרו {len(gold)} שאילתות")  # on Streamlit Cloud the disk resets on restart: download to keep it
    c2.download_button("⬇️ הורדת סט השאילתות", json.dumps(gold, ensure_ascii=False, indent=2), "gold_queries.json", "application/json")
    return gold


def run_eval(gold, k):
    bm25, _ = load_corpus()
    index = pinecone_index()
    retrievers = [
        ("BM25", BM25_MODEL, lambda q: bm25.search(q, k)),
        ("Dense", DENSE_MODEL, lambda q: dense_search(index, embed_query(q, vectors.EMBED_MODEL), k)),
    ]
    rows, hits = [], {}
    bar = st.progress(0.0, text="מריץ שאילתות...")
    for i, g in enumerate(gold):
        for name, model, search in retrievers:
            ranked = search(g["query"])
            rank, rr, recall = score([sid for sid, _ in ranked], g["relevant_section_ids"], k)
            hits[(i, name)] = ranked
            rows.append({
                "retriever": name, "model": model, "k": k, "query": g["query"],
                "relevant": ", ".join(g["relevant_section_ids"]),
                "retrieved": ", ".join(sid for sid, _ in ranked),
                "first_relevant_rank": rank, "reciprocal_rank": rr, "recall": recall,
            })
        bar.progress((i + 1) / len(gold), text=f"{i + 1}/{len(gold)} · {g['query']}")
    bar.empty()
    return {"k": k, "gold": gold, "rows": rows, "hits": hits}


def show_results(res):
    k, df = res["k"], pd.DataFrame(res["rows"])
    mrr, recall = f"MRR@{k}", f"Recall@{k}"
    summary = df.groupby(["retriever", "model"], as_index=False).agg(
        **{mrr: ("reciprocal_rank", "mean"), recall: ("recall", "mean"), "queries": ("query", "count")}
    )
    st.subheader("השוואה")
    st.dataframe(summary, hide_index=True, width="stretch", column_config={
        mrr: st.column_config.NumberColumn(format="%.3f"), recall: st.column_config.NumberColumn(format="%.3f"),
    })
    chart = summary.melt(id_vars="retriever", value_vars=[mrr, recall], var_name="metric", value_name="score")
    st.bar_chart(chart, x="metric", y="score", color="retriever", stack=False)

    st.download_button("⬇️ הורדת התוצאות (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                       f"retrieval_eval_k{k}.csv", "text/csv")
    with st.expander("כל התוצאות לפי שאילתה"):
        st.dataframe(df, hide_index=True, width="stretch")

    st.subheader("פירוט לפי שאילתה")
    _, texts = load_corpus()
    names = sorted(df["retriever"].unique())
    for i, g in enumerate(res["gold"]):
        q_rows = {r["retriever"]: r for r in res["rows"][i * len(names):(i + 1) * len(names)]}
        failed = any(r["reciprocal_rank"] == 0 for r in q_rows.values())
        label = " · ".join(f"{n} RR {q_rows[n]['reciprocal_rank']:.2f}" for n in names)
        with st.expander(f"{'❌' if failed else '✅'} {g['query']} — {label}"):
            relevant = set(g["relevant_section_ids"])
            for col, n in zip(st.columns(len(names)), names):
                with col:
                    r = q_rows[n]
                    st.markdown(f"**{n}** · RR = {r['reciprocal_rank']:.3f} · Recall = {r['recall']:.2f}")
                    st.caption(r["model"])
                    st.dataframe(pd.DataFrame([
                        {"rank": rank, "section": sid, "relevant": "✅" if sid in relevant else "",
                         "score": round(s, 3), "text": texts.get(sid, "")[:90]}
                        for rank, (sid, s) in enumerate(res["hits"][(i, n)], 1)
                    ]), hide_index=True, width="stretch")
                    missing = sorted(relevant - {sid for sid, _ in res["hits"][(i, n)]})
                    if missing:
                        st.markdown(f"לא נמצאו בטופ-{k}: **{', '.join(missing)}**")


def eval_tab():
    st.markdown("מדידת איכות האחזור מול סט שאילתות עם תשובות ידועות: **Dense** (חיפוש וקטורי ב-Pinecone) מול **BM25** (חיפוש מילים על הסעיפים ב-Neon).")
    gold = gold_editor()
    k = st.radio("top_k", [5, 10, 20], index=1, horizontal=True)
    if st.button("▶️ הרצת הערכה", type="primary", disabled=not gold):
        try:
            st.session_state.eval_results = run_eval(gold, k)
        except Exception as e:
            st.error(f"ההרצה נכשלה: {e}")
    if "eval_results" in st.session_state:
        show_results(st.session_state.eval_results)


# ---------- Free question ----------

# Measured on this regulation (see the commit that added this tab): 22 paraphrased gold questions, 14 exact-phrase
# questions and 16 short exact-term queries, scored on whether the right section is ranked first.
FINDING = (
    "בבדיקה על התקנון הזה embeddings ניצחו בשאלות מנוסחות במילים אחרות (21 מול 10 מתוך 22 במקום הראשון), "
    "ו-BM25 ניצח כשהשאלה היא מונח מדויק וקצר כמו מספר, שעה או שם טופס (14.5 מול 11 מתוך 16), "
    "ואילו כשהמונח המדויק מופיע בתוך שאלה מלאה שתי השיטות השתוו (14 מתוך 14)."
)
METHODS = {
    "embeddings": ("Embeddings (cosine)", f"{vectors.EMBED_MODEL} · {vectors.DIMENSION}d"),
    "bm25": ("BM25 (מילות מפתח)", BM25_MODEL),
}


@st.cache_resource(show_spinner=False)
def sections_index():
    return qa.SectionsIndex()


def show_answer(method, r):
    title, model = METHODS[method]
    st.subheader(title)
    st.caption(f"{model} · תשובה: {qa.ANSWER_MODEL}")
    if "error" in r:
        st.error(f"נכשל: {r['error']}")
        return
    st.dataframe(pd.DataFrame([
        {"#": i, "סעיף": c["section"], "מקטע": c["chunk_index"], "ציון": round(s, 3), "פרק": c["chapter"]}
        for i, (c, s) in enumerate(r["hits"], 1)
    ]), hide_index=True, width="stretch")
    st.markdown(r["answer"])
    st.caption(f"⏱️ אחזור {r['retrieval_ms']:.0f} ms · תשובה {r['answer_ms']:.0f} ms · סה״כ {r['total_ms']:.0f} ms")


def free_question_tab():
    st.info(FINDING)
    with st.form("free_question"):
        question = st.text_input("שאלה על התקנון")
        submitted = st.form_submit_button("שאל", type="primary")
    if submitted and question.strip():
        results = {}
        with st.spinner("מאחזר ועונה..."):
            try:
                index = sections_index()
            except Exception as e:
                st.error(f"טעינת {qa.INDEX_PATH.name} נכשלה: {e}")
                return
            for method in METHODS:
                try:
                    results[method] = qa.ask(index, question.strip(), method)
                except Exception as e:
                    results[method] = {"error": str(e)[:300]}
        st.session_state.free_answers = results
    if "free_answers" in st.session_state:
        right, left = st.columns(2)  # the page is RTL, so the first column renders on the right
        with left:
            show_answer("embeddings", st.session_state.free_answers["embeddings"])
        with right:
            show_answer("bm25", st.session_state.free_answers["bm25"])


status_tab, retrieval_tab, free_tab = st.tabs(["מצב המערכת", "Retrieval Eval", "שאלה חופשית"])
with retrieval_tab:
    eval_tab()
with free_tab:
    free_question_tab()
with status_tab:
    with st.spinner("בודק חיבורים..."):
        status_line("חיבור ל-Neon", check_neon)
        status_line("חיבור ל-Pinecone", check_pinecone)
        status_line("מפתח Gemini", check_gemini)
