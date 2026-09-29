import streamlit as st

import config
import db
import vectors

st.set_page_config(page_title="מערכת התקנון", layout="centered")
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


(status_tab,) = st.tabs(["מצב המערכת"])
with status_tab:
    with st.spinner("בודק חיבורים..."):
        status_line("חיבור ל-Neon", check_neon)
        status_line("חיבור ל-Pinecone", check_pinecone)
        status_line("מפתח Gemini", check_gemini)
