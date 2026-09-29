import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# On Streamlit Community Cloud the keys come from the app's Secrets box (st.secrets);
# locally there is no secrets.toml, so they come from .env.
_HAS_SECRETS = st.secrets.load_if_toml_exists()


def _get(name):
    if _HAS_SECRETS and name in st.secrets:
        return st.secrets[name]
    return os.getenv(name)


# ponytail: missing keys are None for now; fail loudly once the code actually uses them
GEMINI_API_KEY = _get("GEMINI_API_KEY")
PINECONE_API_KEY = _get("PINECONE_API_KEY")
PINECONE_INDEX = _get("PINECONE_INDEX")
DATABASE_URL = _get("DATABASE_URL")
