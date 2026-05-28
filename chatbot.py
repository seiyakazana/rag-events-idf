import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from app.rag import build_rag_chain
from app.feedback import show_feedback_widget, show_sidebar_stats

# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Assistant Événements ÎdF",
    page_icon="🎭",
    layout="centered",
)
st.title("🎭 Assistant Événements Île-de-France")
st.caption(
    "Posez vos questions sur les événements publics disponibles "
    "en Île-de-France (emploi, formation, entrepreneuriat…)."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📊 Retours utilisateurs")
    show_sidebar_stats()
    st.divider()
    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state.messages       = []
        st.session_state.rated_messages = {}
        st.rerun()

# ── Build RAG chain ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Chargement de l'index FAISS…")
def _cached_rag_chain():
    return build_rag_chain()

chain = _cached_rag_chain()

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rated_messages" not in st.session_state:
    st.session_state.rated_messages = {}

# ── Chat history ──────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
    if msg["role"] == "assistant":
        show_feedback_widget(msg)

# ── New input ─────────────────────────────────────────────────────────────────

if question := st.chat_input("Votre question sur les événements…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche et génération en cours…"):
            response = chain.invoke(question)
        st.markdown(response)

    msg_record = {
        "role":     "assistant",
        "content":  response,
        "id":       str(uuid.uuid4()),
        "question": question,
    }
    st.session_state.messages.append(msg_record)
    show_feedback_widget(msg_record)
