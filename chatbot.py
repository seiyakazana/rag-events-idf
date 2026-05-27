import os
import sys
import uuid
import json
import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_mistralai import ChatMistralAI

sys.path.insert(0, str(Path(__file__).parent))
from vector_store import FAISSVectorStore

load_dotenv()

INDEX_DIR       = os.environ.get("INDEX_DIR",        "faiss_index")
LLM_MODEL       = os.environ.get("LLM_MODEL",        "mistral-large-latest")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", 0.3))
SEARCH_TOP_K    = int(os.environ.get("SEARCH_TOP_K", 5))
FEEDBACK_FILE   = Path(__file__).parent / "feedback.json"


# ── Feedback persistence ──────────────────────────────────────────────────────

def load_feedbacks() -> list[dict]:
    """Load all feedback entries from disk."""
    if FEEDBACK_FILE.exists():
        with open(FEEDBACK_FILE, encoding="utf-8") as f:
            return json.load(f).get("feedbacks", [])
    return []


def save_feedback(entry: dict) -> None:
    """Append or update a feedback entry on disk (matched by id)."""
    feedbacks = load_feedbacks()
    for i, fb in enumerate(feedbacks):
        if fb["id"] == entry["id"]:
            feedbacks[i] = entry
            break
    else:
        feedbacks.append(entry)
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump({"feedbacks": feedbacks}, f, ensure_ascii=False, indent=2)


# ── RAG chain ─────────────────────────────────────────────────────────────────

def format_docs(docs) -> str:
    if not docs:
        return "Aucun événement trouvé."
    parts = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        parts.append(
            f"Événement {i} : {m.get('title', 'Sans titre')}\n"
            f"  Date    : {m.get('date_start', 'inconnue')}\n"
            f"  Lieu    : {m.get('location', 'inconnu')}\n"
            f"  Détail  : {doc.page_content[:400]}"
        )
    return "\n\n".join(parts)


@st.cache_resource(show_spinner="Chargement de l'index FAISS…")
def build_rag_chain():
    from mistralai.client import Mistral

    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)
    store = FAISSVectorStore.load_local(INDEX_DIR, client=client)
    total_events = len(store._documents)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Tu es un assistant spécialisé dans la recommandation d'événements publics "
            "en Île-de-France.\n"
            f"La base de données contient {total_events} événements au total.\n"
            "Pour chaque question, tu reçois les événements les plus pertinents (extraits "
            "par recherche sémantique). Réponds en français en te basant sur ces extraits. "
            "Pour chaque événement pertinent, mentionne son titre, sa date et son lieu. "
            "Si aucun événement ne correspond à la question, dis-le clairement.\n\n"
            "Événements pertinents :\n{context}",
        ),
        ("human", "{question}"),
    ])

    def retrieve_and_format(question: str) -> str:
        docs = store.similarity_search(question, k=SEARCH_TOP_K)
        return format_docs(docs)

    llm = ChatMistralAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, api_key=api_key)

    return (
        {"context": RunnableLambda(retrieve_and_format), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


# ── Feedback widget ───────────────────────────────────────────────────────────

def show_feedback_widget(msg: dict) -> None:
    """Render the thumbs rating widget below an assistant message."""
    msg_id = msg.get("id")
    if not msg_id:
        return

    rated = st.session_state.rated_messages.get(msg_id)

    if rated is not None:
        # Already submitted — show a compact confirmation
        emoji  = "👍" if rated["rating"] == 1 else "👎"
        suffix = f"  ·  *{rated['comment']}*" if rated.get("comment") else ""
        st.caption(f"{emoji} Merci pour votre retour !{suffix}")
    else:
        # Not yet rated — show the interactive widget
        st.caption("*Cette réponse vous a-t-elle été utile ?*")
        feedback_val = st.feedback("thumbs", key=f"fb_{msg_id}")

        if feedback_val is not None:
            # Thumb was clicked → ask for optional comment + confirm button
            comment = st.text_input(
                "Commentaire (optionnel) :",
                key=f"comment_{msg_id}",
                placeholder="Qu'est-ce qui pourrait être amélioré ?",
                label_visibility="collapsed",
            )
            if st.button("✅ Envoyer le retour", key=f"submit_{msg_id}", type="primary"):
                entry = {
                    "id":        msg_id,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "question":  msg.get("question", ""),
                    "answer":    msg["content"],
                    "rating":    feedback_val,   # 1 = 👍, 0 = 👎
                    "comment":   comment.strip(),
                }
                save_feedback(entry)
                st.session_state.rated_messages[msg_id] = {
                    "rating":  feedback_val,
                    "comment": comment.strip(),
                }
                st.rerun()


# ── Sidebar feedback statistics ───────────────────────────────────────────────

def show_sidebar_stats() -> None:
    """Display aggregate feedback stats in the sidebar."""
    feedbacks = load_feedbacks()

    if not feedbacks:
        st.info("Aucun retour enregistré pour l'instant.")
        return

    total = len(feedbacks)
    pos   = sum(1 for f in feedbacks if f.get("rating") == 1)
    neg   = total - pos
    pct   = int(pos / total * 100) if total else 0

    # Summary metrics
    st.metric("Retours totaux", total)
    c1, c2 = st.columns(2)
    c1.metric("👍 Positifs", pos)
    c2.metric("👎 Négatifs", neg)
    st.progress(pct / 100, text=f"{pct} % de satisfaction")

    # Last 5 feedbacks
    st.subheader("Derniers retours")
    for fb in reversed(feedbacks[-5:]):
        emoji  = "👍" if fb.get("rating") == 1 else "👎"
        ts     = fb.get("timestamp", "")[:16].replace("T", " ")
        q      = fb.get("question", "")
        q_short = q[:60] + "…" if len(q) > 60 else q
        note   = f"\n> *{fb['comment']}*" if fb.get("comment") else ""
        st.markdown(f"{emoji} **{ts}**  \n{q_short}{note}")
        st.divider()


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

chain = build_rag_chain()

# ── Session state initialisation ──────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Maps msg_id → {"rating": 0|1, "comment": str}
if "rated_messages" not in st.session_state:
    st.session_state.rated_messages = {}

# ── Render chat history ───────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
    if msg["role"] == "assistant":
        show_feedback_widget(msg)

# ── Handle new user input ─────────────────────────────────────────────────────

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
        "id":       str(uuid.uuid4()),   # unique ID for feedback tracking
        "question": question,
    }
    st.session_state.messages.append(msg_record)
    # Render the feedback widget immediately after the new response
    show_feedback_widget(msg_record)
