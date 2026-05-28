import datetime
import json
from pathlib import Path

import streamlit as st

FEEDBACK_FILE = Path(__file__).parent.parent / "feedback.json"


# ── Persistence ───────────────────────────────────────────────────────────────

def load_feedbacks() -> list[dict]:
    if FEEDBACK_FILE.exists():
        with open(FEEDBACK_FILE, encoding="utf-8") as f:
            return json.load(f).get("feedbacks", [])
    return []


def save_feedback(entry: dict) -> None:
    feedbacks = load_feedbacks()
    for i, fb in enumerate(feedbacks):
        if fb["id"] == entry["id"]:
            feedbacks[i] = entry
            break
    else:
        feedbacks.append(entry)
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump({"feedbacks": feedbacks}, f, ensure_ascii=False, indent=2)


def delete_feedback(fb_id: str) -> None:
    feedbacks = [f for f in load_feedbacks() if f["id"] != fb_id]
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump({"feedbacks": feedbacks}, f, ensure_ascii=False, indent=2)


# ── Widgets ───────────────────────────────────────────────────────────────────

def _show_submitted_feedback(rated: dict) -> None:
    emoji  = "👍" if rated["rating"] == 1 else "👎"
    suffix = f"  ·  *{rated['comment']}*" if rated.get("comment") else ""
    st.caption(f"{emoji} Merci pour votre retour !{suffix}")


def _show_pending_feedback(msg: dict) -> None:
    msg_id = msg["id"]
    st.caption("*Cette réponse vous a-t-elle été utile ?*")
    feedback_val = st.feedback("thumbs", key=f"fb_{msg_id}")

    if feedback_val is None:
        return

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
            "rating":    feedback_val,
            "comment":   comment.strip(),
        }
        save_feedback(entry)
        st.session_state.rated_messages[msg_id] = {
            "rating":  feedback_val,
            "comment": comment.strip(),
        }
        st.rerun()


def show_feedback_widget(msg: dict) -> None:
    msg_id = msg.get("id")
    if not msg_id:
        return
    rated = st.session_state.rated_messages.get(msg_id)
    if rated is not None:
        _show_submitted_feedback(rated)
    else:
        _show_pending_feedback(msg)


# ── Sidebar stats ─────────────────────────────────────────────────────────────

def show_sidebar_stats() -> None:
    feedbacks = load_feedbacks()

    if not feedbacks:
        st.info("Aucun retour enregistré pour l'instant.")
        return

    total = len(feedbacks)
    pos   = sum(1 for f in feedbacks if f.get("rating") == 1)
    neg   = total - pos
    pct   = int(pos / total * 100) if total else 0

    st.metric("Retours totaux", total)
    c1, c2 = st.columns(2)
    c1.metric("👍 Positifs", pos)
    c2.metric("👎 Négatifs", neg)
    st.progress(pct / 100, text=f"{pct} % de satisfaction")

    st.subheader("Derniers retours")
    for fb in reversed(feedbacks[-5:]):
        emoji   = "👍" if fb.get("rating") == 1 else "👎"
        ts      = fb.get("timestamp", "")[:16].replace("T", " ")
        q       = fb.get("question", "")
        q_short = q[:60] + "…" if len(q) > 60 else q
        note    = f"\n> *{fb['comment']}*" if fb.get("comment") else ""
        st.markdown(f"{emoji} **{ts}**  \n{q_short}{note}")
        st.divider()
