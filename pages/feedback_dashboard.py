"""
Feedback Dashboard — page dédiée à l'analyse des retours utilisateurs.
Accessible via le menu de navigation Streamlit (pages/).
"""

from datetime import datetime

import streamlit as st

from app.feedback import load_feedbacks, delete_feedback


def export_csv(feedbacks: list[dict]) -> str:
    """Return feedbacks as a CSV string."""
    import csv, io
    buf = io.StringIO()
    fieldnames = ["id", "timestamp", "rating", "question", "answer", "comment"]
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for fb in feedbacks:
        row = {k: fb.get(k, "") for k in fieldnames}
        row["rating"] = "👍" if row["rating"] == 1 else "👎"
        writer.writerow(row)
    return buf.getvalue()


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Tableau de bord — Retours",
    page_icon="📊",
    layout="wide",
)
st.title("📊 Tableau de bord des retours utilisateurs")
st.caption("Analyse de tous les retours collectés sur le chatbot.")

feedbacks = load_feedbacks()

if not feedbacks:
    st.info("Aucun retour enregistré. Posez des questions dans le chatbot et évaluez les réponses !")
    st.stop()

# ── KPI metrics ───────────────────────────────────────────────────────────────

total = len(feedbacks)
pos   = sum(1 for f in feedbacks if f.get("rating") == 1)
neg   = total - pos
pct   = round(pos / total * 100, 1) if total else 0
with_comment = sum(1 for f in feedbacks if f.get("comment"))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total des retours", total)
col2.metric("👍 Positifs", pos, delta=f"{pct} %")
col3.metric("👎 Négatifs", neg)
col4.metric("💬 Avec commentaire", with_comment)

st.progress(pct / 100, text=f"Taux de satisfaction : {pct} %")
st.divider()

# ── Filters ───────────────────────────────────────────────────────────────────

col_f1, col_f2 = st.columns([1, 2])
with col_f1:
    rating_filter = st.selectbox(
        "Filtrer par note :",
        options=["Tous", "👍 Positifs uniquement", "👎 Négatifs uniquement"],
    )
with col_f2:
    search_query = st.text_input("🔍 Rechercher dans les questions / commentaires :", "")

filtered = feedbacks[:]
if rating_filter == "👍 Positifs uniquement":
    filtered = [f for f in filtered if f.get("rating") == 1]
elif rating_filter == "👎 Négatifs uniquement":
    filtered = [f for f in filtered if f.get("rating") == 0]

if search_query.strip():
    q = search_query.strip().lower()
    filtered = [
        f for f in filtered
        if q in f.get("question", "").lower()
        or q in f.get("comment", "").lower()
        or q in f.get("answer", "").lower()
    ]

st.caption(f"{len(filtered)} retour(s) affiché(s)")

# ── Export ────────────────────────────────────────────────────────────────────

csv_data = export_csv(filtered)
st.download_button(
    label="⬇️ Exporter en CSV",
    data=csv_data.encode("utf-8"),
    file_name=f"retours_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv",
)

st.divider()

# ── Feedback list ─────────────────────────────────────────────────────────────

if not filtered:
    st.warning("Aucun retour ne correspond aux filtres sélectionnés.")
else:
    for fb in reversed(filtered):   # most recent first
        emoji  = "👍" if fb.get("rating") == 1 else "👎"
        ts     = fb.get("timestamp", "")[:16].replace("T", " ")
        rating_color = "green" if fb.get("rating") == 1 else "red"

        with st.expander(
            f"{emoji}  {ts}  ·  {fb.get('question', '')[:80]}…"
            if len(fb.get("question", "")) > 80
            else f"{emoji}  {ts}  ·  {fb.get('question', '')}",
            expanded=False,
        ):
            st.markdown(f"**Note :** :{rating_color}[{emoji}]")
            st.markdown(f"**Horodatage :** `{ts}`")
            st.markdown(f"**Question :**\n> {fb.get('question', '*—*')}")
            st.markdown(f"**Réponse du chatbot :**")
            st.markdown(fb.get("answer", "*—*"))
            if fb.get("comment"):
                st.markdown(f"**Commentaire utilisateur :**\n> *{fb['comment']}*")

            # Delete button
            if st.button(
                "🗑️ Supprimer ce retour",
                key=f"del_{fb['id']}",
                type="secondary",
            ):
                delete_feedback(fb["id"])
                st.success("Retour supprimé.")
                st.rerun()
