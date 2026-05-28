import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_mistralai import ChatMistralAI
from mistralai.client import Mistral

from .vector_store import FAISSVectorStore

load_dotenv()

INDEX_DIR       = os.environ.get("INDEX_DIR",        "faiss_index")
LLM_MODEL       = os.environ.get("LLM_MODEL",        "mistral-large-latest")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", 0.3))
SEARCH_TOP_K    = int(os.environ.get("SEARCH_TOP_K", 5))


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


def build_rag_chain():
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY manquante — ajoutez-la dans votre fichier .env")
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
            "par recherche sémantique). Réponds en français en te basant UNIQUEMENT sur "
            "ces extraits.\n"
            "Pour chaque événement pertinent, mentionne son titre, sa date et son lieu.\n"
            "RÈGLE ABSOLUE : si les événements fournis ne répondent pas directement à la "
            "question posée, n'en cite AUCUN et indique clairement que la base ne contient "
            "pas d'événement correspondant. Ne suggère jamais d'alternatives ni "
            "d'événements tangentiellement liés.\n\n"
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
