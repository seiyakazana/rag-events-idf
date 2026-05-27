# 🎭 Assistant Événements Île-de-France — RAG Chatbot

Un chatbot basé sur la technique **RAG (Retrieval-Augmented Generation)** qui recommande des événements publics en Île-de-France. Il combine une recherche sémantique via FAISS et un modèle de langage Mistral pour répondre à des questions en langage naturel.

---

## 📐 Architecture

```
data_preprocessing.py  →  vectorize_and_index.py  →  chatbot.py (Streamlit)
(Récupération + Chunking) (Embeddings + Index FAISS)       ↓
                                                  pages/feedback_dashboard.py
```

Module partagé : `vector_store.py` (classe `FAISSVectorStore`, utilisée par
`vectorize_and_index.py` et `chatbot.py`).

### Pipeline de données

| Étape | Script | Rôle |
|-------|--------|------|
| 1 | `data_preprocessing.py` | Interroge l'API OpenAgenda, nettoie et découpe en chunks → `events_chunks.json` |
| 2 | `vectorize_and_index.py` | Vectorise les chunks via Mistral et construit l'index FAISS |
| 3 | `chatbot.py` | Interface Streamlit avec chaîne RAG (recherche sémantique + LLM) |
| 4 | `pages/feedback_dashboard.py` | Tableau de bord d'analyse des retours utilisateurs |

---

## 🚀 Installation

### Prérequis

- Python 3.10+
- Une clé API [Mistral AI](https://console.mistral.ai/)

### 1. Cloner le projet et créer l'environnement virtuel

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
```

### 2. Installer les dépendances

```bash
pip install streamlit langchain langchain-mistralai mistralai faiss-cpu \
            numpy pandas python-dotenv requests
```

### 3. Configurer les variables d'environnement

Copiez `.env` et renseignez votre clé API :

```env
# Clés API
MISTRAL_API_KEY=your_mistral_api_key_here

# Modèles Mistral
EMBEDDING_MODEL=mistral-embed
LLM_MODEL=mistral-large-latest
LLM_TEMPERATURE=0.3

# Chemins fichiers
CHUNKS_PATH=events_chunks.json
JSON_PATH=events_vectorized.json
INDEX_DIR=faiss_index

# Paramètres FAISS
EMBEDDING_DIM=1024
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Paramètres chatbot
SEARCH_TOP_K=5

# Paramètres API OpenAgenda
API_URL=https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records/
API_LANG=fr
API_LIMIT=100
API_REGION=Île-de-France
API_KEYWORDS=en physique
```

> ⚠️ Ne commitez jamais votre `.env` — ajoutez-le à `.gitignore`.

---

## ▶️ Utilisation

### Étape 1 — Prétraiter les événements

Ce script récupère les événements de l'API, affiche un aperçu, nettoie les données
et découpe les descriptions en chunks texte :

```bash
python data_preprocessing.py
```

Résultat : `events_chunks.json` (chunks texte, sans vecteurs)

### Étape 2 — Vectoriser et indexer

Ce script charge les chunks, génère leurs embeddings via Mistral et construit l'index FAISS :

```bash
python vectorize_and_index.py
```

Résultat : `events_vectorized.json` (chunks + vecteurs) et dossier `faiss_index/`

### Étape 3 — Lancer le chatbot

```bash
streamlit run chatbot.py
```

Ouvrir [http://localhost:8501](http://localhost:8501) dans votre navigateur.

---

## 🌐 Interface

### Chatbot (`chatbot.py`)

- **Zone de chat** : posez vos questions en langage naturel sur les événements
- **Système de feedback** : évaluez chaque réponse avec 👍 / 👎 et un commentaire optionnel
- **Sidebar** : statistiques de satisfaction en temps réel + bouton pour effacer la conversation

### Tableau de bord (`pages/feedback_dashboard.py`)

Accessible via le menu de navigation Streamlit :

- KPIs : total des retours, taux de satisfaction, nombre de commentaires
- Filtres : par note (positif/négatif) et par mot-clé
- Export CSV des retours filtrés
- Suppression individuelle d'un retour

---

## 📁 Structure du projet

```
Projet 11 - RAG/
├── .env                          # Variables d'environnement (ne pas commiter)
├── vector_store.py               # Classe FAISSVectorStore partagée
├── data_preprocessing.py         # Étape 1 : récupération, nettoyage, chunking → events_chunks.json
├── vectorize_and_index.py        # Étape 2 : vectorisation Mistral + construction index FAISS
├── chatbot.py                    # Application Streamlit principale (RAG)
├── check_packages.py             # Vérification des dépendances installées
├── test_vector_db_integrity.py   # Suite de tests pytest (intégrité des données)
├── events_chunks.json            # Chunks texte (généré par data_preprocessing.py)
├── events_vectorized.json        # Chunks vectorisés (généré par vectorize_and_index.py)
├── feedback.json                 # Retours utilisateurs (généré automatiquement)
├── faiss_index/                  # Index FAISS (généré par vectorize_and_index.py)
│   ├── index.faiss
│   └── index.pkl
└── pages/
    └── feedback_dashboard.py     # Page Streamlit du tableau de bord des retours
```

---

## 🛠️ Dépannage

### Vérifier les dépendances

```bash
python check_packages.py
```

### Erreur `MISTRAL_API_KEY not set`

Assurez-vous que le fichier `.env` est présent à la racine du projet et contient une clé valide.

### Index FAISS introuvable

Exécutez `vectorize_events.py` puis `build_faiss_index.py` avant de lancer le chatbot.

---

## 🔧 Paramètres configurables

| Variable | Défaut | Description |
|----------|--------|-------------|
| `EMBEDDING_MODEL` | `mistral-embed` | Modèle d'embeddings Mistral |
| `LLM_MODEL` | `mistral-large-latest` | Modèle LLM pour la génération |
| `LLM_TEMPERATURE` | `0.3` | Créativité des réponses (0 = déterministe) |
| `CHUNK_SIZE` | `500` | Taille max d'un chunk (en caractères) |
| `CHUNK_OVERLAP` | `50` | Chevauchement entre chunks |
| `SEARCH_TOP_K` | `5` | Nombre de documents retournés par la recherche |
| `API_REGION` | `Île-de-France` | Région filtrée dans l'API |
| `API_KEYWORDS` | `en physique` | Mots-clés filtrés dans l'API |

---

## 📦 Technologies utilisées

| Technologie | Rôle |
|-------------|------|
| [Mistral AI](https://mistral.ai/) | Embeddings (`mistral-embed`) et LLM (`mistral-large`) |
| [FAISS](https://github.com/facebookresearch/faiss) | Index vectoriel pour la recherche sémantique |
| [LangChain](https://www.langchain.com/) | Orchestration de la chaîne RAG |
| [Streamlit](https://streamlit.io/) | Interface web interactive |
| [OpenAgenda / OpenDataSoft](https://public.opendatasoft.com/) | Source de données des événements publics |
