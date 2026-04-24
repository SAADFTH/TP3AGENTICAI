# Agent Conversationnel LangChain Avancé (RAG & Streamlit)

Ce projet implémente un agent conversationnel sophistiqué avec une architecture middleware complète, des outils avancés, un système RAG et une interface Streamlit.

## Fonctionnalités Avancées

### Architecture Middleware
L'agent utilise une cascade de middlewares pour la sécurité, la performance et le contrôle :
- **Dynamic Model Selection** : Bascule automatiquement entre `GPT-3.5-Turbo` et `GPT-4-Turbo` selon la complexité de la requête.
- **Dynamic Prompt Adaptation** : Adapte le persona de l'agent selon le contexte détecté.
- **GuardRails** : Filtrage bidirectionnel (Input/Output) pour bloquer les mots-clés interdits et prévenir les fuites de données sensibles.
- **Tool Error Handling** : Gestion centralisée des erreurs d'outils avec des messages d'erreur conviviaux.
- **Human In The Loop (HITL)** : Interception manuelle obligatoire pour les actions critiques comme l'exécution de code Python.

### Système RAG (Retrieval-Augmented Generation)
L'agent peut consulter des documents locaux (PDF, TXT) pour enrichir ses réponses :
- **Indexation locale** : Utilise FAISS et OpenAI Embeddings pour stocker et rechercher des informations dans vos documents.
- **Recherche de similarité** : Retrouve les fragments de documents les plus pertinents pour répondre à vos questions.

### Interface Streamlit
Une interface web moderne pour interagir avec l'agent :
- **Chat en direct** : Interface de chat intuitive.
- **Gestion des documents** : Téléchargez et indexez de nouveaux documents directement depuis l'interface.
- **Historique des sessions** : Les conversations sont persistantes grâce à SQLite.

## Structure du Projet
- `app.py` : Interface utilisateur Streamlit.
- `rag_agent.py` : Cœur de l'agent RAG avec ses outils et middlewares.
- `rag_system.py` : Logique d'indexation et de recherche documentaire.
- `agent.py` : Version console de l'agent (TP1/TP2).
- `middleware.py` : Logique métier des middlewares.
- `tools.py` : Configuration des outils et du gestionnaire d'erreurs.
- `memory.py` : Persistance de la mémoire via SQLite.
- `data/` : Dossier contenant vos documents à indexer.

## Installation
1. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
2. Configurez votre fichier `.env` :
   ```env
   OPENAI_API_KEY="VOTRE_CLÉ_OPENAI"
   TAVILY_API_KEY="VOTRE_CLÉ_TAVILY"
   ```

## Utilisation
### Version Web (Recommandé)
Lancez l'interface Streamlit :
```bash
streamlit run app.py
```

### Version Console
Lancez l'agent en mode texte :
```bash
python agent.py
```

## Tests
```bash
pytest test_middleware.py
```
