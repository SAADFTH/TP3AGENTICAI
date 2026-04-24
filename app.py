import streamlit as st
import os
import datetime
from rag_agent import run_rag_agent
from rag_system import rag_system
from memory import get_chat_history, get_all_sessions
from dotenv import load_dotenv

# Chargement de la configuration
load_dotenv()

# Configuration de la page Streamlit
st.set_page_config(page_title="RAG Agentic AI Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 RAG Agentic Chatbot")

# --- Initialisation de la Session ---

if "session_id" not in st.session_state:
    st.session_state.session_id = f"Chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Barre latérale : Style ChatGPT ---

with st.sidebar:
    st.header("💬 Historique des Chats")
    
    # Bouton "Nouveau Chat"
    if st.button("➕ Nouveau Chat", use_container_width=True):
        st.session_state.session_id = f"Chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Liste des sessions précédentes
    existing_sessions = get_all_sessions()
    
    if existing_sessions:
        st.caption("Conversations récentes")
        for session in reversed(existing_sessions): # On montre les plus récents en premier
            # On utilise un bouton pour chaque session pour imiter ChatGPT
            if st.button(f"🗨️ {session}", key=f"btn_{session}", use_container_width=True, 
                         type="secondary" if session != st.session_state.session_id else "primary"):
                st.session_state.session_id = session
                st.session_state.messages = [] # Forcer le rechargement
                st.rerun()
    else:
        st.info("Aucun historique pour le moment.")

    st.divider()
    
    # Options supplémentaires
    with st.expander("⚙️ Paramètres & Documents"):
        if st.button("🗑️ Effacer la session actuelle", use_container_width=True):
            history = get_chat_history(st.session_state.session_id)
            history.clear()
            st.session_state.messages = []
            st.success("Historique effacé !")
            st.rerun()
            
        st.divider()
        st.header("📂 Gestion des Documents")
        uploaded_files = st.file_uploader("Ajouter des documents (.txt, .pdf)", type=["txt", "pdf"], accept_multiple_files=True)
        
        if st.button("Indexation des Documents", use_container_width=True):
            if uploaded_files:
                if not os.path.exists("data"):
                    os.makedirs("data")
                    
                for uploaded_file in uploaded_files:
                    with open(os.path.join("data", uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                with st.spinner("Indexation en cours..."):
                    rag_system.refresh_index()
                    st.success("Documents indexés !")
            else:
                st.warning("Veuillez d'abord télécharger des documents.")

        if st.button("Réinitialiser l'Index", use_container_width=True):
            with st.spinner("Réinitialisation..."):
                rag_system.refresh_index()
                st.success("Index réinitialisé !")

# --- Chat Principal ---

# Chargement de l'historique depuis SQLite si session_state.messages est vide
if not st.session_state.messages:
    history = get_chat_history(st.session_state.session_id)
    st.session_state.messages = []
    for msg in history.messages:
        role = "user" if msg.type == "human" else "assistant"
        st.session_state.messages.append({"role": role, "content": msg.content})

# Affichage du message de bienvenue si la session est toujours vide
if not st.session_state.messages:
    st.info(f"✨ **Nouvelle session commencée : {st.session_state.session_id}**")
    st.markdown("""
    Bienvenue ! Je suis votre assistant intelligent. Vous pouvez :
    - Me poser des questions sur vos documents.
    - Me demander des informations récentes sur le web.
    - Me faire exécuter du code Python.
    
    *Comment puis-je vous aider aujourd'hui ?*
    """)

# Affichage des messages de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrée utilisateur
if prompt := st.chat_input("Posez votre question ici..."):
    # Ajouter le message utilisateur à l'historique local
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Réponse de l'agent
    with st.chat_message("assistant"):
        with st.spinner("Réflexion en cours..."):
            # Exécution de l'agent RAG avec la session_id sélectionnée
            response = run_rag_agent(prompt, session_id=st.session_state.session_id)
            st.markdown(response)
    
    # Ajouter la réponse de l'assistant à l'historique local
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- Pied de page ---
st.divider()
st.caption(f"ID Session : **{st.session_state.session_id}** | Propulsé par LangChain, OpenAI et Streamlit. TP3 - Agentic AI.")
