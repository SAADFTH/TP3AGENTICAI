import sqlite3
import os
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Chemin de la base de données SQLite pour la persistance
DB_PATH = "sqlite:///chat_history.db"
DB_FILE = "chat_history.db"

def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    """Retourne l'historique des messages pour une session donnée, stocké en SQLite."""
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=DB_PATH
    )

def get_all_sessions():
    """Récupère la liste de tous les session_id existants dans la base SQLite."""
    if not os.path.exists(DB_FILE):
        return []
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        # SQLChatMessageHistory utilise par défaut la table 'message_store'
        cursor.execute("SELECT DISTINCT session_id FROM message_store")
        sessions = [row[0] for row in cursor.fetchall()]
        conn.close()
        return sessions
    except Exception as e:
        print(f"Erreur lors de la récupération des sessions : {e}")
        return []
