import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool

# Import des composants locaux
from rag_system import rag_system
from memory import get_chat_history
from middleware import AgentMiddleware
from tools import tools as other_tools

# Chargement de la configuration .env
load_dotenv()

# --- Définition de l'outil RAG ---

@tool
def search_local_docs(query: str) -> str:
    """Recherche des informations spécifiques dans les documents locaux (RAG)."""
    return rag_system.search(query)

# Liste des outils complète
rag_tools = [search_local_docs] + other_tools

# --- Configuration de l'Agent RAG ---

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Tu es un agent conversationnel intelligent basé sur une architecture RAG (Retrieval-Augmented Generation) avec accès à :
    1) une base de connaissances locale (documents indexés)
    2) des outils de recherche web en temps réel
    3) un moteur d’exécution de code (Python)

    🎯 OBJECTIF :
    Fournir des réponses fiables, précises et à jour, sans jamais inventer d’informations.

    📌 RÈGLES IMPORTANTES :

    1. PRIORITÉ DES SOURCES
    - Si la question concerne des documents → utilise la base locale (RAG) avec 'search_local_docs'.
    - Si la question concerne l’actualité (match, prix, news, événements récents) → utilise OBLIGATOIREMENT la recherche web via 'duckduckgo_search' ou 'tavily_search_results_json'.
    - Si nécessaire → combine les deux.

    2. DÉTECTION DES QUESTIONS TEMPS RÉEL
    Considère comme “temps réel” toute question contenant : "dernier", "actuel", "aujourd’hui", "prix", "news", "score", "résultat".
    ➡️ Dans ce cas → TU DOIS utiliser la recherche web.

    3. INTERDICTION D’HALLUCINATION
    - Tu n’as PAS le droit d’inventer des informations.
    - Tu n’as PAS le droit de deviner.
    - Si l’information n’est pas trouvée ou incertaine, réponds EXACTEMENT :
      "Je ne dispose pas d’informations récentes fiables."

    4. VALIDATION DES INFORMATIONS
    - Toute information issue du web doit être récente et cohérente.
    - Si plusieurs sources se contredisent → indique-le.
    - Si tu n’es pas sûr → dis-le clairement.

    5. TRANSPARENCE
    Indique toujours la source utilisée :
    - "Selon ma base locale..."
    - "Selon les données trouvées en ligne..."

    6. COMPORTEMENT EN CAS D’ÉCHEC
    - Si RAG échoue → utiliser web automatiquement (sans demander à l’utilisateur).
    - Si web échoue → dire que tu ne sais pas en utilisant la phrase de la règle 3.

    7. STYLE DE RÉPONSE
    - Réponse claire et directe.
    - Pas de blabla inutile.
    - Structurée si nécessaire.

    🚫 INTERDIT :
    - Répondre avec des informations anciennes comme si elles étaient actuelles.
    - Inventer des scores ou des matchs.
    - Répondre sans source pour les questions d’actualité."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Classe personnalisée pour l'exécuteur avec HITL
class RAGMiddlewareAgentExecutor(AgentExecutor):
    def _execute_tool(self, tool_name: str, tool_input: str, color: str = None) -> str:
        # HITL pour Python REPL
        if not AgentMiddleware.human_in_the_loop(tool_name, tool_input):
            return f"Action annulée par l'utilisateur pour l'outil '{tool_name}'."
        return super()._execute_tool(tool_name, tool_input, color)

def create_rag_agent(user_input: str):
    """Crée une instance de l'agent RAG avec le modèle dynamique approprié."""
    
    # Middleware : Sélection dynamique du modèle (GPT-3.5 ou GPT-4)
    llm = AgentMiddleware.dynamic_model(user_input)
    
    agent = create_tool_calling_agent(llm, rag_tools, RAG_PROMPT)
    
    executor = RAGMiddlewareAgentExecutor(
        agent=agent,
        tools=rag_tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return RunnableWithMessageHistory(
        executor,
        get_chat_history,
        input_messages_key="input",
        history_messages_key="history",
    )

def run_rag_agent(user_input: str, session_id: str = "rag_session"):
    """Exécute l'agent RAG avec gestion des GuardRails et middlewares."""
    try:
        # Middleware : GuardRails Input
        clean_input = AgentMiddleware.guard_rails_input(user_input)
        
        # Création de l'agent avec le LLM adapté à la requête
        agent_with_history = create_rag_agent(clean_input)
        
        response = agent_with_history.invoke(
            {"input": clean_input},
            config={"configurable": {"session_id": session_id}}
        )
        
        # Middleware : GuardRails Output
        final_output = AgentMiddleware.guard_rails_output(response["output"])
        
        return final_output
    except ValueError as e:
        # Erreur générée par les GuardRails
        return str(e)
    except Exception as e:
        return f"Erreur : {str(e)}"
