import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import BaseMessage
from typing import List, Dict, Any, Union

# Chargement de la configuration .env
load_dotenv()

# Import des composants locaux
from tools import tools
from memory import get_chat_history
from middleware import AgentMiddleware

# Base prompt template
BASE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Tu es un agent conversationnel intelligent basé sur une architecture RAG (Retrieval-Augmented Generation) avec accès à :
    1) une base de connaissances locale (documents indexés)
    2) des outils de recherche web en temps réel
    3) un moteur d’exécution de code (Python)

    🎯 OBJECTIF :
    Fournir des réponses fiables, précises et à jour, sans jamais inventer d’informations.

    📌 RÈGLES IMPORTANTES :

    1. PRIORITÉ DES SOURCES
    - Si la question concerne des documents → utilise la base locale (RAG) via les outils appropriés.
    - Si la question concerne l’actualité (match, prix, news, événements récents) → utilise OBLIGATOIREMENT la recherche web.
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

# --- Custom Agent Executor with HITL ---

class MiddlewareAgentExecutor(AgentExecutor):
    """Extension d'AgentExecutor pour intégrer les middlewares HITL et Error Handling."""
    
    def _execute_tool(self, tool_name: str, tool_input: str, color: str = None) -> str:
        """Surcharge l'exécution de l'outil pour ajouter HITL validation."""
        # Middleware : Human In The Loop
        if not AgentMiddleware.human_in_the_loop(tool_name, tool_input):
            return f"Action annulée par l'utilisateur : l'outil '{tool_name}' n'a pas été exécuté."
        
        # Exécution normale (inclut déjà tool_error_handler via tools.py)
        return super()._execute_tool(tool_name, tool_input, color)

def run_agent(user_input: str, session_id: str = "default_session"):
    """Exécute l'agent avec l'architecture middleware complète en cascade."""
    
    try:
        # 1. Middleware : GuardRails Input
        clean_input = AgentMiddleware.guard_rails_input(user_input)
        
        # 2. Middleware : Dynamic Model Selection
        llm = AgentMiddleware.dynamic_model(clean_input)
        
        # 3. Middleware : Dynamic Prompt Selection
        current_prompt = AgentMiddleware.dynamic_prompt(clean_input, BASE_PROMPT)
        
        # 4. Initialisation de l'agent avec le LLM et Prompt dynamiques
        agent = create_tool_calling_agent(llm, tools, current_prompt)
        
        # 5. Configuration de l'AgentExecutor avec HITL middleware
        executor = MiddlewareAgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
        
        # 6. Gestion de l'historique
        agent_with_history = RunnableWithMessageHistory(
            executor,
            get_chat_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        
        # 7. Exécution
        response = agent_with_history.invoke(
            {"input": clean_input},
            config={"configurable": {"session_id": session_id}}
        )
        
        # 8. Middleware : GuardRails Output
        final_output = AgentMiddleware.guard_rails_output(response["output"])
        
        return {"output": final_output}

    except ValueError as e:
        # Erreur générée par les GuardRails
        return {"output": str(e)}
    except Exception as e:
        # Erreur générale gérée par le middleware de gestion d'erreurs global
        return {"output": f"Une erreur système s'est produite : {str(e)}"}

if __name__ == "__main__":
    print("--- Agent Conversationnel avec Architecture Middleware ---")
    print("Agent prêt. Tapez 'quit' pour quitter.")
    while True:
        try:
            user_msg = input("\nVous : ")
            if user_msg.lower() == 'quit':
                break
            
            response = run_agent(user_msg)
            print(f"\nAgent : {response['output']}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nErreur : {str(e)}")
