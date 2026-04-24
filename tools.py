import os
from langchain_community.tools import DuckDuckGoSearchRun, TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain.tools import tool
from langchain_core.tools import Tool
import requests

# --- Configuration des Outils ---

# Recherche DuckDuckGo
search_tool = DuckDuckGoSearchRun()

# Recherche Tavily (Optimisé pour LLM)
# Nécessite TAVILY_API_KEY dans le .env
try:
    tavily_tool = TavilySearchResults(k=3)
except Exception:
    # Fallback si la clé est manquante lors de l'initialisation (pour les tests)
    tavily_tool = Tool(
        name="tavily_search_results_json",
        description="Recherche Tavily (Non configurée)",
        func=lambda x: "Erreur : La clé API Tavily est manquante. Veuillez configurer TAVILY_API_KEY dans le .env."
    )

# Environnement d'exécution Python
python_repl = PythonREPLTool()

@tool
def calculate(expression: str) -> str:
    """Calcule le résultat d'une expression mathématique simple."""
    try:
        # Utilisation de eval avec prudence pour cet exemple
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Erreur de calcul : {str(e)}"

@tool
def get_external_joke() -> str:
    """Récupère une blague aléatoire depuis une API externe."""
    try:
        response = requests.get("https://official-joke-api.appspot.com/random_joke", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return f"{data['setup']} - {data['punchline']}"
        return "Impossible de récupérer une blague."
    except Exception as e:
        return f"Erreur API : {str(e)}"

# --- Gestion centralisée des erreurs d'outils (Middleware) ---

def tool_error_handler(error: Exception) -> str:
    """Gère les erreurs d'exécution des outils de manière centralisée."""
    error_msg = str(error)
    if "TAVILY_API_KEY" in error_msg:
        return "Erreur : La clé API Tavily est manquante ou invalide."
    if "timeout" in error_msg.lower():
        return "Erreur : L'outil a mis trop de temps à répondre. Veuillez réessayer."
    return f"Une erreur s'est produite lors de l'utilisation de l'outil : {error_msg}"

# Attribution du gestionnaire d'erreurs à chaque outil
for t in [search_tool, tavily_tool, python_repl, calculate, get_external_joke]:
    t.handle_tool_error = tool_error_handler

# Liste des outils exportés
tools = [search_tool, tavily_tool, python_repl, calculate, get_external_joke]
