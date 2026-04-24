import pytest
from middleware import AgentMiddleware
from tools import tools, tool_error_handler
from langchain_core.prompts import ChatPromptTemplate
from unittest.mock import MagicMock, patch
from dotenv import load_dotenv
import os

# Chargement des variables d'environnement pour les tests
load_dotenv()

# --- Tests Middlewares ---

def test_guard_rails_input():
    """Teste le filtrage des entrées."""
    assert AgentMiddleware.guard_rails_input("Bonjour") == "Bonjour"
    with pytest.raises(ValueError, match="Contenu bloqué"):
        AgentMiddleware.guard_rails_input("Ceci est un mot_secret")

def test_guard_rails_output():
    """Teste le filtrage des sorties."""
    assert AgentMiddleware.guard_rails_output("Réponse normale") == "Réponse normale"
    assert AgentMiddleware.guard_rails_output("Ceci est CONFIDENTIEL") == "[SORTIE FILTRÉE POUR SÉCURITÉ]"

def test_dynamic_model_selection():
    """Teste la sélection dynamique du modèle."""
    
    # Entrée simple -> GPT-3.5
    model_simple = AgentMiddleware.dynamic_model("Bonjour")
    assert model_simple.model_name == "gpt-3.5-turbo"
    
    # Entrée complexe -> GPT-4
    model_complex = AgentMiddleware.dynamic_model("Écris un code python complexe")
    assert model_complex.model_name == "gpt-4-turbo-preview"

def test_dynamic_prompt_adaptation():
    """Teste l'adaptation du prompt."""
    base_prompt = ChatPromptTemplate.from_messages([("system", "Base")])
    
    # Pas de changement pour entrée simple
    prompt_simple = AgentMiddleware.dynamic_prompt("Bonjour", base_prompt)
    assert prompt_simple == base_prompt
    
    # Changement pour entrée type code
    prompt_code = AgentMiddleware.dynamic_prompt("Aide moi en python", base_prompt)
    # Vérification du message system injecté via son template
    assert "expert en programmation Python" in prompt_code.messages[0].prompt.template

def test_human_in_the_loop():
    """Teste la validation humaine (toujours vrai dans Streamlit)."""
    # Toujours vrai pour Python_REPL (avec log console)
    assert AgentMiddleware.human_in_the_loop("Python_REPL", "print(1)") is True
    # Outil non critique
    assert AgentMiddleware.human_in_the_loop("calculate", "2+2") is True

# --- Tests Outils et Gestion d'erreurs ---

def test_tool_error_handling():
    """Teste le gestionnaire d'erreurs d'outils."""
    err_tavily = Exception("TAVILY_API_KEY missing")
    assert "clé API Tavily est manquante" in tool_error_handler(err_tavily)
    
    err_timeout = Exception("Request timeout occurred")
    assert "mis trop de temps à répondre" in tool_error_handler(err_timeout)
    
    err_generic = Exception("Unknown error")
    assert "Une erreur s'est produite" in tool_error_handler(err_generic)

def test_tools_integration():
    """Vérifie que tous les outils sont présents et configurés."""
    tool_names = [t.name for t in tools]
    assert "duckduckgo_search" in tool_names
    assert "tavily_search_results_json" in tool_names
    assert "Python_REPL" in tool_names
    assert "calculate" in tool_names
    
    for t in tools:
        assert t.handle_tool_error is not None

if __name__ == "__main__":
    pytest.main([__file__])
