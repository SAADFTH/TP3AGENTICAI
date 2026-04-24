import pytest
from tools import calculate, get_external_joke
from memory import get_chat_history
from agent import run_agent
import os
from unittest.mock import patch, MagicMock

# --- Tests des Outils ---

def test_calculate_tool():
    """Teste l'outil de calcul."""
    assert calculate.run("2 + 2") == "4"
    assert calculate.run("10 * 5") == "50"
    assert "Erreur" in calculate.run("invalid expression")

def test_joke_tool():
    """Teste l'outil de blague (mocké)."""
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "setup": "Pourquoi les plongeurs plongent toujours en arrière ?",
            "punchline": "Parce que sinon ils tombent dans le bateau."
        }
        result = get_external_joke.run({})
        assert "Pourquoi les plongeurs" in result
        assert "Parce que sinon" in result

# --- Tests de la Mémoire ---

def test_memory_persistence():
    """Teste la persistance de la mémoire SQLite."""
    session_id = "test_session_123"
    history = get_chat_history(session_id)
    
    # Nettoyage préalable (si nécessaire)
    history.clear()
    
    # Ajout de messages
    history.add_user_message("Bonjour")
    history.add_ai_message("Salut !")
    
    # Récupération et vérification
    new_history = get_chat_history(session_id)
    messages = new_history.messages
    assert len(messages) == 2
    assert messages[0].content == "Bonjour"
    assert messages[1].content == "Salut !"
    
    # Nettoyage final
    history.clear()

# --- Tests de l'Agent ---

@patch("agent.agent_with_history")
def test_agent_run(mock_agent_with_history):
    """Teste l'exécution de l'agent avec un mock de l'objet agent_with_history."""
    mock_agent_with_history.invoke.return_value = {"output": "Réponse simulée"}
    
    response = run_agent("Test message", session_id="test_agent_session")
    assert response["output"] == "Réponse simulée"
    mock_agent_with_history.invoke.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__])
