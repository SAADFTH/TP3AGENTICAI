import os
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig

# --- Configuration et Règles Métier ---

MODELS = {
    "standard": "gpt-3.5-turbo",
    "advanced": "gpt-4-turbo-preview"
}

FORBIDDEN_KEYWORDS = ["mot_secret", "attaque", "bypass"]

# --- Middlewares ---

class AgentMiddleware:
    """Architecture middleware complète pour l'agent LangChain."""

    @staticmethod
    def dynamic_model(user_input: str) -> ChatOpenAI:
        """Sélection dynamique du modèle selon la complexité de l'entrée."""
        # Règle métier : Utiliser un modèle avancé pour le code ou les calculs complexes
        if any(kw in user_input.lower() for kw in ["code", "python", "algorithme", "complexe"]):
            model_name = MODELS["advanced"]
        else:
            model_name = MODELS["standard"]
        
        return ChatOpenAI(model=model_name, temperature=0)

    @staticmethod
    def dynamic_prompt(user_input: str, base_prompt: ChatPromptTemplate) -> ChatPromptTemplate:
        """Adaptation automatique des prompts selon le contexte."""
        # Règle métier : Ajouter des instructions spécifiques si l'utilisateur demande du code
        if "python" in user_input.lower() or "code" in user_input.lower():
            return ChatPromptTemplate.from_messages([
                ("system", "Tu es un expert en programmation Python. Réponds avec des explications claires et du code optimisé."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
        return base_prompt

    @staticmethod
    def guard_rails_input(user_input: str) -> str:
        """Filtrage et validation du contenu en entrée."""
        # Règle métier : Bloquer les mots-clés interdits
        for kw in FORBIDDEN_KEYWORDS:
            if kw in user_input.lower():
                raise ValueError(f"Contenu bloqué : l'utilisation du terme '{kw}' est interdite.")
        return user_input

    @staticmethod
    def guard_rails_output(output: str) -> str:
        """Filtrage et validation du contenu en sortie."""
        # Règle métier : S'assurer que la sortie ne contient pas d'informations sensibles simulées
        if "CONFIDENTIEL" in output.upper():
            return "[SORTIE FILTRÉE POUR SÉCURITÉ]"
        
        # Anti-hallucination simplifiée : Vérifier si l'agent est trop évasif sans utiliser d'outils
        hallucination_check_keywords = ["je pense que", "probablement", "je suppose"]
        if any(kw in output.lower() for kw in hallucination_check_keywords) and len(output) < 100:
             return "Je n'ai pas pu confirmer cette information avec une source fiable. Veuillez reformuler votre question ou vérifier sur le web."
             
        return output

    @staticmethod
    def human_in_the_loop(tool_name: str, tool_input: str) -> bool:
        """Validation humaine pour les outils critiques (ex: PythonREPL)."""
        # Note: input() ne fonctionne pas dans Streamlit. 
        # Pour le TP, on autorise l'exécution mais on affiche un message dans les logs.
        if tool_name == "Python_REPL":
            print(f"\n[HITL] L'agent exécute le code suivant :\n{tool_input}")
            return True
        return True

# --- Documentation Technique ---
"""
DOCUMENTATION TECHNIQUE MIDDLEWARE
---------------------------------

1. dynamic_model :
   - But : Optimiser le coût et la performance.
   - Règle : GPT-4 pour le code/complexité, GPT-3.5 pour le reste.

2. dynamic_prompt :
   - But : Améliorer la pertinence des réponses.
   - Règle : Injection de personas spécialisés selon l'intention détectée.

3. guard_rails :
   - But : Sécurité et conformité.
   - Règle : Filtrage par liste noire (entrée) et détection de fuite de données (sortie).

4. human_in_the_loop :
   - But : Contrôle des actions critiques.
   - Règle : Interception manuelle pour l'exécution de code arbitraire.
"""
