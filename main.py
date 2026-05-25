import os
import json
import datetime
import pickle
import re
import torch
import torch.nn as nn
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

# Monkey-patch pour corriger le bug CrewAI x Groq (cache_breakpoint)
import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda msg: msg

# ---------------------------------------------------------
# CONFIGURATION LLM
# ---------------------------------------------------------
# Assurez-vous que la variable d'environnement GROQ_API_KEY est définie
# par exemple dans un fichier .env ou dans votre terminal.
# os.environ["GROQ_API_KEY"] = "VOTRE_CLE_API"

# Utilisation de llama-3.3-70b-versatile (qui possède un compteur de quota séparé)
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.2,
    max_retries=5
)

# ---------------------------------------------------------
# 1. REDÉFINITION DU MODÈLE PYTORCH
# ---------------------------------------------------------
class HateSpeechMLP(nn.Module):
    def __init__(self, input_dim):
        super(HateSpeechMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.out = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.out(x)
        return x

def clean_text(text):
    """Fonction de nettoyage pour l'inférence."""
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    return text.lower()

# ---------------------------------------------------------
# 2. OUTILS (TOOLS) - Format Orienté Objet pour CrewAI
# ---------------------------------------------------------
class PyTorchInferenceTool(BaseTool):
    name: str = "PyTorchInferenceTool"
    description: str = "Utilise un modèle PyTorch de Deep Learning pour prédire si un texte est Haineux (Hate Speech) ou Non-Haineux."

    def _run(self, text: str) -> str:
        try:
            if not os.path.exists('tfidf_vectorizer.pkl') or not os.path.exists('hate_speech_model.pth'):
                return "Erreur : Les fichiers du modèle DL n'ont pas été trouvés. Avez-vous lancé train_model.py ?"
                
            with open('tfidf_vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
                
            cleaned_text = clean_text(text)
            X_tfidf = vectorizer.transform([cleaned_text]).toarray()
            X_tensor = torch.tensor(X_tfidf, dtype=torch.float32)
            
            input_dim = X_tensor.shape[1]
            model = HateSpeechMLP(input_dim)
            model.load_state_dict(torch.load('hate_speech_model.pth', weights_only=True))
            model.eval()
            
            with torch.no_grad():
                outputs = model(X_tensor)
                probability = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
            class_name = "Haineux" if predicted.item() == 1 else "Non-Haineux/Neutre"
            conf_score = probability[0][predicted.item()].item() * 100
            
            return f"Prédiction du DL : {class_name} (Confiance: {conf_score:.2f}%)"
        except Exception as e:
            return f"Erreur lors de l'inférence PyTorch : {str(e)}"

class ContextCheckerTool(BaseTool):
    name: str = "ContextCheckerTool"
    description: str = "Analyse lexicale basique (N-Grams/Lexique codé en dur) pour repérer des mots toxiques évidents."

    def _run(self, text: str) -> str:
        toxic_lexicon = ["kill", "murder", "hate", "scum", "idiot", "stupid", "die", "bitch", "bastard"]
        words = re.findall(r'\b\w+\b', text.lower())
        found = [word for word in words if word in toxic_lexicon]
        
        if found:
            return f"Alerte Lexicale : Mots toxiques repérés : {', '.join(found)}."
        return "Analyse Lexicale : Aucun mot toxique du lexique n'a été repéré."

pytorch_inference_tool = PyTorchInferenceTool()
context_checker_tool = ContextCheckerTool()

# ---------------------------------------------------------
# 3. AGENTS ET SYSTÈME MULTI-AGENTS
# ---------------------------------------------------------
# ---------------------------------------------------------
# 3. AGENT HYBRIDE (Solution anti-Rate Limit)
# ---------------------------------------------------------
# Pour éviter de dépasser la stricte limite de tokens de Groq, nous fusionnons 
# l'expertise dans un seul super-agent qui va utiliser les deux outils à la suite.
hybrid_analyzer_agent = Agent(
    role='Super Analyste (Lexique & Deep Learning)',
    goal="Analyser le texte à l'aide de ContextCheckerTool ET PyTorchInferenceTool, puis donner un court verdict final.",
    backstory="Expert hybride. Tu utilises tes outils pour extraire les faits, puis tu rédiges une conclusion en 2 lignes.",
    verbose=True,
    allow_delegation=False,
    tools=[pytorch_inference_tool, context_checker_tool],
    llm=llm
)

# ---------------------------------------------------------
# 4. FONCTION DE LOGGING (Robustesse)
# ---------------------------------------------------------
def save_log(text, report, human_validation):
    """Sauvegarde les actions et la validation dans un fichier JSON horodaté."""
    log_file = "hate_speech_logs.json"
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "input_text": text,
        "final_report": report,
        "human_validation": human_validation
    }
    
    logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            pass
            
    logs.append(log_entry)
    
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)
        
    print(f"\n[INFO] Log sauvegardé avec succès dans '{log_file}'.")

# ---------------------------------------------------------
# 5. EXECUTION CREWAI
# ---------------------------------------------------------
def run_analysis(user_input):
    """Exécute l'agent hybride CrewAI sur un texte donné et renvoie le rapport."""
    task_analysis = Task(
        description=f"1. Utilise ContextCheckerTool sur la phrase : '{user_input}'.\n2. Utilise PyTorchInferenceTool sur la phrase : '{user_input}'.\n3. Synthétise les deux résultats en un verdict final de 2 ou 3 phrases.",
        expected_output="Un court rapport final justifiant si le texte est Haineux ou Non-Haineux basé sur les deux outils.",
        agent=hybrid_analyzer_agent
    )
    
    crew = Crew(
        agents=[hybrid_analyzer_agent],
        tasks=[task_analysis],
        verbose=True,
        cache=False
    )
    
    result = crew.kickoff()
    return str(result)

def run_cli():
    print("\n===========================================================")
    print(" 🛡️  SYSTÈME MULTI-AGENTS DE DÉTECTION DE DISCOURS HAINEUX")
    print("===========================================================\n")
    
    while True:
        try:
            user_input = input("\n📝 Entrez une phrase à analyser (ou tapez 'quit' pour quitter) : \n> ").strip()
            
            if user_input.lower() in ['quit', 'q', 'exit']:
                print("Fermeture du système. À bientôt !")
                break
                
            if not user_input:
                continue
                
            print("\n[INFO] Lancement des agents CrewAI...\n")
            result = run_analysis(user_input)
            
            print("\n" + "="*60)
            print("📜 RAPPORT FINAL DE L'ORCHESTRATEUR")
            print("="*60)
            print(result)
            print("="*60 + "\n")
            
            while True:
                validation = input("🧑‍⚖️ Validation Humaine requise. Approuvez-vous ce rapport final ? (O/N) : ").strip().upper()
                if validation in ['O', 'N']:
                    break
                print("Erreur : Veuillez répondre par 'O' pour Oui, ou 'N' pour Non.")
                
            status = "Approuvé" if validation == 'O' else "Rejeté"
            print(f"\n✅ Rapport {status} par l'opérateur.")
            
            save_log(user_input, result, status)
            
        except KeyboardInterrupt:
            print("\nArrêt manuel du système.")
            break
        except Exception as e:
            print(f"\n[ERREUR CRITIQUE] Une erreur s'est produite lors de l'exécution : {e}")

if __name__ == "__main__":
    run_cli()
