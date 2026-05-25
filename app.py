from flask import Flask, render_template, request, jsonify
from main import run_analysis, save_log

app = Flask(__name__)

@app.route('/')
def home():
    """Rend la page principale du Dashboard HITL."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Reçoit le texte du frontend et déclenche l'analyse CrewAI."""
    data = request.get_json()
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({"error": "Texte vide"}), 400
        
    try:
        # Exécution des agents
        report = run_analysis(text)
        return jsonify({"report": report})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/validate', methods=['POST'])
def validate():
    """Enregistre la décision humaine."""
    data = request.get_json()
    text = data.get('text')
    report = data.get('report')
    status = data.get('status')
    
    if not all([text, report, status]):
        return jsonify({"error": "Données incomplètes"}), 400
        
    try:
        save_log(text, report, status)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Démarrage du Serveur Flask - Dashboard Antigravity HITL")
    app.run(debug=True, port=5000)
