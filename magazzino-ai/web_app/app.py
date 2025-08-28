# app.py
from flask import Flask, request, jsonify, render_template
import requests
import ollama

app = Flask(__name__)

OLLAMA_MODEL_NAME = "tinyllama-magazzino:latest"
API_URL_PHP = "https://sclab.altervista.org/gm/api/cerca_componente.php"

# Endpoint per servire l'interfaccia web
@app.route('/')
def home():
    return render_template('index.html')

# Endpoint per la logica dell'assistente (quello che prima era in mcp_server.py)
@app.route('/ask', methods=['POST'])
def ask_model():
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({"error": "Query non fornita"}), 400

    # FASE DI RECUPERO
    words = user_query.split()
    potential_code = words[-1].replace('?', '')
    
    try:
        response = requests.get(f"{API_URL_PHP}?cod={potential_code}")
        response.raise_for_status()
        db_results = response.json()
    except requests.exceptions.RequestException as e:
        # Invece di restituire un errore JSON, lo formattiamo per la chat
        error_message = f"Errore di connessione all'API del database: {e}"
        return jsonify({"answer": error_message})

    # FASE DI AUMENTO
    if not db_results:
        context = "Dal database non è stato trovato alcun componente corrispondente alla ricerca."
    else:
        context_parts = ["Dati recuperati dal database in tempo reale:"]
        for item in db_results:
            context_parts.append(f"- Codice: {item['cod']}, Descrizione: {item['des']}, Quantità: {item['qta']}, Posizione: {item['sc']}")
        context = "\n".join(context_parts)
    
    # FASE DI GENERAZIONE
    system_prompt = "Sei un assistente AI per la gestione di un magazzino. Basandoti ESCLUSIVAMENTE sui 'Dati recuperati dal database' forniti, rispondi alla 'Domanda dell'utente'."
    
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f"Contesto: {context}\n\nDomanda: {user_query}"}
            ],
            stream=False
        )
        final_answer = response['message']['content']
        return jsonify({"answer": final_answer})
    except Exception as e:
        error_message = f"Errore durante la chiamata al modello LLM: {str(e)}"
        return jsonify({"answer": error_message})

if __name__ == '__main__':
    # Usiamo 0.0.0.0 per essere accessibili dall'esterno del container
    app.run(host='0.0.0.0', port=5000)