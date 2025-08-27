import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
from CFG import percorso_modello_unito_magazzino # Importa il percorso dal file di configurazione

# --- 1. Carica il tuo modello fine-tunato all'avvio del server ---
print(f"Caricamento del modello di magazzino da: {percorso_modello_unito_magazzino}...")
model_path = percorso_modello_unito_magazzino
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto" # Manda il modello sulla GPU
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Modello caricato e pronto.")

# --- 2. Inizializza il server web Flask ---
app = Flask(__name__)

# --- 3. Definisci l'endpoint per ricevere le domande ---
@app.route('/ask', methods=['POST'])
def ask_model():
    # Estrai la domanda dell'utente dal corpo della richiesta
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({"error": "La richiesta non contiene una 'query'"}), 400

    # --- FASE DI RECUPERO (Retrieval) ---
    # Logica semplice per estrarre un potenziale codice dall'input dell'utente
    words = user_query.split()
    potential_code = words[-1].replace('?', '')

    # URL della tua API PHP su Altervista
    api_url = f"https://sclab.altervista.org/gm/api/cerca_componente.php?cod={potential_code}"
    
    # L'autenticazione è temporaneamente disabilitata
    # Per RIATTIVARLA, de-commenta le due righe seguenti e la parola 'headers=headers' nella chiamata
    # API_KEY = "LA_TUA_CHIAVE_API_SEGRETA_E_LUNGA"
    # headers = {'X-API-Key': API_KEY}
    
    try:
        # Chiama l'API per ottenere i dati in tempo reale
        response = requests.get(api_url) # Per riattivare auth, aggiungi: headers=headers
        response.raise_for_status() # Controlla se ci sono stati errori HTTP (es. 404, 500)
        db_results = response.json()
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Errore durante la connessione all'API del database: {e}"}), 500

    # --- FASE DI AUMENTO (Augmentation) ---
    # Costruisci il contesto da dare al modello basandoti sui dati ricevuti
    if not db_results:
        context = "Dal database non è stato trovato alcun componente corrispondente alla ricerca."
    else:
        context_parts = ["Dati recuperati dal database in tempo reale:"]
        for item in db_results:
            context_parts.append(f"- Codice: {item['cod']}, Descrizione: {item['des']}, Quantità: {item['qta']}, Posizione: {item['sc']}")
        context = "\n".join(context_parts)

    # --- FASE DI GENERAZIONE (Generation) ---
    # Crea il prompt finale per l'LLM, unendo contesto e domanda originale
    system_prompt = "Sei un assistente AI per la gestione di un magazzino. Basandoti ESCLUSIVAMENTE sui 'Dati recuperati dal database' forniti, rispondi alla 'Domanda dell'utente'."
    prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{context}

Domanda dell'utente: '{user_query}'<|im_end|>
<|im_start|>assistant
"""
    # Tokenizza il prompt e invialo alla GPU
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Genera la risposta
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
    
    # Decodifica la risposta e puliscila per restituire solo la parte dell'assistente
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    final_answer = response_text.split("<|im_start|>assistant")[-1].strip()

    # Invia la risposta finale al client
    return jsonify({"answer": final_answer})

# --- 4. Avvia il server ---
if __name__ == '__main__':
    # Esegue il server, rendendolo accessibile da altri computer sulla stessa rete
    app.run(host='0.0.0.0', port=5000, debug=True)