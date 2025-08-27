import requests
import json

SERVER_URL = "http://localhost:5000/ask"

print("Client per l'assistente di magazzino AI (digita 'esci' per terminare)")
print("-" * 60)

while True:
    query = input(">>> Domanda: ")
    if query.lower() == 'esci':
        break
        
    try:
        # Invia la domanda al server MCP
        response = requests.post(SERVER_URL, json={'query': query})
        response.raise_for_status()
        
        # Stampa la risposta
        answer = response.json().get('answer', 'Nessuna risposta ricevuta.')
        print(f"🤖 Assistente: {answer}")
        
    except requests.exceptions.RequestException as e:
        print(f"\nErrore di connessione al server: {e}")
    except json.JSONDecodeError:
        print("\nErrore: Il server ha restituito una risposta non valida.")

print("Sessione terminata.")