import json
import random

def pulisci_dati(dati_grezzi):
    """Estrae e pulisce i dati rilevanti dal file JSON di input."""
    # Trova la sezione 'table' e estrai la lista 'data'
    tabella_dati = next((item for item in dati_grezzi if item.get("type") == "table"), None)
    if not tabella_dati:
        raise ValueError("Formato JSON non valido: nessuna tabella trovata.")
        
    dati_puliti = []
    for item in tabella_dati["data"]:
        # Ignora i campi non necessari e gestisci i valori nulli
        dati_puliti.append({
            "cod": item.get("cod", "").strip(),
            "des": item.get("des", "N/A") or "Descrizione non disponibile",
            "qta": str(item.get("qta", "0")),
            "sc": item.get("sc", "N/A") or "Posizione non specificata"
        })
    return dati_puliti

def genera_esempi_training(dati_magazzino):
    """Genera una lista di esempi domanda/risposta per il fine-tuning."""
    esempi = []

    # 1. Genera domande per ogni singolo articolo
    for item in dati_magazzino:
        cod = item['cod']
        des = item['des']
        qta = item['qta']
        sc = item['sc']

        # Variazione 1: Domanda generica
        instruction1 = f"Dammi informazioni sul codice {cod}."
        output1 = f"Il componente con codice {cod} è descritto come '{des}'. Quantità disponibile: {qta}. Si trova nel contenitore: {sc}."
        esempi.append({"instruction": instruction1, "output": output1})

        # Variazione 2: Domanda sulla quantità
        instruction2 = f"Quanti pezzi ci sono del componente {cod}?"
        output2 = f"Del componente {cod} ci sono {qta} pezzi disponibili."
        esempi.append({"instruction": instruction2, "output": output2})
        
        # Variazione 3: Domanda sulla posizione
        instruction3 = f"Dove si trova il codice {cod}?"
        output3 = f"Il componente {cod} si trova nel contenitore {sc}."
        esempi.append({"instruction": instruction3, "output": output3})

    # 2. Genera domande per codici parziali (per testare l'elenco)
    prefissi_comuni = ["BD", "TIP", "74", "SN74LS", "C", "R"]
    for prefisso in prefissi_comuni:
        corrispondenze = [item for item in dati_magazzino if item['cod'].upper().startswith(prefisso)]
        if 1 < len(corrispondenze) < 10: # Limita per non creare output troppo lunghi
            instruction = f"Mostrami tutti i componenti che iniziano con {prefisso}."
            output_lines = [f"Ho trovato più componenti che iniziano con '{prefisso}':"]
            for item in corrispondenze:
                output_lines.append(f"- Codice: {item['cod']}, Descrizione: {item['des']}, Qta: {item['qta']}, Posizione: {item['sc']}")
            output = "\n".join(output_lines)
            esempi.append({"instruction": instruction, "output": output})

    # 3. Genera domande per codici inesistenti
    for _ in range(50): # Aggiungiamo 50 esempi di fallimento
        codice_falso = f"FAKECODE{random.randint(1000, 9999)}"
        instruction = f"Cerca informazioni sul componente {codice_falso}."
        output = f"Mi dispiace, non ho trovato nessun componente con il codice {codice_falso} nel magazzino."
        esempi.append({"instruction": instruction, "output": output})
        
    return esempi

# --- Script Principale ---
if __name__ == "__main__":
    # Carica il file JSON del magazzino
    with open('mag.json', 'r', encoding='utf-8') as f:
        dati_grezzi = json.load(f)
    
    # Pulisci i dati
    dati_magazzino = pulisci_dati(dati_grezzi)
    
    # Genera gli esempi di training
    dataset_training = genera_esempi_training(dati_magazzino)
    
    # Mescola il dataset per un training migliore
    random.shuffle(dataset_training)
    
    # Salva il nuovo dataset in un formato JSON Lines (un oggetto JSON per riga)
    with open('magazzino_dataset.jsonl', 'w', encoding='utf-8') as f:
        for entry in dataset_training:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"Dataset generato con successo! Creato il file 'magazzino_dataset.jsonl' con {len(dataset_training)} esempi.")
    print("\nEsempio di un dato generato:")
    print(dataset_training[0])