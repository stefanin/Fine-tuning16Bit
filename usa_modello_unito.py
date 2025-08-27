import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- 1. Definisci il Percorso del Tuo Modello ---

# Questo ora è un "model name" locale, la tua cartella.
model_path = directoryM+"/modello-unito-soprannomi-16bit"

print(f"--- Caricamento del modello fine-tunato da: {model_path} ---")
# --- 2. Carica il Modello e il Tokenizer ---
# Carichiamo il modello direttamente dalla cartella, transformers capirà tutto da solo.
# Assicuriamoci di caricarlo nello stesso formato in cui è stato salvato (bfloat16) per efficienza.
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto" # Manda il modello sulla GPU
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

print("--- Modello pronto per l'inferenza! ---")

# --- 3. Interroga il Modello (usando una pipeline per semplicità) ---

# Creiamo una pipeline di transformers, un modo semplice per fare inferenza
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# Prepara una domanda. Ricorda di usare lo stesso formato del training
# per ottenere i risultati migliori.
system_prompt = "Sei un assistente che conosce i soprannomi delle persone."
instruction = "Qual è il sopranome di Guglielmo"

# La pipeline gestisce la formattazione del template per noi se il tokenizer è configurato bene,
# ma per essere sicuri, applichiamolo manualmente.
# NOTA: Per Qwen2, il template di chat è importante.
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": instruction}
]

# Usa il tokenizer per applicare il template di chat corretto
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("\n--- Generazione della risposta ---")

# Genera la risposta usando la pipeline
outputs = pipe(
    prompt,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
)

# Estrai e stampa la risposta generata
generated_text = outputs[0]['generated_text']

# Spesso la risposta include anche il prompt, quindi puliamola per vedere solo la parte nuova
assistant_response = generated_text.split("<|im_start|>assistant")[1].strip()

print(f"Domanda: {instruction}")
print(f"Risposta del Modello: {assistant_response}")
print("---------------------------------")
