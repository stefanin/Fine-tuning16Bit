import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- 1. Definisci i Nomi/Percorsi ---

# Il modello base che hai usato per il training (la versione a precisione piena)
base_model_name = "Qwen/Qwen2-0.5B-Instruct"

# La cartella dove hai salvato i tuoi adattatori LoRA dal training a 16-bit
adapter_path = "./magazzino-bot-16bit"

# La cartella dove salvare il modello finale unito
output_dir = "./modello-magazzino-bot-16bit"

print("--- Caricamento del modello base a 16-bit ---")
# --- 2. Carica il Modello Base e il Tokenizer ---
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto" # Mettilo sulla GPU per unire più velocemente
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

print(f"--- Caricamento e unione degli adattatori da {adapter_path} ---")
# --- 3. Carica gli Adattatori e Uniscili ---
model = PeftModel.from_pretrained(base_model, adapter_path)

# Questa volta, il comando funzionerà!
merged_model = model.merge_and_unload()
print("Pesi uniti con successo!")


# --- 4. Salva il Modello Finale ---
print(f"Salvataggio del modello unito in {output_dir}...")
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("--- Operazione completata! ---")
print(f"Il tuo modello autonomo e fine-tunato si trova in: {output_dir}")
