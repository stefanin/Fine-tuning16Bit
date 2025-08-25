# --- 1. Import delle Librerie ---
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# --- 2. Caricamento Modello e Tokenizer (METODO A 16-BIT) ---

# Usiamo il modello base standard, NON la versione GPTQ
model_name = "Qwen/Qwen2-0.5B-Instruct" 

print(f"Caricamento del modello a 16-bit: {model_name}")

# Carichiamo il modello specificando il tipo di dato a 16-bit (bfloat16)
# `bfloat16` è ottimo per le GPU moderne (serie RTX 3000 e successive)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, # <-- La modifica chiave! Carica in 16-bit
    device_map="auto"           # Mette il modello sulla GPU
)

# Caricamento del tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

print("Modello e tokenizer a 16-bit caricati con successo!")

# --- IL RESTO DELLO SCRIPT È QUASI IDENTICO ---

# --- 3. Preparazione del Dataset ---
def generate_prompt(data_point):
    system_prompt = "Sei un assistente che conosce i soprannomi delle persone."
    full_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{data_point['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{data_point['output']}<|im_end|>"
    )
    return {"text": full_prompt}

dataset = load_dataset("json", data_files="prova.json", split="train")
dataset = load_dataset("json", data_files="prova2.json", split="train") # Aggiunto per avere più dati
dataset = dataset.map(generate_prompt)


# --- 4. Configurazione LoRA ---
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model.config.use_cache = False
model = get_peft_model(model, lora_config)

# --- 5. Argomenti di Training ---
training_arguments = TrainingArguments(
    output_dir="./qwen2-finetuned-soprannomi-16bit",
    num_train_epochs=10,
    per_device_train_batch_size=1, # Partiamo con 1 per sicurezza con la VRAM
    gradient_accumulation_steps=4, # Compensiamo il batch size piccolo
    optim="adamw_torch",
    logging_steps=1,
    learning_rate=2e-4,
    fp16=False, # Non necessario, il modello è già in bfloat16
    bf16=True,  # Abilita il training in bfloat16
    max_grad_norm=0.3,
    warmup_ratio=0.03,
)

# --- 6. Creazione e Avvio del Trainer ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
)

print("Avvio del fine-tuning a 16-bit...")
trainer.train()
trainer.save_model() # Salva solo gli adattatori LoRA
print("Fine-tuning completato!")
