import torch

# Questo è il test fondamentale. Deve restituire True.
is_available = torch.cuda.is_available()
print(f"CUDA è disponibile? {is_available}")

if is_available:
    # Se è disponibile, vediamo i dettagli
    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Sono state trovate {gpu_count} GPU.")
    print(f"Nome della GPU: {gpu_name}")
else:
    print("PyTorch non riesce a trovare una GPU compatibile con CUDA.")