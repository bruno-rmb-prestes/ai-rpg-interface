import torch
from unified_backend import init_models
init_models()
print("VRAM allocated after init_models():", torch.cuda.memory_allocated() / 1024**3)
