import torch
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
print("VRAM after offload init:", torch.cuda.memory_allocated() / 1024**3)
try:
    pipe("A cat", height=1024, width=1024, num_inference_steps=2).images[0]
    print("VRAM during/after gen:", torch.cuda.max_memory_allocated() / 1024**3)
except Exception as e:
    print("Error:", e)
