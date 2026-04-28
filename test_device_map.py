import torch
from diffusers import DiffusionPipeline
import time

pipe = DiffusionPipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo", 
    torch_dtype=torch.bfloat16,
    device_map={"text_encoder": "cpu", "text_encoder_2": "cpu", "": "cuda:0"}
)
print("VRAM loaded:", torch.cuda.memory_allocated() / 1024**3)
try:
    t0 = time.time()
    pipe("A magical sword", height=1024, width=1024, num_inference_steps=2).images[0]
    print("VRAM after gen:", torch.cuda.max_memory_allocated() / 1024**3)
    print("Time taken:", time.time() - t0)
except Exception as e:
    print("Error:", e)
