import torch
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig

quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_8bit", 
    quant_kwargs={"load_in_8bit": True}
)

pipe = DiffusionPipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo", 
    torch_dtype=torch.bfloat16,
    quantization_config=quant_config
)
print("VRAM loaded:", torch.cuda.memory_allocated() / 1024**3)
try:
    pipe.to("cpu")
    print("VRAM after to(cpu):", torch.cuda.memory_allocated() / 1024**3)
except Exception as e:
    print("Error:", e)
