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
    quantization_config=quant_config,
    device_map="balanced"
)
print("VRAM loaded GPU0:", torch.cuda.memory_allocated(0) / 1024**3)
print("VRAM loaded GPU1:", torch.cuda.memory_allocated(1) / 1024**3)
