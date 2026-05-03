import torch
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig

quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit", 
    quant_kwargs={"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.bfloat16}
)

try:
    pipe = DiffusionPipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", 
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
    )
    print("VRAM loaded GPU0:", torch.cuda.memory_allocated(0) / 1024**3)
except Exception as e:
    print("Error:", e)
