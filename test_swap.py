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
)
print("GPU memory:", torch.cuda.memory_allocated(0))
try:
    pipe.to("cpu")
    print("Moved to CPU successfully.")
except Exception as e:
    print("Failed to move to CPU:", e)
