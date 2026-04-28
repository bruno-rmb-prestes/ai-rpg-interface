import torch
import gc
from diffusers import DiffusionPipeline
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
dtype = torch.bfloat16

print("VRAM initially:", torch.cuda.memory_allocated() / 1024**3)

print("Loading Z-Image-Turbo...")
pipe_gen = DiffusionPipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=dtype,
    low_cpu_mem_usage=False,
)
pipe_gen.to("cpu")
print("VRAM after gen to cpu:", torch.cuda.memory_allocated() / 1024**3)

print("Loading FireRed...")
pipe_edit = QwenImageEditPlusPipeline.from_pretrained(
    "FireRedTeam/FireRed-Image-Edit-1.1",
    transformer=QwenImageTransformer2DModel.from_pretrained(
        "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19",
        torch_dtype=dtype,
        device_map=None
    ),
    torch_dtype=dtype
)
pipe_edit.to("cpu")
print("VRAM after edit to cpu:", torch.cuda.memory_allocated() / 1024**3)

gc.collect()
torch.cuda.empty_cache()
print("VRAM after empty cache:", torch.cuda.memory_allocated() / 1024**3)

pipe_gen.to("cuda")
print("VRAM after gen to cuda:", torch.cuda.memory_allocated() / 1024**3)

pipe_gen.to("cpu")
gc.collect()
torch.cuda.empty_cache()
pipe_edit.to("cuda")
print("VRAM after gen to cpu, edit to cuda:", torch.cuda.memory_allocated() / 1024**3)
