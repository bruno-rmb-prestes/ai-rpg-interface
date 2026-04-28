import torch
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
dtype = torch.bfloat16

print("Loading FireRed to CPU...")
pipe_edit = QwenImageEditPlusPipeline.from_pretrained(
    "FireRedTeam/FireRed-Image-Edit-1.1",
    transformer=QwenImageTransformer2DModel.from_pretrained(
        "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19",
        torch_dtype=dtype,
        device_map="cpu"
    ),
    torch_dtype=dtype,
    device_map="cpu"
)
print("VRAM:", torch.cuda.memory_allocated() / 1024**3)
