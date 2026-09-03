import torch, transformers, deepspeed, h5py, omegaconf, timm, einops, cv2, imageio, av, plotly, wandb
from torch.nn.attention.flex_attention import flex_attention
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("transformers", transformers.__version__, "deepspeed", deepspeed.__version__, "h5py", h5py.__version__)
