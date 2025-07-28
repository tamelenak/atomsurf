import torch.nn as nn
from diffusion_net import DiffusionNetBlock

class StackedDiffusionNetBlock(nn.Module):
    """
    """

    def __init__(self, C_width: int = 128, dropout: float = 0.1,
                 use_bn: bool = False, use_layernorm: bool = True,
                 init_time: float = 10.0, init_std: float = 10.0,
                 num_repeats: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([
            DiffusionNetBlock(
                C_width=C_width,
                dropout=dropout,
                use_bn=use_bn,
                use_layernorm=use_layernorm,
                init_time=init_time,
                init_std=init_std,
            ) for _ in range(num_repeats)
        ])

    def forward(self, surface):
        for blk in self.blocks:
            surface = blk(surface)
        return surface 