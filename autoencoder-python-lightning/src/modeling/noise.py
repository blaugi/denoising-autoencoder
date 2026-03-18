import torch
from torch import Tensor

class SaltAndPepperNoise(torch.nn.Module):
    def __init__(self, amount: float = 0.05, salt_vs_pepper: float = 0.5):
        """
        Adds salt and pepper noise to an image.
        Args:
            amount (float): Proportion of pixels to replace with noise.
            salt_vs_pepper (float): Proportion of salt (white) vs pepper (black) noise.
        """
        super().__init__()
        self.amount = amount
        self.salt_vs_pepper = salt_vs_pepper

    def forward(self, img: Tensor) -> Tensor:
        spatial_shape = img.shape[1:] if img.ndim == 3 else img.shape[2:]
        mask = torch.rand(spatial_shape, device=img.device)
        
        out = img.clone()
        salt_mask = mask < (self.amount * self.salt_vs_pepper)
        pepper_mask = (mask >= (self.amount * self.salt_vs_pepper)) & (mask < self.amount)
        
        out[..., salt_mask] = 1.0
        out[..., pepper_mask] = 0.0 # apply to all channels
        return out
