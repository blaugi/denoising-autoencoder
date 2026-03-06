import lightning as L
import torch
from torchvision.transforms import v2
from collections.abc import Callable

from modeling.decoder import Decoder
from modeling.encoder import Encoder


# original implementation from https://github.com/watasabi
class AutoEncoder(L.LightningModule):
    def __init__(
        self,
        encoder: type[Encoder],
        decoder: type[Decoder],
        base_channel_size: int,
        latent_dim: int,
        num_input_channels: int,
        width: int = 32,
        height: int = 32,
        noise_fn: Callable = v2.GaussianNoise,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder(num_input_channels, base_channel_size, self.latent_dim)
        self.decoder = decoder(num_input_channels, base_channel_size, self.latent_dim)
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)
        self.noise_fn = noise_fn

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        x_clean, _ = batch
        x_noisy = self.noise_fn(x_clean)
        x_hat = self.forward(x_noisy)

        loss = torch.nn.functional.mse_loss(x_hat, x_clean)
        return loss, x_noisy, x_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=10, min_lr=5e-5
        )
        return {"optimizer": optimizer, "scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, x_noisy, x_hat = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        if batch_idx == 0:
            self.validation_step_outputs = (batch[0], x_noisy, x_hat)
        return loss

    def test_step(self, batch, batch_idx):
        loss, _ ,_  = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)