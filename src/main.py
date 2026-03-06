from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torchvision.transforms import v2
import matplotlib.pyplot as plt

from modeling.autoencoder import AutoEncoder
from modeling.dataset import CIFAR10DataModule
from modeling.decoder import Decoder
from modeling.encoder import Encoder

def main():
    seed_everything(42, workers=True)

    model = AutoEncoder(
        decoder=Decoder,
        encoder=Encoder,
        base_channel_size=128,
        latent_dim=256,
        num_input_channels=3,
    )

    dm = CIFAR10DataModule(num_workers=3)

    trainer = Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
    trainer.fit(model,datamodule=dm)

    data_loader_iter = iter(dm.val_dataloader())
    n_images = 5

    images = []
    noise_fn = v2.GaussianNoise(mean=0.0, sigma=0.1, clip=True)

    images = []
    for _ in range(n_images):
        x, _ = next(data_loader_iter)
        x_noisy = noise_fn(x)
        prediction = model.forward(x_noisy)

        images.append((x, x_noisy, prediction))

    fig, axes = plt.subplots(n_images, 3, figsize=(8, 3 * n_images))
    for i, (original, noisy, reconstruction) in enumerate(images):
        img_orig = (original[0].detach().cpu() * 0.5) + 0.5
        img_noisy = (noisy[0].detach().cpu() * 0.5) + 0.5
        img_recon = (reconstruction[0].detach().cpu() * 0.5) + 0.5
        
        # Plot Original
        axes[i, 0].imshow(img_orig.permute(1, 2, 0))
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        # Plot Noisy
        axes[i, 1].imshow(img_noisy.permute(1, 2, 0))
        axes[i, 1].set_title("Noisy Input")
        axes[i, 1].axis("off")

        # Plot Reconstruction
        axes[i, 2].imshow(img_recon.permute(1, 2, 0))
        axes[i, 2].set_title("Reconstructed")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()