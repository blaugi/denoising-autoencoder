from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torchvision.transforms import v2
import matplotlib.pyplot as plt

from modeling.autoencoder import AutoEncoder
from modeling.dataset import CIFAR10DataModule
from modeling.decoder import Decoder
from modeling.encoder import Encoder
from modeling.noise import SaltAndPepperNoise

def main():
    seed_everything(42, workers=True)

    # noise_fn = v2.RandomErasing()
    noise_fn = v2.GaussianNoise(mean=0.0, sigma=0.1, clip=True)
    # noise_fn = SaltAndPepperNoise(amount=0.05)
    model = AutoEncoder(
        decoder=Decoder,
        encoder=Encoder,
        base_channel_size=64,
        latent_dim=256,
        num_input_channels=3,
        noise_fn=noise_fn,
    )

    dm = CIFAR10DataModule(num_workers=3)

    trainer = Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
    trainer.fit(model,datamodule=dm)

    originals, noisies, reconstructions = model.validation_step_outputs
    n_images = 5

    fig, axes = plt.subplots(n_images, 3, figsize=(8, 3 * n_images))
    for i in range(n_images):
        # Index into the batch saved in model.validation_step_outputs
        img_orig = (originals[i].detach().cpu() * 0.5) + 0.5
        img_noisy = (noisies[i].detach().cpu() * 0.5) + 0.5
        img_recon = (reconstructions[i].detach().cpu() * 0.5) + 0.5
        
        axes[i, 0].imshow(img_orig.permute(1, 2, 0))
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img_noisy.permute(1, 2, 0))
        axes[i, 1].set_title("Noisy Input")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(img_recon.permute(1, 2, 0))
        axes[i, 2].set_title("Reconstructed")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()