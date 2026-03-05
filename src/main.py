from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
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
        base_channel_size=64,
        latent_dim=128,
        num_input_channels=3,
    )

    dm = CIFAR10DataModule(num_workers=3)

    trainer = Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
    trainer.fit(model,datamodule=dm)

    data_loader_iter = iter(dm.val_dataloader())
    n_images = 5

    images = []
    for _ in range(n_images):
        x, _ = next(data_loader_iter)
        prediction = model.forward(x)

        images.append((x, prediction))

    fig, axes = plt.subplots(n_images, 2, figsize=(8, 3 * n_images))
    for i, (original, reconstruction) in enumerate(images):
        img_orig = original[0].detach().cpu()
        img_recon = reconstruction[0].detach().cpu()

        img_orig = (img_orig * 0.5) + 0.5
        img_recon = (img_recon * 0.5) + 0.5

        axes[i, 0].imshow(img_orig.permute(1, 2, 0))
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img_recon.permute(1, 2, 0))
        axes[i, 1].set_title("Reconstructed")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show() 
        
        


if __name__ == "__main__":
    main()