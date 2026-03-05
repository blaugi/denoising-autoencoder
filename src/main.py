from lightning.pytorch import Trainer, seed_everything

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

    trainer = Trainer()
    trainer.fit(model,datamodule=dm)


if __name__ == "__main__":
    main()