import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(
        self, num_workers: int, data_dir: str = "./data", batch_size: int = 64
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.base_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.noise_transform = v2.GaussianNoise(mean=0.0, sigma=0.1, clip=True)

    def setup(self, stage=None):
        # Apply transforms and split dataset
        cifar_full = CIFAR10(
            self.data_dir, train=True, transform=self.base_transform, download=True
        )
        self.train_set, self.val_set = random_split(cifar_full, [45000, 5000])

    def train_dataloader(self):
        pw = bool(self.num_workers > 0)
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=pw,
        )

    def val_dataloader(self):
        pw = bool(self.num_workers > 0)
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=pw,
        )
