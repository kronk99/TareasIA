import pytorch_lightning as pl
from torch.utils.data import DataLoader

class MVTECDatamodule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers, image_size):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

    def setup(self, stage=None):
        # Aquí luego cargamos dataset normal/anómalo
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)