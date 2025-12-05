import pytorch_lightning as pl

class VAE(pl.LightningModule):
    def __init__(self, latent_dim=64, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass