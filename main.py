import torch
from torch import nn
import pytorch_lightning as pl
from torchvision.utils import make_grid
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import kornia


class SSIM1CombinadaL1(nn.Module):
    """
    Función de pérdida combinada: SSIM + L1
    Combina similitud estructural (SSIM) con error absoluto (L1).
    """
    def __init__(self, window_size=11):
        super().__init__()
        self.ssim = kornia.losses.SSIMLoss(window_size=window_size)
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        """
        Args:
            pred: Tensor (B, C, H, W) - predicción
            target: Tensor (B, C, H, W) - ground truth
        Returns:
            Pérdida combinada (escalar)
        """
        loss_ssim = self.ssim(pred, target)
        loss_l1 = self.l1(pred, target)
        return loss_ssim + loss_l1


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, z_dim=256, lr=1e-3, loss_fn='L1'):
        super().__init__()
        self.save_hyperparameters()
        # Esto es el encoder,debe de cambiarse por los hiperparametros de las herramientas
        #si no definir como variables
        #autoencoder clasico, stride de 2
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        # Bottleneck
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(512 * 8 * 8, z_dim)  # 128x128, 8x8 después de 4 convoluciones con stride 2
        self.fc_decode = nn.Linear(z_dim, 512 * 8 * 8)
        # parte del decoder 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() #uso de sigmoide en lugar de tahn
        )
        # Selección de función de pérdida
        if loss_fn.upper() == 'L1':
            self.criterion = nn.L1Loss()
        elif loss_fn.upper() == 'L2':
            self.criterion = nn.MSELoss()  # L2
        elif loss_fn.upper() == 'SSIM':
            self.criterion = kornia.losses.SSIMLoss(window_size=11)
        elif loss_fn.upper() == 'SSIM_L1':
            self.criterion = SSIM1CombinadaL1()
        else:
            raise ValueError(f"Funcion de pérdida {loss_fn} no incluida. Opciones: L1, L2, SSIM, SSIM_L1")

    def encode(self, x):
        h = self.encoder(x)
        h = self.flatten(h)
        z = self.fc_mu(h)
        return z

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 512, 8, 8)
        x_hat = self.decoder(h)
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
    #error aca , corregir
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = self.criterion(x_hat, x)
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = self.criterion(x_hat, x)
        self.log('val/loss', loss, prog_bar=True)
        # registrar algunos ejemplos para reconstrucción
        if batch_idx == 0:
            # tomar las primeras 16 imágenes
            images = x[:16]
            reconstructions = x_hat[:16]
            # ensamblar rejillas para visualización
            grid = make_grid(torch.cat([images, reconstructions], dim=0), nrow=16)
            self.logger.log_image(key='reconstructions', images=[grid])
        return {'val_loss': loss, 'embeddings': self.encode(x)}

    def validation_epoch_end(self, outputs):
        # Crear t-SNE del espacio latente
        embeddings = torch.cat([o['embeddings'] for o in outputs], dim=0)
        embeddings_np = embeddings.detach().cpu().numpy()
        # Usar t‑SNE para reducir a 2D
        tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca')
        embeddings_2d = tsne.fit_transform(embeddings_np)
        # Crear gráfico (cambiarlo a wandb)
        fig, ax = plt.subplots()
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=3, alpha=0.6)
        ax.set_title('t‑SNE del vector latente')
        # Registrar en WandB
        self.logger.log_image(key='tsne_latent', images=[fig])
        plt.close(fig)

    def configure_optimizers(self): #el tipo de optimizador es adam
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

