import torch
from torch import nn
import pytorch_lightning as pl
from torchvision.utils import make_grid
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

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
        # Selección de función de pérdida, agregar aca las otras funciones de perdida
        if loss_fn.upper() == 'L1':
            self.criterion = self.l1_loss
        elif loss_fn.upper() == 'L2':
            self.criterion = nn.MSELoss()  # L2
        elif loss_fn.upper()  == 'SSIM':
            # esto es 1- ssim segun lo investigado
            self.criterion = self.ssim_loss
        else:
            raise ValueError(f"Funcion de pérdida {loss_fn} no incluida")


    # ------------------------------------------------------------------
    # Loss functions
    def l1_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        #funcion de perdida L1, compara los pixeles entre los tensores y 
        #calcula la la media del valor absoluto entre valor origial y valor reconstruido
        #valor original es y, valor reconstruido y_hat
        return torch.mean(torch.abs(y_hat - y))

    def _gaussian_window(self, window_size: int, sigma: float, channels: int) -> torch.Tensor:
        """crea una ventana gaussiana para ssim
        Parametros
        window_size : int
            tamaño del sliding window.  tamaño tipico  11 (usado en otras pruebas es 2)
        sigma : desviacion estandar del kernel gaussiano
        channels : numero de canales.

        Retorna un tensor tipo
        torch.Tensor
        forma: (channels, 1, window_size, window_size).
        """
        # kernel gaussiano de una dimension 
        coords = torch.arange(window_size, dtype=torch.float32, device=self.device) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        # genera un kernel 2d
        window_2d = g[:, None] * g[None, :]
        window_2d = window_2d.unsqueeze(0).unsqueeze(0)
        window_2d = window_2d.expand(channels, 1, window_size, window_size)
        return window_2d

    def _ssim_index(self, x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
       
        #asegura que los tensores sean del mismo tamaño
        assert x.shape == y.shape, "los tensores deben de ser del mismo tamaño"
        N, C, H, W = x.shape
        # genera (C,1,w,w) genera un kernel gaussiano en el mismo dispositivo
        window = self._gaussian_window(window_size, sigma, C)
        # medias locales por convolucion:
        mu_x = F.conv2d(x, window, padding=window_size // 2, groups=C)
        mu_y = F.conv2d(y, window, padding=window_size // 2, groups=C)
        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy = mu_x * mu_y
        # variaciones y covariaciones locales
        sigma_x_sq = F.conv2d(x * x, window, padding=window_size // 2, groups=C) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, window, padding=window_size // 2, groups=C) - mu_y_sq
        sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=C) - mu_xy
        #los valores K1=0.01 and K2=0.03 estabilizan bien la funcion de perdida
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2
        # SSIM map 
        num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        den = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        ssim_map = num / den
        # promedio espacial de dimensiones
        ssim_per_batch = ssim_map.view(N, C, -1).mean(dim=[1, 2])
        return ssim_per_batch
#definicion de la funcion de perdida
    def ssim_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        #esto computa ssim_loss como 1- promedio ssim
        #considere y_hat como valor anterior 
        ssim_score = self._ssim_index(y_hat, y).mean()
        return 1.0 - ssim_score

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
