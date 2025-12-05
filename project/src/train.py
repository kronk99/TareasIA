import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):

    # import tardío para evitar errores circulares
    from src.models.vae import VAE
    from src.data.mvtec_datamodule import MVTECDatamodule
    from src.utils.callbacks import get_early_stopping

    datamodule = MVTECDatamodule(**cfg.data)
    model = VAE(**cfg.model)

    trainer = pl.Trainer(**cfg.trainer, callbacks=[get_early_stopping()])
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()