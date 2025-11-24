# train.py
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from data_module import MVTecDataModule
from model import LitAutoEncoder

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # fijar semilla
    pl.seed_everything(cfg.seed)

    # instanciar DataModule y modelo
    data_module = MVTecDataModule(cfg.data_dir, batch_size=cfg.model.batch_size, num_workers=cfg.model.num_workers)
    data_module.setup()
    model = LitAutoEncoder(z_dim=cfg.model.z_dim, lr=cfg.model.lr, loss_fn=cfg.model.loss_fn)

    # configurar WandB
    wandb_logger = WandbLogger(project=cfg.logger.project,
                               name=cfg.logger.name,
                               save_dir=cfg.logger.save_dir,
                               log_model=cfg.logger.log_model,
                               offline=cfg.logger.offline)
    # añadir hiperparámetros a WandB
    wandb_logger.experiment.config.update({
        'z_dim': cfg.model.z_dim,
        'lr': cfg.model.lr,
        'loss_fn': cfg.model.loss_fn,
        'batch_size': cfg.model.batch_size,
        'max_epochs': cfg.trainer.max_epochs
    })

    # inicializar trainer
    trainer = Trainer(max_epochs=cfg.trainer.max_epochs,
                      accelerator=cfg.trainer.accelerator,
                      devices=cfg.trainer.devices,
                      log_every_n_steps=cfg.trainer.log_every_n_steps,
                      logger=wandb_logger)

    # entrenamiento y validación
    trainer.fit(model, datamodule=data_module)
    # test: reconstrucción de imágenes buenas y defectuosas
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()
