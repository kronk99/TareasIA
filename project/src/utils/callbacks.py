from pytorch_lightning.callbacks import EarlyStopping

def get_early_stopping():
    return EarlyStopping(monitor="val_loss", patience=10, mode="min")