import lightning
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from config import get_config
from dataset import get_dataset, DataModule
from lightning_model import LightningModelWrapper
from model import GraphNN


def train():
    dataset = get_dataset()
    config = get_config()
    lightning.seed_everything(config.seed)

    model = GraphNN(config, dataset)
    lightning_model = LightningModelWrapper(model, config)
    data_module = DataModule(config, dataset)
    logger = TensorBoardLogger(config.log_dir, name=config.model)
    save_callback = ModelCheckpoint(save_weights_only=True, mode="min", monitor="train_loss")
    trainer = Trainer(
        logger=logger,
        callbacks=[save_callback],
        max_epochs=config.num_epochs,
        accelerator=config.device,
        enable_progress_bar=config.progress_bar,
    )

    trainer.fit(lightning_model, datamodule=data_module)

if __name__ == '__main__':
    train()
