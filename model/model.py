import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger


def train(predictor, device, loader, num_epochs, batch_size, learning_rate, checkpoint_dir, checkpoint_freq=50):
    logger = TensorBoardLogger("runs", name="tigerlake")
    predictor = predictor.to(device)
    if device.type == 'cuda':
        accelerator = "gpu"
        #strategy = "deepspeed_stage_2_offload"
        strategy = "auto"
        precision = 32
    else:
        accelerator = "cpu"
        strategy = "auto"
        precision = 32
    trainer = pl.Trainer(max_epochs=num_epochs, logger=logger, strategy=strategy, accelerator=accelerator, precision=precision)
    trainer.fit(predictor, train_dataloaders=loader)
