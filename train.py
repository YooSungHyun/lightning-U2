import os
import shutil
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from datetime import timedelta
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from models.u2.lightningmodule import BiU2
from models.u2.datamodule import BiU2DataModule
from simple_parsing import ArgumentParser
from arguments.training_args import TrainingArguments
from utils.comfy import dataclass_to_namespace


def main(hparams):
    wandb_logger = WandbLogger(project="U2_2plus", name="default", save_dir="./")
    pl.seed_everything(hparams.seed)
    os.makedirs(hparams.output_dir, exist_ok=True)
    hparams.logger = wandb_logger

    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.output_dir,
        save_top_k=1,
        mode="min",
        monitor="val_cer",
        filename="lightning-template-{epoch:02d}-{val_cer:.4f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    hparams.callbacks = [checkpoint_callback, lr_monitor]

    if hparams.accelerator == "cpu" and hparams.valid_on_cpu is True:
        print("If you run on cpu, valid must go on cpu, It set automatically")
        hparams.valid_on_cpu = False
    elif hparams.strategy == "ddp":
        hparams.strategy = DDPStrategy(timeout=timedelta(days=30))
    elif hparams.strategy == "deepspeed_stage_2":
        if hparams.deepspeed_config is not None:
            from pytorch_lightning.strategies import DeepSpeedStrategy

            hparams.strategy = DeepSpeedStrategy(config=hparams.deepspeed_config)
    elif hparams.accelerator != "cpu" and (hparams.strategy is not None and "deepspeed" in hparams.strategy):
        raise NotImplementedError("If you want to another deepspeed option and config, PLZ IMPLEMENT FIRST!!")
    trainer = pl.Trainer.from_argparse_args(hparams)

    datamodule = BiU2DataModule(hparams)
    model = BiU2(hparams)
    # tokenizer and model config save
    shutil.copyfile(hparams.model_config, os.path.join(hparams.output_dir, "config.yaml"))
    model.tokenizer.save_pretrained(hparams.output_dir)
    wandb_logger.watch(model, log="all")
    trainer.fit(model, datamodule=datamodule)
    checkpoint_callback.best_model_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_arguments(TrainingArguments, dest="training_args")
    args = parser.parse_args()
    args = dataclass_to_namespace(args, "training_args")
    main(args)
