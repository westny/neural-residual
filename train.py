# Copyright 2024, Theodor Westny. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import warnings
from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from utils import load_config, import_from_module
from arguments import args


torch.set_float32_matmul_precision('medium')
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", ".*Checkpoint directory*")

config = load_config(args.network_conf)
residual = load_config(args.residual_conf, "residuals")

TorchModel = import_from_module(config["model"]["module"], config["model"]["class"])
LitDataModule = import_from_module(config["datamodule"]["module"], config["datamodule"]["class"])
LitModel = import_from_module(config["litmodule"]["module"], config["litmodule"]["class"])


def main(save_name: str):
    ds = config["dataset"]
    ckpt_path = Path("saved_models") / ds / save_name

    # Check if checkpoint exists and the overwrite flag is not set
    if ckpt_path.with_suffix(".ckpt").exists() and not args.overwrite:
        ckpt = str(ckpt_path) + ".ckpt"
    else:
        ckpt = None

    # Setup callbacks list for training
    callback_list = []
    if args.store_model:
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(ckpt_path.parent),  # Using parent directory of the checkpoint
            filename=save_name,
            monitor="val_loss",
            mode="min"
        )
        callback_list.append(checkpoint_callback)

    model = TorchModel(config["model"], residual)
    lit_model = LitModel(model, config["training"])
    datamodule = LitDataModule(args, config["datamodule"], residual)

    try:
        if torch.cuda.is_available() and args.use_cuda:
            devices = -1 if torch.cuda.device_count() > 1 else 1
            strategy = 'ddp' if devices == -1 else 'auto'
            accelerator = "auto"
        else:
            devices, strategy, accelerator = 1, 'auto', "cpu"

        if args.dry_run or not args.use_logger:
            logger = False
        else:
            run_name = f"{save_name}_{time.strftime('%d-%m_%H:%M:%S')}"
            logger = WandbLogger(project=f"engine-residual", name=run_name)

        clip_val = config["training"]["clip"] if config["training"]["clip"] else None

        # Trainer configuration
        trainer = Trainer(max_epochs=config["training"]["epochs"],
                          logger=logger,
                          devices=devices,
                          strategy=strategy,
                          accelerator=accelerator,
                          callbacks=callback_list,
                          gradient_clip_val=clip_val,
                          fast_dev_run=args.dry_run,
                          enable_checkpointing=args.store_model)

        # Model fitting
        trainer.fit(lit_model, datamodule=datamodule, ckpt_path=ckpt)

    except Exception as e:
        print(f"An error occurred during setup: {e}")


if __name__ == "__main__":
    seed_everything(args.main_seed, workers=True)

    if args.scnd_seed is not None:
        config["datamodule"]["data_seed"] = args.scnd_seed

    if args.dry_run:
        config["datamodule"]["batch_size"] = 8

    if args.train_file is not None:
        config["datamodule"]["train_file"] = args.train_file

    if args.test_file is not None:
        config["datamodule"]["test_file"] = args.test_file

    if args.add_name:
        add_name = f"_{args.add_name}"
    else:
        add_name = ""

    model_name = f'{residual["description"]}_{config["architecture"]}{add_name}'

    print(f'Preparing to train model: {model_name}')

    main(model_name)
