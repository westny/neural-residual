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

import os
import warnings
from pathlib import Path

import torch
from torch.multiprocessing import set_sharing_strategy
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import Logger, CSVLogger

from utils import load_config, import_from_module
from arguments import args


torch.set_float32_matmul_precision('medium')
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", ".*Checkpoint directory*")

set_sharing_strategy('file_system')

config = load_config(args.network_conf)
residual = load_config(args.residual_conf, "residuals")

TorchModel = import_from_module(config["model"]["module"], config["model"]["class"])
LitDataModule = import_from_module(config["datamodule"]["module"], config["datamodule"]["class"])
LitModel = import_from_module(config["litmodule"]["module"], config["litmodule"]["class"])


def main(save_name: str):
    ds = config["dataset"]
    path = Path("saved_models") / ds / save_name

    # Check if checkpoint exists
    if path.with_suffix(".ckpt").exists():
        ckpt = path.with_suffix(".ckpt")
    elif path.with_name(path.name + "-v1").with_suffix(".ckpt").exists():
        ckpt = path.with_name(path.name + "-v1").with_suffix(".ckpt")
    else:
        if not args.dry_run:
            raise NameError(f"Could not find model with name: {save_name}")

    # Determine the number of devices, and accelerator
    if torch.cuda.is_available() and args.use_cuda:
        devices, accelerator = -1,  "auto"
    else:
        devices, accelerator = 1, "cpu"

    # Setup logger
    logger: bool | Logger
    if args.dry_run:
        logger = False
        args.small_ds = True
    elif not args.use_logger:
        logger = False
    else:
        logger = CSVLogger(save_dir=os.path.join("lightning_logs", ds), name=save_name)

    model = TorchModel(config["model"], residual)
    lit_model = LitModel(model, config["training"])
    datamodule = LitDataModule(args, config["datamodule"], residual)

    # Load checkpoint into model
    try:
        ckpt_dict = torch.load(ckpt)
    except UnboundLocalError:
        if not args.dry_run:
            raise FileNotFoundError(f"Could not find checkpoint: {ckpt}")
    else:
        print(f"Loading checkpoint: {ckpt}")
        lit_model.load_state_dict(ckpt_dict["state_dict"], strict=False)

    # Setup trainer
    trainer = Trainer(accelerator=accelerator, devices=devices, logger=logger)

    # Start testing
    trainer.test(lit_model, datamodule=datamodule, verbose=True)


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

    # check if args.residual_conf ends with .yml, if so, remove it
    if args.residual_conf.endswith('.yml'):
        save_catalog = args.residual_conf[:-4] + add_name
    else:
        save_catalog = args.residual_conf + add_name

    config['training']['catalog'] = save_catalog

    model_name = f'{residual["description"]}_{config["architecture"]}{add_name}'

    print(f'Preparing to evaluate model: {model_name}')

    main(model_name)
