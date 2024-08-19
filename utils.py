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

from typing import Callable
import yaml
import importlib
from pathlib import Path


def load_config(config: str, sub_dir="networks") -> dict:
    # check if file contains ".yml" extension
    if not config.endswith(".yml"):
        config += ".yml"

    # check if file exists in "configs/[sub_dir]":
    config_path = Path("configs") / sub_dir

    # get all files in subdirectory
    files = [f for f in config_path.iterdir() if f.is_file()]

    # check if config is any of the files
    if not any([config in f.name for f in files]):
        raise FileNotFoundError(f"Config file {config} not found.")
    else:
        config = [f for f in files if config in f.name][0]

    with open(config) as f:
        conf = yaml.safe_load(f)

    if "residual" in sub_dir:
        conf = convert_config(conf)

    return conf


def convert_config(config: dict) -> dict:
    # Convert the config file from human-readable strings to integers

    # Function to replace strings with their mapped integer values
    def replace_with_ints(lst, map_dict):
        return [map_dict[item] if item in map_dict else item for item in lst]

    sig_map = {}
    state_map = {}

    for i, (_, signal) in enumerate(config["signals"].items()):
        sig_map[signal] = i

    for i, state in enumerate(config["dynamic"].keys()):
        state_map[state] = i

    # Replace states and inputs in the dynamic section
    for key, value in config["dynamic"].items():
        value["states"] = replace_with_ints(value["states"], state_map)
        value["inputs"] = replace_with_ints(value["inputs"], sig_map)

    # Replace states and inputs in the predictors section
    for key, value in config["predictors"].items():
        value["states"] = replace_with_ints(value["states"], state_map)
        value["inputs"] = replace_with_ints(value["inputs"], sig_map)

    return config


def import_module(module_name: str) -> object:
    return importlib.import_module(module_name)


def import_from_module(module_name: str, class_name: str) -> Callable:
    module = import_module(module_name)
    return getattr(module, class_name)
