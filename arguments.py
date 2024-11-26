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

from argparse import ArgumentParser, ArgumentTypeError


def str_to_bool(value):
    """Used for boolean arguments in argparse; avoiding `store_true` and `store_false`."""
    true_vals = ("yes", "true", "t", "y", "1")
    false_vals = ("no", "false", "f", "n", "0")
    if isinstance(value, bool):
        return value
    if value.lower() in true_vals:
        return True
    elif value.lower() in false_vals:
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


parser = ArgumentParser(description='Engine residual arguments')

# Program arguments
parser.add_argument('--main-seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--scnd-seed', type=int, default=None,
                    help='re-random seed. Used for different data splits (default: None)')
parser.add_argument('--num-workers', type=int, default=4,
                    help='number of workers in dataloader (default: 4)')
parser.add_argument('--use-logger', type=str_to_bool, default=False,
                    const=True, nargs="?", help='if logger should be used (default: False)')
parser.add_argument('--use-cuda', type=str_to_bool, default=True,
                    const=True, nargs="?", help='if cuda exists and should be used (default: True)')
parser.add_argument('--balance-data', type=str_to_bool, default=False,
                    const=True, nargs="?", help='if data should be balanced (default: False)')
parser.add_argument('--noise-level', type=float, default=0.0,
                    help='noise level for data augmentation (default: 0.0)')
parser.add_argument('--store-model', type=str_to_bool, default=True,
                    const=True, nargs="?", help='if checkpoints should be stored (default: False)')
parser.add_argument('--overwrite', type=str_to_bool, default=True,
                    const=True, nargs="?", help='overwrite if model exists (default: True)')
parser.add_argument('--add-name', type=str, default="",
                    help='additional string to add to save name (default: "")')
parser.add_argument('--dry-run', type=str_to_bool, default=True,
                    const=True, nargs="?", help=' debug mode, runs one fwd pass (default: True)')
parser.add_argument('--pin-memory', type=str_to_bool, default=True,
                    const=True, nargs="?", help='if the data should be pinned to memory (default: True)')
parser.add_argument('--persistent-workers', type=str_to_bool, default=True,
                    const=True, nargs="?", help='if the workers should be persistent (default: True)')
parser.add_argument('--network-conf', type=str, default="neuralode.yml",
                    help='config file for network configuration (default: neuralode.yml)')
parser.add_argument('--residual-conf', type=str, default="greybox.yml",
                    help='config file for residual configuration (default: greybox.yml)')
parser.add_argument('--train-file', type=str, default=None,
                    help='file for training data (default: wltp_NF.csv)')
parser.add_argument('--test-file', type=str, default=None,
                    help='file for testing data (default: wltp_NF_2.csv)')

args = parser.parse_args()
