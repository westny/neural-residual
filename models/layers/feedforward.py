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

import warnings
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self,
                 num_inputs: int,
                 num_outputs: int,
                 num_hidden: int = 8,
                 num_layers: int = 1,
                 activation_fn: str = "relu",
                 dropout_prob: float = 0.0,
                 norm_layer: str = "none",
                 **kwargs) -> None:
        super().__init__()

        self.net = self._create_net(num_inputs,
                                    num_outputs,
                                    num_hidden,
                                    num_layers,
                                    activation_fn,
                                    dropout_prob,
                                    norm_layer)

    @staticmethod
    def _create_net(num_inputs: int,
                    num_outputs: int,
                    num_hidden: int,
                    num_layers: int,
                    activation_fn: str = "relu",
                    dropout_prob: float = 0.0,
                    norm_layer: str = "none") -> nn.Sequential:
        def block(in_features: int, out_features: int) -> list:

            layers = [nn.Linear(in_features, out_features)]

            match norm_layer:
                case "layer":
                    layers.append(nn.LayerNorm(out_features))
                case "batch":
                    layers.append(nn.BatchNorm1d(out_features))
                case "group":
                    assert out_features % 2 == 0, "GroupNorm requires even number of features."
                    layers.append(nn.GroupNorm(out_features // 2, out_features))
                case "none":
                    pass
                case _:
                    warnings.warn(f"Unknown normalization layer: {norm_layer}. Using no normalization.")
                    pass

            if dropout_prob:
                layers.append(nn.Dropout(dropout_prob))

            match activation_fn:
                case "relu":
                    layers.append(nn.ReLU(inplace=True))
                case "lrelu":
                    layers.append(nn.LeakyReLU(negative_slope=1e-2, inplace=True))
                case "elu":
                    layers.append(nn.ELU(inplace=True))
                case "silu":
                    layers.append(nn.SiLU(inplace=True))
                case "gelu":
                    layers.append(nn.GELU(approximate="none"))
                case "tanh":
                    layers.append(nn.Tanh())
                case "sigmoid":
                    layers.append(nn.Sigmoid())
                case "none":
                    # Will return a linear model
                    pass
                case _:
                    warnings.warn(f"Unknown activation function: {activation_fn}. Using ReLU.")
                    layers.append(nn.ReLU(inplace=True))

            return layers

        net = nn.Sequential(
            # input layer
            *block(num_inputs, num_hidden),

            # hidden layers
            *[module for _ in range(num_layers - 1) for
              module in block(num_hidden, num_hidden)],

            # output layer
            nn.Linear(num_hidden, num_outputs),
        )

        return net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
