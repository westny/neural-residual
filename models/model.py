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

import torch
import torch.nn as nn

from models.modules.encoder import Encoder
from models.modules.dyn_factory import DynamicFactory
from models.modules.predictor import Predictor
from models.utils import get_unique_inputs, get_num_states


class ResidualPredictor(nn.Module):
    def __init__(self,
                 config: dict,
                 residual: dict) -> None:
        super().__init__()

        self.variational = config["vae"]
        self.num_signals = len(residual['signals'])
        self.num_states = get_num_states(residual)
        self.used_signals = get_unique_inputs(residual)

        # Encoder model, only used if variational is True
        self.g = Encoder(self.num_states,
                         self.num_signals,
                         self.used_signals,
                         config["encoder_model"],
                         self.variational)

        # Dynamic equations, defined by residual config
        self.f = DynamicFactory.create_dynamic(self.num_states,
                                               self.num_signals,
                                               residual["dynamic"],
                                               config["dynamic_model"])

        # Predictor models, defined by residual config
        self.h = Predictor(self.num_states,
                           self.num_signals,
                           residual["predictors"],
                           config["predictor_model"])

    def sample(self, u: torch.Tensor, training: bool = True):
        # z0 is a zero tensor if variational is False
        z0, q = self.g(u, training=training)
        return z0, q

    def forward(self,
                signals: torch.Tensor,
                seq_len: int,
                sample_time: float = 0.05,
                training: bool = True):
        """
        Parameters
        ----------
        signals: torch.Tensor [seq_len, batch_size, num_signals]
        seq_len: int []
        sample_time: float []
        training: bool []

        Returns
        -------
        y_pred: torch.Tensor [seq_len, batch_size, num_outputs]
        states: torch.Tensor [seq_len, batch_size, num_states]
        q: torch.distributions.Normal / None

        """

        # Sample initial states from encoder model
        z0, q = self.sample(signals[0], training=training)

        # Simulate states using dynamic model
        states = self.f.simulate(z0, signals, seq_len, sample_time)

        # Predict using predictor model and states
        y_pred = self.h((states, signals))

        return y_pred, states, q
