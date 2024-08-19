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

from typing import Optional, Any, Callable

import torch
import torch.nn as nn

from models.layers.masked_rnn import MaskedRNN
from models.layers.masked_tf import MaskedTransformer


class DynamicBlackBox(nn.Module):
    def __init__(self,
                 num_states: int,
                 num_signals: int,
                 state_config: dict,
                 net_config: dict,
                 ) -> None:
        super().__init__()
        self.num_states = num_states
        self.num_signals = num_signals

        module = MaskedRNN if net_config["module"] == "rnn" else MaskedTransformer

        self.f = nn.ModuleList(self.create_net(state_config, net_config["net"], module))

    def create_net(self,
                   dynamic_states: dict,
                   net_config: dict,
                   module: Callable,
                   ) -> list[Any]:
        f_list = []

        for key, value in dynamic_states.items():
            inputs = value["inputs"]
            states = value["states"]

            num_outputs = 1

            if key == "latent":
                states = list(range(self.num_states))
                num_outputs = self.num_states

            f = module(self.num_states,
                       self.num_signals,
                       states,
                       inputs,
                       net_config,
                       num_outputs)

            f_list.append(f)
        return f_list

    def forward(self,
                x: torch.Tensor,
                u: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Discrete update
        if u is None:
            u = torch.zeros_like(x)

        states = []
        for f in self.f:
            state_pred = f((x, u))
            states.append(state_pred)

        states = torch.cat(states, dim=-1)
        return states

    def simulate(self,
                 xt: torch.Tensor,
                 signals: torch.Tensor,
                 seq_len: int,
                 sample_time: float
                 ) -> torch.Tensor:

        states = self(xt, signals)

        return states
