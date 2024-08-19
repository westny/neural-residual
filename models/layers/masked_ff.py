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

import torch
import torch.nn as nn

from models.layers.single_mask import Mask
from models.layers.feedforward import FeedForward


class MaskedNet(FeedForward):
    def __init__(self,
                 tot_num_states: int,
                 tot_num_signals: int,
                 states: list[int],
                 inputs: list[int],
                 net_config: dict,
                 num_outputs: int = 1) -> None:
        super().__init__(len(states) + len(inputs), num_outputs, **net_config)
        self.states = states
        self.inputs = inputs
        self.controlled = len(inputs) > 0
        self.num_outputs = num_outputs

        self.state_mask = Mask(tot_num_states, states) \
            if len(states) > 0 else nn.Identity()

        self.signal_mask = Mask(tot_num_signals, inputs) \
            if len(inputs) > 0 else nn.Identity()

    def initialize_net(self,
                       init_strategy: Callable,
                       step_size: float,
                       solver_order: int,
                       complex_poles: bool,
                       num_non_states: int = 0) -> None:
        if solver_order > 0:
            init_strategy(self.net,
                          self.num_outputs,
                          len(self.inputs),
                          step_size,
                          solver_order,
                          complex_poles,
                          n_non_states=num_non_states)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        states, signals = x
        x = self.state_mask(states)

        if self.controlled:
            signals = self.signal_mask(signals)
            x = torch.cat((x, signals), dim=-1)
        return self.net(x)
