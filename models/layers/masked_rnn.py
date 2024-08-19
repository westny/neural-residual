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

from models.layers.single_mask import Mask


class MaskedRNN(nn.Module):
    def __init__(self,
                 tot_num_states: int,
                 tot_num_signals: int,
                 states: list[int],
                 inputs: list[int],
                 net_config: dict,
                 num_outputs: int = 1) -> None:
        super().__init__()
        self.states = states
        self.inputs = inputs
        self.controlled = len(inputs) > 0

        self.num_outputs = num_outputs
        self.num_states = len(states)
        self.num_inputs = max(1, len(inputs))

        self.state_mask = Mask(tot_num_states, states) \
            if len(states) > 0 else nn.Identity()

        self.signal_mask = Mask(tot_num_signals, inputs) \
            if len(inputs) > 0 else nn.Identity()

        self.num_hidden = net_config["num_hidden"]
        self.num_layers = net_config["num_layers"]
        self.rnn = net_config["rnn_type"]

        self.embed = nn.Linear(self.num_states, self.num_hidden * self.num_layers)
        self.net = self.select_rnn(self.rnn, net_config["dropout_prob"])
        self.decode = nn.Linear(self.num_hidden, self.num_outputs)

    def select_rnn(self, layer_type="gru", dropout_prob: float = 0.0) -> nn.Module:
        match layer_type:
            case "elman":
                return nn.RNN(self.num_inputs, self.num_hidden,
                              num_layers=self.num_layers, batch_first=False, dropout=dropout_prob)
            case "lstm":
                return nn.LSTM(self.num_inputs, self.num_hidden,
                               num_layers=self.num_layers, batch_first=False, dropout=dropout_prob)
            case "gru":
                return nn.GRU(self.num_inputs, self.num_hidden,
                              num_layers=self.num_layers, batch_first=False, dropout=dropout_prob)
            case _:
                warnings.warn(f"RNN type {layer_type} not recognized. Defaulting to GRU.")
                return nn.GRU(self.num_inputs, self.num_hidden,
                              num_layers=self.num_layers, batch_first=False, dropout=dropout_prob)

    def initialize_net(self, *args) -> None:
        pass

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        states, signals = x
        x = self.state_mask(states)

        h0 = self.embed(x)
        h0 = h0.view(-1, self.num_hidden, self.num_layers)
        h0 = h0.permute(-1, 0, 1)

        if self.controlled:
            signals = self.signal_mask(signals)
        else:
            signals = torch.zeros_like(signals[..., :1])

        if self.rnn == "lstm":
            c0 = torch.zeros_like(h0)
            h0 = (h0, c0)

        o, _ = self.net(signals, h0)

        out = self.decode(o)

        return out
