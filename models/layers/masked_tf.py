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

import math
from typing import Optional

import torch
import torch.nn as nn

from models.layers.single_mask import Mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, learnable=False):
        super().__init__()

        # Create a long enough position tensor
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = nn.Parameter(pe.unsqueeze(1), requires_grad=learnable)  # Shape: (max_len, 1, d_model)

    def forward(self, x):
        # x shape: (seq, batch, d_model)
        return x + self.pe[:x.size(0), :]


class MaskedTransformer(nn.Module):
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

        self.num_hidden = net_config["num_hidden"]

        self.state_mask = Mask(tot_num_states, states) \
            if len(states) > 0 else nn.Identity()

        self.signal_mask = Mask(tot_num_signals, inputs) \
            if len(inputs) > 0 else nn.Identity()

        # Positional Encoding
        self.pos_emb = PositionalEncoding(d_model=self.num_hidden)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.num_hidden,
                                                   nhead=net_config["num_heads"],
                                                   batch_first=False,
                                                   activation=net_config["activation_fn"],
                                                   dropout=net_config["dropout_prob"],
                                                   dim_feedforward=self.num_hidden * 4)

        self.net = nn.TransformerEncoder(encoder_layer, num_layers=net_config["num_layers"], enable_nested_tensor=False)

        # Transform to latent dim
        self.state_embed = nn.Linear(self.num_states, self.num_hidden)
        self.input_embed = nn.Linear(self.num_inputs, self.num_hidden)
        self.norm = nn.LayerNorm(self.num_hidden)

        # Decode
        self.decode = nn.Linear(self.num_hidden, self.num_outputs)

    def initialize_net(self, *args) -> None:
        pass

    @staticmethod
    def generate_square_subsequent_mask(size: int,
                                        device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = torch.device('cpu')

        return torch.triu(
            torch.full((size, size), float('-inf'), dtype=torch.float32, device=device),
            diagonal=1,
        )

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        states, signals = x
        x = self.state_mask(states)

        x = self.state_embed(x).view(1, -1, self.num_hidden)

        if self.controlled:
            signals = self.signal_mask(signals)
            u = self.input_embed(signals)
        else:
            u = torch.zeros(signals.size(0), signals.size(1),
                            self.num_hidden, device=signals.device)

        x = x + u

        x = self.norm(x)

        x = self.pos_emb(x)

        # Generate a causal mask
        mask = self.generate_square_subsequent_mask(x.size(0), x.device)

        out = self.net(x, mask=mask)

        out = self.decode(out)

        return out
