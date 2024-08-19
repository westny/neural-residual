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

from typing import Optional

import torch
import torch.nn as nn
from torch import func

from torchdiffeq import odeint

from initialization.stb_init import init_strategy
from models.layers.masked_ff import MaskedNet
from models.utils import get_solver_order


class NeuralODE(nn.Module):
    _solver = 'euler'
    _h = 0.01
    _opts = {}
    _atol = 1e-9
    _rtol = 1e-7

    def __init__(self,
                 num_states: int,
                 num_signals: int,
                 state_config: dict,
                 net_config: dict,
                 ) -> None:
        super().__init__()
        self.num_states = num_states
        self.num_signals = num_signals

        self.solver = net_config["solver"]
        self.step_size = net_config["step_size"]
        self.stability_init = net_config["stability_init"]
        self.complex_poles = net_config["complex_poles"]
        self.solver_order = get_solver_order(self.solver, self.stability_init)

        self.f = nn.ModuleList(self.create_net(state_config, net_config["net"]))

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, method):
        assert method in ('euler', 'heun', 'midpoint', 'rk3', 'ssprk3', 'rk4', 'adaptive_heun', 'dopri5')
        self._solver = method
        if method in ('adaptive_heun', 'dopri5'):
            self._opts = {}

    @property
    def opts(self):
        return self._opts

    @opts.setter
    def opts(self, options):
        assert options is dict
        self._opts = options

    @property
    def step_size(self):
        return self._h

    @step_size.setter
    def step_size(self, h):
        self._h = h
        if self._solver in ('euler', 'midpoint', 'rk3', 'rk4'):
            self._opts = {'step_size': self._h}
        else:
            self._opts = {}

    @property
    def atol(self):
        return self._atol

    @atol.setter
    def atol(self, atol):
        self._atol = atol

    @property
    def rtol(self):
        return self._rtol

    @rtol.setter
    def rtol(self, rtol):
        self._rtol = rtol

    def create_net(self,
                   dynamic_states: dict,
                   net_config: dict
                   ) -> list[MaskedNet]:
        f_list = []

        for key, value in dynamic_states.items():
            inputs = value["inputs"]
            states = value["states"]

            num_outputs = 1
            num_non_states = len(states) - 1

            if key == "latent":
                states = list(range(self.num_states))
                num_outputs = self.num_states
                num_non_states = 0

            f = MaskedNet(self.num_states,
                          self.num_signals,
                          states,
                          inputs,
                          net_config,
                          num_outputs)

            f.initialize_net(init_strategy,
                             self.step_size,
                             self.solver_order,
                             self.complex_poles,
                             num_non_states)

            f_list.append(f)
        return f_list

    def update(self,
               t: torch.Tensor,
               X: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # x, u = X
        dx = self.state_transition(X)
        du = torch.zeros_like(X[-1])
        return dx, du

    def state_transition(self, X: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        dxs = []
        for f in self.f:
            dx = f(X)
            dxs.append(dx)
        dxs = torch.cat(dxs, dim=-1)
        return dxs

    @torch.inference_mode(False)
    def state_jacobian(self,
                       x: torch.Tensor,
                       u: Optional[torch.Tensor] = None
                       ) -> torch.Tensor:
        batch_size, feat_dim = x.shape
        if u is None:
            u = torch.zeros_like(x)
        # calculates the Jacobian of the state transition function w.r.t. the states X
        jacobian = func.vmap(func.jacrev(self.state_transition, argnums=0))(x, u)
        jac_mat = jacobian.view(batch_size, feat_dim, feat_dim)
        return jac_mat

    def forward(self,
                t: torch.Tensor,
                x: torch.Tensor,
                u: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Discrete update
        if u is None:
            u = torch.zeros_like(x)

        x_next, _ = odeint(self.update, (x, u), t=t,
                           rtol=self._rtol, atol=self._atol,
                           method=self._solver, options=self._opts)

        return x_next

    def simulate(self,
                 xt: torch.Tensor,
                 signals: torch.Tensor,
                 seq_len: int,
                 sample_time: float
                 ) -> torch.Tensor:

        # Store predictions
        states = []

        # Initialize time vector
        t_vec = torch.tensor([0., sample_time], device=signals.device)

        for t in range(seq_len):
            # Update states using dynamic model
            xt = self(t_vec, xt, signals[t])
            xt = xt[-1]
            states.append(xt)

            # We detach xt to avoid backpropagation through all time steps
            # xt = xt.detach()

            # Update time vector
            t_vec = t_vec + sample_time

        states = torch.stack(states, dim=0)

        return states
