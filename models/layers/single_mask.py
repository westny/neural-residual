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


class Mask(nn.Module):
    def __init__(self,
                 tot_num: int,
                 inputs: list[int]) -> None:
        super().__init__()
        in_dim = tot_num
        out_dim = len(inputs)

        assert out_dim <= in_dim, "incompatible mask dimensions"

        if out_dim == 0:
            self.mask = None
        else:
            self.mask = nn.Parameter(torch.zeros(in_dim, out_dim), requires_grad=False)

            i = 0
            for si in inputs:
                self.mask[si, i] = 1.
                i += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask is None:
            return x
        else:
            return x @ self.mask


if __name__ == "__main__":
    n = 10
    signals = [0, 1, 4, 5, 8]
    mask = Mask(n, signals)

    x = torch.ones((1, n)).cumsum(dim=-1)
    y = mask(x)

    print(x, y)
