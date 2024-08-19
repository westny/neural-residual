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


def get_solver_order(solver: str, stability_init: bool = False) -> int:
    if not stability_init:
        return 0

    match solver:
        case "euler":
            return 1
        case "midpoint":
            return 2
        case "heun":
            return 2
        case "adaptive_heun":
            return 2
        case "rk3":
            return 3
        case "ssprk3":
            return 3
        case "rk4":
            return 4
        case "dopri5":
            return 4
        case _:
            warnings.warn(f"Unknown solver: {solver}. Using order 0.")
            return 0


def get_unique_inputs(residual: dict) -> list[int]:
    config = residual["dynamic"]
    inputs = []
    for _, value in config.items():
        for key in value.keys():
            if key.startswith("inputs"):
                inputs.extend(value[key])
    return list(set(inputs))


def get_num_states(residual: dict) -> int:
    num_states = len(residual["dynamic"])
    if "latent" in residual["dynamic"].keys():
        num_states = residual["dynamic"]["latent"]["num_latents"]
    return num_states
