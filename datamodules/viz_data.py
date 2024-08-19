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

import os
import pandas as pd
import matplotlib.pyplot as plt

signals = {
    "intercooler_pressure": "y_p_ic",
    "intercooler_temperature": "y_T_ic",
    "intake_manifold_pressure": "y_p_im",
    "air_mass_flow": "y_W_af",
    "engine_speed": "y_omega_e",
    "throttle_position": "y_alpha_th",
    "wastegate_position": "y_u_wg",
    "injected_fuel_mass": "y_wfc",
    "ambient_temperature": "y_T_amb",
    "ambient_pressure": "y_p_amb",
}


file1 = "wltp_NF.csv"
# file2 = "wltp_NF_2.csv"
file2 = "wltp_f_waf_115.csv"

root = "../data/engine"

path1 = os.path.join(root, file1)
path2 = os.path.join(root, file2)

df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)

df1.columns = df1.columns.str.lower()
df2.columns = df2.columns.str.lower()

df1 = df1.rename(columns=signals)
df2 = df2.rename(columns=signals)

# standardize data
# df1 = (df1 - df1.mean()) / df1.std()
# df2 = (df2 - df2.mean()) / df2.std()

t1 = df1["time"]
t2 = df2["time"]

# plot data
plt.figure(figsize=(10, 10))
for i, (key, value) in enumerate(signals.items()):
    plt.subplot(5, 2, i + 1)
    plt.plot(t1, df1[value], label="df1")
    plt.plot(t2, df2[value], label="df2")
    plt.title(key)
    plt.legend()
plt.tight_layout()
plt.show()
