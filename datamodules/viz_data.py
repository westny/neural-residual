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


def plot_engine_data(file, root, signals=None, standardize=False, file2=None):
    """
        Plots engine data from one or two CSV files.

        Args:
            file (str): The name of the primary CSV file.
            root (str): The root directory containing the CSV files.
            signals (dict): A dictionary mapping signal names to column names in the CSV files.
            standardize (bool, optional): Whether to standardize the data before plotting. Defaults to False.
            file2 (str, optional): The name of the secondary CSV file. Defaults to None.

        Returns:
            None
    """

    path = os.path.join(root, file)
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()

    if signals is not None:
        df = df.rename(columns=signals)
    if standardize:
        df = (df - df.mean()) / df.std()

    if file2 is not None:
        path2 = os.path.join(root, file2)
        df2 = pd.read_csv(path2)
        df2.columns = df2.columns.str.lower()
        if signals is not None:
            df2 = df2.rename(columns=signals)
        if standardize:
            df2 = (df2 - df2.mean()) / df2.std()
        t2 = df2["time"]
    else:
        df2 = None
        t2 = None

    t = df["time"]
    plt.figure(figsize=(10, 10))
    for i, (key, value) in enumerate(signals.items()):
        plt.subplot(5, 2, i + 1)
        plt.plot(t, df[value], label="df1")
        if file2 is not None:
            plt.plot(t2, df2[value], label="df2")
        plt.title(key)
        plt.legend()
    plt.tight_layout()
    plt.show()


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


if __name__ == "__main__":
    file1 = "wltp_NF.csv"
    file2 = "wltp_f_waf_110.csv"
    root = "../data/engine/train"

    plot_engine_data(file1, root, signals, standardize=False)
    # plot_engine_data(file2, root, signals, standardize=False, file2=file2)
