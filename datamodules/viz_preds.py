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


def plot_pred_data(df: pd.DataFrame, only_with_predictions=True):
    """
    Plots engine data from one CSV file. Prediction columns (ending with '_hat')
    are plotted alongside their corresponding original columns.

    Args:
        df (pd.DataFrame): The dataframe containing the predictions.
        only_with_predictions (bool): If True, only plot signals with prediction counterparts.

    Returns:
        None
    """
    t = df["time"]

    # Group columns by base name (removing '_hat' from predictions)
    columns = df.columns
    base_signals = {col.rstrip("_hat"): col for col in columns if not col.endswith("_hat")}
    prediction_signals = {col.rstrip("_hat"): col for col in columns if col.endswith("_hat")}

    # Filter for signals that have prediction counterparts, if requested
    if only_with_predictions:
        base_signals = {base: col for base, col in base_signals.items() if base in prediction_signals}
    else:
        return None

    # Determine the number of plots
    num_plots = len(base_signals)
    if num_plots == 0:
        print("No signals to plot.")
        return

    # Dynamically adjust figure height
    fig_height = max(3, num_plots // 2) * 2  # Ensure at least 2 and scale with num_plots
    plt.figure(figsize=(5, fig_height))

    # Plot each base signal with its prediction (if available)
    plot_index = 1
    for base, base_col in base_signals.items():
        if plot_index > 10:  # Safety check for grid size
            print("Warning: Too many columns to fit in the 5x2 grid. Skipping extra columns.")
            break

        plt.subplot(2, 1, plot_index)
        plt.plot(t, df[base_col], label=f"{base_col} (actual)")

        # Plot the prediction column
        if base in prediction_signals:
            pred_col = prediction_signals[base]
            plt.plot(t, df[pred_col], label=f"{pred_col} (prediction)")

        plt.title(base)
        plt.legend()
        plot_index += 1

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # check if folder 'predictions' exists
    pred_folder = os.path.join("..", "predictions")

    if not os.path.exists(pred_folder):
        raise FileNotFoundError("Folder 'predictions' does not exist.")

    catalog = 'greybox'  # depending on the model used

    # check if catalog exists within 'predictions'
    catalog_path = os.path.join(pred_folder, catalog)
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalog '{catalog}' does not exist.")

    # check if file that starts with 'test_predictions' exists within catalog
    files = os.listdir(catalog_path)
    if not any([file.startswith("test_predictions") for file in files]):
        raise FileNotFoundError("No files starting with 'test_predictions' found.")

    # get the latest file
    latest_file = max(
        (os.path.join(catalog_path, file) for file in files if file.startswith("test_predictions")),
        key=os.path.getctime
    )
    print(f"Latest file: {os.path.basename(latest_file)}")

    # load the latest file (use latest_file directly)
    df = pd.read_csv(latest_file)
    print(df.head())

    # plot the data
    plot_pred_data(df)
