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
from typing import Optional

import requests
import zipfile
import pandas as pd

TRAIN_URL = "https://vehsys.gitlab-pages.liu.se/diagnostic_competition/competition/training_data/trainingdata.zip"
TEST_URL = "https://vehsys.gitlab-pages.liu.se/diagnostic_competition/competition/test_data/testdata.zip"


def process_data(file: str = "wltp_NF.csv",
                 root: str = "./data/engine",
                 train: bool = True,
                 download: bool = True,
                 remap: Optional[dict] = None,
                 zero_out: Optional[list[str]] = None,
                 single_file: bool = False) -> pd.DataFrame:
    # Check if data exists
    root = os.path.join(root, "train" if train else "test")

    if not os.path.exists(root):
        os.makedirs(root)

    file_path = os.path.join(root, file)

    if not os.path.exists(file_path):
        if download:
            print(f'Downloading {file}...')
            url = TRAIN_URL if train else TEST_URL

            dl_path = os.path.join(root, 'data.zip')

            download_file(url, dl_path)
            if single_file:
                extract_specific_file(dl_path, file, root, train)
            else:
                extract_all_files(dl_path, root, train)
            os.remove(f'{root}/data.zip')
        else:
            raise FileNotFoundError(file)

    # Load data from engine
    try:
        df = pd.read_csv(file_path, engine='pyarrow')
    except ImportError:
        df = pd.read_csv(file_path)

    # Convert column names to lowercase
    df.columns = df.columns.str.lower()

    # index by signal_csv keys
    if remap is not None:
        df = df.rename(columns=remap)

    # Zero out specific columns
    if zero_out is not None:
        df[zero_out] = 0.

    return df


def download_file(url, filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size_in_bytes = int(r.headers.get('content-length', 0))
        total_size_in_gb = total_size_in_bytes / (1024 * 1024 * 1024)  # Convert to GB
        block_size = 1 * 1024 * 1024  # 10 MB
        progress = 0

        with open(filename, 'wb') as file:
            for data in r.iter_content(block_size):
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
                progress += len(data)
                downloaded_in_gb = progress / (1024 * 1024 * 1024)  # Convert to GB
                done = int(50 * progress / total_size_in_bytes)
                print(f"\r[{'=' * done}{' ' * (50 - done)}] {downloaded_in_gb:.2f}/{total_size_in_gb:.2f} GB", end='')
        print()


def extract_specific_file(zip_filename, target_filename, extract_to_folder='.', train=True):
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        # List all file names in the zip
        file_names = zip_ref.namelist()

        # Construct the full path of the file in the zip
        folder = "trainingdata" if train else "testdata"
        full_path_in_zip = os.path.join(folder, target_filename)

        if full_path_in_zip in file_names:
            # Extract the file data
            file_data = zip_ref.read(full_path_in_zip)

            # Define the full path for the extracted file
            extracted_file_path = os.path.join(extract_to_folder, target_filename)

            # Write the extracted file
            with open(extracted_file_path, 'wb') as f:
                f.write(file_data)

            print(f"Extracted {target_filename} to {extracted_file_path}")
        else:
            print(f"{target_filename} not found in the zip file.")


def extract_all_files(zip_filename, extract_to_folder='.', train=True):
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        # List all file names in the zip
        file_names = zip_ref.namelist()

        # Construct the full path of the file in the zip
        folder = "trainingdata" if train else "testdata"

        file_names = [file.split('/')[-1] for file in file_names if ".csv" in file and "__MACOSX" not in file]

        for file in file_names:
            # Extract the file data
            full_path_in_zip = os.path.join(folder, file)

            file_data = zip_ref.read(full_path_in_zip)

            # Define the full path for the extracted file
            extracted_file_path = os.path.join(extract_to_folder, file)

            # Write the extracted file
            with open(extracted_file_path, 'wb') as f:
                f.write(file_data)

            print(f"Extracted {file} to {extracted_file_path}")


if __name__ == '__main__':
    y, _, conf = process_data()
