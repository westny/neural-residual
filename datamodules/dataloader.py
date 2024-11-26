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
import numpy as np
import lightning.pytorch as pl

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from datamodules.dataset import MTSDataset
from datamodules.process import process_data


def standardize(data, mean, std):
    return (data - mean) / std


class LitDataModule(pl.LightningDataModule):
    train_inp = None
    train_trg = None
    val_inp = None
    val_trg = None
    test_inp = None
    test_trg = None

    mean = 0.
    std = 1.

    def __init__(self,
                 args: ArgumentParser,
                 config: dict,
                 residual: dict) -> None:
        super().__init__()
        self.seed = config["data_seed"]
        self.batch_size = config["batch_size"]
        self.segment_len = config["segment_len"]
        self.seq_len = config["sequence_len"]
        self.test_seq_len = config["test_sequence_len"]
        self.sample_time = config["sample_time"]
        self.signals = list(residual['signals'].values())
        self.targets = list(residual['predictors'].keys())
        self.train_file = config["train_file"]
        self.test_file = config["test_file"]

        self.balance = args.balance_data
        self.noise_level = args.noise_level
        self.n_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.persistent = args.persistent_workers

        assert len(self.signals) > 0, "No signals found in the residual dictionary"
        assert len(self.targets) > 0, "No targets found in the residual dictionary"

        self.train_data = process_data(file=self.train_file,
                                       root=config["root"],
                                       train=True,
                                       download=True,
                                       remap=residual["signals"],
                                       zero_out=residual["zeroed_signals"])

        self.test_data = process_data(file=self.test_file,
                                      root=config["root"],
                                      train=False,
                                      download=True,
                                      remap=residual["signals"],
                                      zero_out=residual["zeroed_signals"])

        self.train_val_split()
        self.test_split()

    def train_val_split(self):
        inp_indices = [i for i, c in enumerate(self.train_data.columns) if c in self.signals]
        trg_indices = [i for i, c in enumerate(self.train_data.columns) if c in self.targets]

        data = self.train_data.to_numpy()

        processed_data = self.process_data(data, self.segment_len, self.seq_len, trg_indices, inp_indices, self.seed)

        for key, value in processed_data.items():
            setattr(self, key, value)

    def test_split(self):
        inp_indices = [i for i, c in enumerate(self.test_data.columns) if c in self.signals]
        trg_indices = [i for i, c in enumerate(self.test_data.columns) if c in self.targets]

        data = self.test_data.to_numpy()

        # Standardize all data
        test_data_standardized = standardize(data, self.mean, self.std)

        N = test_data_standardized.shape[0]

        if self.test_seq_len > 0:
            assert N > self.test_seq_len, f"Test data length {N} is less than sequence length {self.test_seq_len}"

            # Calculate the number of segments
            segments = []

            # Split data into segments of length seq_len (with maximum overlap)
            for i in range(N - self.test_seq_len + 1):
                segments.append(test_data_standardized[i:i + self.test_seq_len])

        else:
            # No sub-segmentation, use the entire data as a single segment
            segments = [test_data_standardized]

        # Separate input and target data
        test_inp = [seg[:, inp_indices] for seg in segments]
        test_trg = [seg[:, trg_indices] for seg in segments]

        # Convert to torch tensors
        test_input_tensor = torch.from_numpy(np.stack(test_inp, axis=0)).float()
        test_target_tensor = torch.from_numpy(np.stack(test_trg, axis=0)).float()

        self.test_inp = test_input_tensor
        self.test_trg = test_target_tensor

    def process_data(self, data, l, n, trg_cols_indices, inp_cols_indices, seed=0):
        # Split data into l-length segments
        segments = [data[i:i + l] for i in range(0, len(data), l) if i + l <= len(data)]

        # Randomly split segments into train and test sets using the RNG
        train_segments, val_segments = train_test_split(segments, test_size=0.2, random_state=seed)

        # Sub-segment the data into n-length samples with overlap
        def create_subsegments(segments, n, every_ith=1):
            subsegments = []
            for segment in segments:
                subsegments.extend([segment[i:i + n] for i in range(0, len(segment) - n + 1, every_ith)])
            return subsegments

        train_samples = create_subsegments(train_segments, n)
        val_samples = create_subsegments(val_segments, n)

        # Calculate mean and std on train set
        train_data = np.vstack(train_samples)
        mean = np.mean(train_data, axis=0)
        std = np.std(train_data, axis=0)

        # Handle division by zero
        std[std == 0] = 1

        # Standardize all data
        train_data_standardized = [standardize(sample, mean, std) for sample in train_samples]
        val_data_standardized = [standardize(sample, mean, std) for sample in val_samples]

        # Separate input and target data
        train_inp = [seg[:, inp_cols_indices] for seg in train_data_standardized]
        train_trg = [seg[:, trg_cols_indices] for seg in train_data_standardized]
        val_inp = [seg[:, inp_cols_indices] for seg in val_data_standardized]
        val_trg = [seg[:, trg_cols_indices] for seg in val_data_standardized]

        if self.balance:
            train_inp, train_trg = self.balance_data(train_inp, train_trg)

        # Convert to torch tensors
        train_input_tensor = torch.from_numpy(np.stack(train_inp, axis=0)).float()
        train_target_tensor = torch.from_numpy(np.stack(train_trg, axis=0)).float()
        val_input_tensor = torch.from_numpy(np.stack(val_inp, axis=0)).float()
        val_target_tensor = torch.from_numpy(np.stack(val_trg, axis=0)).float()

        # Get the mean and std statistics for the targets
        trg_mean = mean[trg_cols_indices]
        trg_std = std[trg_cols_indices]

        # Convert to torch tensors
        trg_mean = torch.from_numpy(trg_mean).float().view(1, -1)
        trg_std = torch.from_numpy(trg_std).float().view(1, -1)

        return {
            'train_inp': train_input_tensor,
            'train_trg': train_target_tensor,
            'val_inp': val_input_tensor,
            'val_trg': val_target_tensor,
            'mean': mean,
            'std': std,
            'trg_mean': trg_mean,
            'trg_std': trg_std
        }

    def balance_data(
            self,
            train_inp: list[np.array],
            train_trg: list[np.array],
            sensitivity: float = 2.0,
            p_target: float = 0.7) -> tuple[list[np.array], list[np.array]]:

        seq_store = []

        for seq in train_trg:
            diff = np.diff(seq, axis=0, prepend=seq[:1])
            abs_cusum = np.abs(diff).cumsum(0).sum(1)
            seq_store.append(abs_cusum)

        seq_cat = np.concatenate(seq_store, axis=0)
        seq_med = np.median(seq_cat, axis=0)
        seq_mad = np.median(np.abs(seq_cat - seq_med), axis=0)

        threshold = seq_med + sensitivity * seq_mad

        above_threshold_idx = [i for i, seq in enumerate(seq_store) if seq[-1] > threshold]

        # Calculate how many instances should be added
        tot_num = len(seq_store)
        k_start = len(above_threshold_idx)
        p_init = k_start / tot_num

        if p_init >= p_target:
            return train_inp, train_trg

        k_end = p_target * (tot_num - k_start) / (1 - p_target)
        k_add = int(k_end - k_start)  # Number of instances to add to reach the target percentage

        new_train_inp = [arr for arr in train_inp]
        new_train_trg = [arr for arr in train_trg]

        for _ in range(k_add):
            idx_1 = np.random.choice(above_threshold_idx)
            inp_1 = train_inp[idx_1]
            trg_1 = train_trg[idx_1]

            inp_noise = self.noise_level * np.random.randn(inp_1.shape[0], 1)
            trg_noise = self.noise_level * np.random.randn(trg_1.shape[0], 1)

            inp_1 = inp_1 + inp_noise
            trg_1 = trg_1 + trg_noise

            new_train_inp.append(inp_1)
            new_train_trg.append(trg_1)

        print(f"Data was balanced by adding {k_add} minority instances.")

        return new_train_inp, new_train_trg

    def train_dataloader(self):
        dataset = MTSDataset(self.train_inp, self.train_trg)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.n_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent,
                          collate_fn=MTSDataset.collate_fn)

    def val_dataloader(self):
        dataset = MTSDataset(self.val_inp, self.val_trg)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.n_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent,
                          collate_fn=MTSDataset.collate_fn)

    def test_dataloader(self):
        dataset = MTSDataset(self.test_inp, self.test_trg)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.n_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent,
                          collate_fn=MTSDataset.collate_fn)
