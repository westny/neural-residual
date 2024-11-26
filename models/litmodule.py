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
import lightning.pytorch as pl
import torch.distributions as tdist

mse_loss = nn.MSELoss(reduction='none')
huber_loss = nn.SmoothL1Loss(reduction='none')


class LitModel(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 config: dict) -> None:
        super().__init__()
        self.model = model

        self.lr = config["lr"]
        self.decay = config["decay"]
        self.beta_max = config["beta_max"]
        self.beta_anneal = config["beta_anneal"]
        self.sample_time = config["sample_time"]
        self.epochs = config["epochs"]
        self.loss_fn = mse_loss if config["criterion"] == "mse" else huber_loss
        self.denormalize = config["denormalize"]
        self.store_prediction = config["store_test_pred"]
        self.catalog = config.get("catalog", None)

        self.save_hyperparameters(ignore=['model'])

    def forward(self, *args, **kwargs) -> None:
        pass

    @staticmethod
    def data_info(target: torch.Tensor) -> tuple[int, int]:
        return target.shape[0:2]

    def training_step(self, data, batch_idx) -> torch.Tensor:
        inputs, target = data

        int_steps, batch_size = self.data_info(target)
        pred, _, q = self.model(inputs, int_steps, self.sample_time, self.training)
        if q is not None:
            std_normal = tdist.Normal(torch.zeros_like(q.loc), torch.ones_like(q.scale))
            kl_div = tdist.kl_divergence(q, std_normal).sum(-1).mean()
            beta = min(1.0, self.current_epoch / self.beta_anneal) * self.beta_max
            kl_loss = beta * kl_div
        else:
            kl_loss = 0.

        recon_loss = self.loss_fn(pred, target).sum(-1).mean()

        loss = recon_loss + kl_loss

        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size, prog_bar=True)
        return loss

    def validation_step(self, data, batch_idx) -> None:
        inputs, target = data
        int_steps, batch_size = self.data_info(target)
        pred, *_ = self.model(inputs, int_steps, self.sample_time, self.training)
        loss = mse_loss(pred, target).mean()
        self.log("val_loss", loss, on_epoch=True, batch_size=batch_size, prog_bar=True)

    def test_step(self, data, batch_idx) -> None:
        inputs, target = data
        int_steps, batch_size = self.data_info(target)

        pred, states, _ = self.model(inputs, int_steps, self.sample_time, self.training)  # get the states

        if self.denormalize:
            # denormalize data for evaluation
            mean = self.trainer.datamodule.trg_mean.to(self.device)
            std = self.trainer.datamodule.trg_std.to(self.device)
            pred = pred * std + mean
            target = target * std + mean

        loss = mse_loss(pred, target).mean()

        if self.store_prediction:
            from time import strftime
            from pathlib import Path
            from pandas import DataFrame
            assert batch_size == 1, "Currently, only batch size 1 is supported for storing predictions. " \
                                    "This corresponds to setting test_sequence_len=-1."

            save_path = Path("predictions") / self.catalog if self.catalog is not None else Path("predictions")
            curr_time = strftime('%d-%m_%H:%M:%S')

            pred = pred.detach().cpu().numpy()
            signals = [signal + "_hat" for signal in self.trainer.datamodule.targets]
            test_df = self.trainer.datamodule.test_data

            # add predictions to dataframe
            for i, signal in enumerate(signals):
                test_df[signal] = pred[:, :, i].flatten()

            test_file = self.trainer.datamodule.test_file.split('.')[0]

            # save dataframe to csv
            save_name = f"test_predictions_{test_file}_{curr_time}.csv"
            save_pred_path = save_path / save_name
            test_df.to_csv(save_pred_path, index=False)

            # Store states in separate csv
            states = states.detach().cpu().numpy().reshape(-1, states.shape[-1])
            state_names = [f"state_{i}" for i in range(states.shape[-1])]
            state_df = DataFrame(states, columns=state_names)

            save_name = f"states_{test_file}_{curr_time}.csv"
            save_state_path = save_path / save_name

            state_df.to_csv(save_state_path, index=False)

        self.log("test_loss", loss, on_epoch=True, batch_size=batch_size, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.decay)
        return [optimizer], [scheduler]
