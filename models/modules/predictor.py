import torch
import torch.nn as nn

from models.layers.masked_ff import MaskedNet


class Predictor(nn.Module):
    def __init__(self,
                 num_states: int,
                 num_signals: int,
                 predictor_config: dict,
                 net_config: dict,
                 ) -> None:
        super().__init__()
        self.num_states = num_states
        self.num_signals = num_signals
        self.num_predictors = len(predictor_config)

        self.h = nn.ModuleList(self.create_net(predictor_config, net_config["net"]))

    def create_net(self,
                   predictors: dict,
                   net_config: dict
                   ) -> list[MaskedNet]:
        h_list = []

        for _, value in predictors.items():
            inputs = value["inputs"]
            states = value["states"]

            if "use_latent" in value.keys():
                if value["use_latent"]:
                    states = list(range(self.num_states))

            h = MaskedNet(self.num_states,
                          self.num_signals,
                          states,
                          inputs,
                          net_config)

            h_list.append(h)
        return h_list

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        y = []
        for n in range(self.num_predictors):
            y_n = self.h[n](x)
            y.append(y_n)
        y = torch.cat(y, dim=-1)
        return y
