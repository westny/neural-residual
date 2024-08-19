import torch
import torch.nn as nn
import torch.distributions as tdist

from models.layers.masked_ff import MaskedNet


class Encoder(nn.Module):

    def __init__(self,
                 num_states: int,
                 num_signals: int,
                 used_signals: list[int],
                 net_config: dict,
                 variational: bool = False,
                 ) -> None:
        super().__init__()
        self.num_states = num_states
        self.num_signals = num_signals
        self.used_signals = used_signals
        self.variational = variational

        self.net = nn.Identity()

        if variational:
            self.net = MaskedNet(self.num_signals,
                                 self.num_signals,
                                 self.used_signals,
                                 [],
                                 net_config["net"],
                                 self.num_states * 2)

    def forward(self, signal: torch.Tensor, training: bool = True):
        """
        Parameters
        ----------
        signal: torch.Tensor [batch_size, num_signals]
        training: bool []

        Returns
        -------
        z0: torch.Tensor [batch_size, num_states]
        q: torch.distributions.Normal / None

        """

        q = None
        batch_size = signal.shape[0]

        if self.variational:
            if training:
                null = torch.zeros_like(signal)
                out = self.net((signal, null))
                qz0_mean, qz0_logvar = out[:, :self.num_states], out[:, self.num_states:]
                q = tdist.Normal(qz0_mean, torch.exp(qz0_logvar / 2.) + 1e-8)
                z0 = q.rsample()
            else:
                z0 = torch.randn(batch_size, self.num_states, device=signal.device)
        else:
            z0 = torch.zeros(batch_size, self.num_states, device=signal.device)

        return z0, q
