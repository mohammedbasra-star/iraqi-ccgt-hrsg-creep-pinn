"""
Multi-Physics Physics-Informed Neural Network (PINN) for Creep-Fatigue Damage.

Approximates coupled fields:
    (r, t) -> [ Temperature T(r, t), Hoop Stress sigma_theta(r, t), Creep Damage D(r, t) ]
"""

from typing import Tuple
import torch
import torch.nn as nn


class HRSGCreepPINN(nn.Module):
    """
    PINN with smooth activations (tanh) for second-order autograd derivatives.
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 4,
        ri: float = 0.120,
        ro: float = 0.160,
        t_max: float = 3600.0,
    ) -> None:
        super().__init__()
        self.ri = ri
        self.ro = ro
        self.t_max = t_max

        self.in_proj = nn.Linear(in_dim, hidden_dim)

        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Output heads:
        # Head 1: Temperature T (normalized around 300 to 900 K)
        # Head 2: Hoop stress sigma_theta (MPa)
        # Head 3: Damage fraction D in [0, 1]
        self.head_temp = nn.Linear(hidden_dim, 1)
        self.head_stress = nn.Linear(hidden_dim, 1)
        self.head_damage = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, r: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        r: (N, 1) radial coordinate
        t: (N, 1) time coordinate
        returns (T, sigma_theta, D)
        """
        # Coordinate normalization
        r_norm = 2.0 * (r - self.ri) / (self.ro - self.ri) - 1.0
        t_norm = 2.0 * (t / self.t_max) - 1.0

        coords = torch.cat([r_norm, t_norm], dim=-1)

        h = torch.tanh(self.in_proj(coords))
        for layer in self.hidden_layers:
            h = h + torch.tanh(layer(h))

        # Temperature in range ~ [300, 850] K
        T = 300.0 + 550.0 * torch.sigmoid(self.head_temp(h))

        # Hoop stress in MPa
        sig_th = 50.0 + 200.0 * torch.tanh(self.head_stress(h))

        # Damage in [0, 1]
        D = torch.sigmoid(self.head_damage(h))

        return T, sig_th, D
