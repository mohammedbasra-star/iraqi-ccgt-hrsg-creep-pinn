"""
Spatial and Temporal Collocation Point Generator for HRSG Creep PINN.
"""

from typing import Dict, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class HRSGCollocationDataset(Dataset):
    """
    Collocation points for thick cylinder P91 superheater header wall.
    Radial coordinate r in [ri, ro] = [0.12, 0.16] m
    Time coordinate t in [0, 3600] seconds (1-hour start-up cycle)
    """

    def __init__(
        self,
        n_interior: int = 3000,
        n_boundary: int = 500,
        ri: float = 0.120,
        ro: float = 0.160,
        t_max: float = 3600.0,
    ) -> None:
        self.ri = ri
        self.ro = ro
        self.t_max = t_max
        self.n_int = n_interior
        self.n_bnd = n_boundary

        self.resample()

    def resample(self) -> None:
        # Interior collocation points
        r_int = np.random.uniform(self.ri, self.ro, self.n_int).astype(np.float32)
        t_int = np.random.uniform(0.0, self.t_max, self.n_int).astype(np.float32)
        self.r_int = torch.tensor(r_int, requires_grad=True).unsqueeze(1)
        self.t_int = torch.tensor(t_int, requires_grad=True).unsqueeze(1)

        # Initial conditions (t = 0)
        r_init = np.random.uniform(self.ri, self.ro, self.n_bnd).astype(np.float32)
        t_init = np.zeros_like(r_init)
        self.r_init = torch.tensor(r_init, requires_grad=True).unsqueeze(1)
        self.t_init = torch.tensor(t_init, requires_grad=True).unsqueeze(1)

        # Inner surface (r = ri)
        t_inner = np.random.uniform(0.0, self.t_max, self.n_bnd).astype(np.float32)
        r_inner = np.full_like(t_inner, self.ri)
        self.r_inner = torch.tensor(r_inner, requires_grad=True).unsqueeze(1)
        self.t_inner = torch.tensor(t_inner, requires_grad=True).unsqueeze(1)

        # Outer surface (r = ro)
        t_outer = np.random.uniform(0.0, self.t_max, self.n_bnd).astype(np.float32)
        r_outer = np.full_like(t_outer, self.ro)
        self.r_outer = torch.tensor(r_outer, requires_grad=True).unsqueeze(1)
        self.t_outer = torch.tensor(t_outer, requires_grad=True).unsqueeze(1)

    def get_batches(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        return {
            "interior": (self.r_int, self.t_int),
            "initial": (self.r_init, self.t_init),
            "inner": (self.r_inner, self.t_inner),
            "outer": (self.r_outer, self.t_outer),
        }
