"""
Training Pipeline for Iraqi CCGT HRSG Superheater Creep-Fatigue PINN.
"""

from typing import Dict, List, Tuple
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from physics.creep_fatigue import HRSGSuperheaterPhysics
from data.dataset import HRSGCollocationDataset
from models.creep_pinn import HRSGCreepPINN
from utils.visualization import (
    plot_header_radial_temperature_stress,
    plot_creep_fatigue_damage,
    plot_remaining_life_vs_ramping,
    plot_pinn_convergence,
)


def run_training(
    epochs: int = 35,
    lr: float = 3e-3,
    n_interior: int = 2500,
    n_boundary: int = 400,
    device_name: str = "cpu",
    output_dir: str = "checkpoints",
    figures_dir: str = "figures",
) -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, output_dir)
    figures_dir = os.path.join(base_dir, figures_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    device = torch.device(device_name if torch.cuda.is_available() and device_name != "cpu" else "cpu")
    print(f"=== Iraqi CCGT HRSG Header Creep-Fatigue PINN Training ===")
    print(f"Device: {device} | Epochs: {epochs} | Interior Points: {n_interior}")

    physics = HRSGSuperheaterPhysics()
    dataset = HRSGCollocationDataset(
        n_interior=n_interior, n_boundary=n_boundary, ri=physics.ri, ro=physics.ro, t_max=3600.0
    )

    model = HRSGCreepPINN(
        in_dim=2, hidden_dim=64, num_layers=4, ri=physics.ri, ro=physics.ro, t_max=3600.0
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    total_losses = []
    pde_losses = []
    bc_losses = []

    best_loss = float("inf")
    start_time = time.time()

    diffusivity = physics.thermal_diffusivity  # ~6.23e-6 m^2/s

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        batches = dataset.get_batches()
        r_int, t_int = batches["interior"][0].to(device), batches["interior"][1].to(device)

        # 1. Interior Thermal Conduction & Equilibrium Residual
        T, sig_th, D = model(r_int, t_int)

        # dT/dt
        dT_dt = torch.autograd.grad(T, t_int, torch.ones_like(T), create_graph=True)[0]
        # dT/dr
        dT_dr = torch.autograd.grad(T, r_int, torch.ones_like(T), create_graph=True)[0]
        # (1/r) * d/dr (r * dT/dr) = d^2T/dr^2 + (1/r) * dT/dr
        dT_drr = torch.autograd.grad(dT_dr, r_int, torch.ones_like(dT_dr), create_graph=True)[0]
        radial_laplacian = dT_drr + (1.0 / r_int) * dT_dr

        # PDE residual: dT/dt - alpha * Laplacian(T) = 0
        res_thermal = dT_dt - diffusivity * radial_laplacian

        # Damage evolution residual: dD/dt - (sig_th / A_D)^r
        dD_dt = torch.autograd.grad(D, t_int, torch.ones_like(D), create_graph=True)[0]
        sig_eff = torch.clamp(sig_th / physics.A_D, min=0.0, max=2.5)
        res_damage = dD_dt - 1e-4 * (sig_eff ** 3.0)

        loss_pde = torch.mean(res_thermal ** 2) + 1e2 * torch.mean(res_damage ** 2)

        # 2. Boundary Conditions
        # (a) Initial condition (t = 0): T(r, 0) = 300 K (cold ambient)
        r_0, t_0 = batches["initial"][0].to(device), batches["initial"][1].to(device)
        T_0, _, D_0 = model(r_0, t_0)
        loss_init = torch.mean((T_0 - 300.0) ** 2) + torch.mean(D_0 ** 2)

        # (b) Inner surface (r = ri): T(ri, t) heated to 565 C = 838.15 K
        r_in, t_in = batches["inner"][0].to(device), batches["inner"][1].to(device)
        T_in, _, _ = model(r_in, t_in)
        loss_inner = torch.mean((T_in - 838.15) ** 2)

        loss_bc = 0.1 * loss_init + 0.1 * loss_inner

        loss_total = loss_pde + loss_bc
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        total_losses.append(loss_total.item())
        pde_losses.append(loss_pde.item())
        bc_losses.append(loss_bc.item())

        if loss_total.item() < best_loss:
            best_loss = loss_total.item()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_loss": best_loss,
                },
                os.path.join(output_dir, "best_hrsg_pinn_model.pt"),
            )

        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Loss: {loss_total.item():.4f} (PDE: {loss_pde.item():.4f}, BC: {loss_bc.item():.4f})"
            )

    elapsed = time.time() - start_time
    print(f"PINN Training completed in {elapsed:.2f} s. Best Loss: {best_loss:.4f}")

    # Generate Publication Figures
    print("\nGenerating publication figures...")

    # Figure 1: Header Radial Stresses and Temperature
    r_eval = np.linspace(physics.ri, physics.ro, 100)
    sig_r_m, sig_th_m, _ = physics.lame_mechanical_stresses(r_eval)
    _, sig_th_t = physics.thermal_transient_stress(r_eval, dT_wall=65.0)
    sig_tot = sig_th_m + sig_th_t
    T_c = 565.0 - 65.0 * (np.log(r_eval / physics.ri) / np.log(physics.ro / physics.ri))

    plot_header_radial_temperature_stress(
        r_mm=r_eval * 1e3,
        T_c=T_c,
        sig_mech=sig_th_m,
        sig_total=sig_tot,
        save_path=os.path.join(figures_dir, "header_radial_temperature_stress.png"),
    )

    # Figure 2: Creep-Fatigue Damage Accumulation
    sim_data = physics.simulate_damage_accumulation(num_cycles=6000, ramp_rate_c_min=5.0)
    plot_creep_fatigue_damage(
        sim_data=sim_data,
        save_path=os.path.join(figures_dir, "creep_fatigue_damage_accumulation.png"),
    )

    # Figure 3: Remaining Life vs Ramping Rate
    plot_remaining_life_vs_ramping(
        save_path=os.path.join(figures_dir, "remaining_life_vs_ramping_rate.png")
    )

    # Figure 4: PINN Convergence
    plot_pinn_convergence(
        epochs=list(range(1, epochs + 1)),
        total_loss=total_losses,
        pde_loss=pde_losses,
        bc_loss=bc_losses,
        save_path=os.path.join(figures_dir, "pinn_creep_stress_convergence.png"),
    )
    print("All 4 publication figures generated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_training(epochs=args.epochs, lr=args.lr, device_name=args.device)
