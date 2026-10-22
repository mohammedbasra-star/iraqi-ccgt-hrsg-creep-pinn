"""
Publication-Grade Visualizations for Iraqi CCGT HRSG Creep-Fatigue PINN.

Generates 300 DPI figures:
1. Radial Temperature and Hoop Stress Distributions through P91 Header Wall.
2. Creep and Fatigue Damage Accumulation Envelopes over Operational Cycles.
3. Remaining Operating Life vs Combustor / Turbine Ramping Rates.
4. PINN Multi-Objective Loss Convergence Diagnostics.
"""

from typing import Dict, List, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.8,
})


def plot_header_radial_temperature_stress(
    r_mm: np.ndarray,
    T_c: np.ndarray,
    sig_mech: np.ndarray,
    sig_total: np.ndarray,
    save_path: str = "figures/header_radial_temperature_stress.png",
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Temperature profile
    ax1.plot(r_mm, T_c, color="#d62728", linewidth=2.5, label="Temperature $T(r)$")
    ax1.axvline(120.0, color="gray", linestyle=":", label="Inner Bore ($r_i = 120\\mathrm{ mm}$)")
    ax1.axvline(160.0, color="black", linestyle=":", label="Outer Surface ($r_o = 160\\mathrm{ mm}$)")
    ax1.set_xlabel("Header Radius $r$ (mm)")
    ax1.set_ylabel(r"Wall Temperature ($^\circ\mathrm{C}$)")
    ax1.set_title("(a) Radial Thermal Gradient across 40 mm Wall")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Hoop Stress: Mechanical vs Total (Mechanical + Thermal)
    ax2.plot(r_mm, sig_mech, color="#1f77b4", linestyle="--", linewidth=2.0, label="Mechanical Lamé Stress $\\sigma_{\\theta,\\mathrm{mech}}$")
    ax2.plot(r_mm, sig_total, color="#d62728", linewidth=2.5, label="Total Stress (Start-up Peak)")
    ax2.axhline(310.0, color="darkred", linestyle="-.", label="P91 Yield Limit ($310\\mathrm{ MPa}$)")
    ax2.set_xlabel("Header Radius $r$ (mm)")
    ax2.set_ylabel("Hoop Stress $\\sigma_\\theta$ (MPa)")
    ax2.set_title("(b) Stress Concentration at Critical Inner Bore")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("HRSG Superheater Header Thermo-Mechanical State (P91 Steel)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_creep_fatigue_damage(
    sim_data: Dict[str, np.ndarray],
    save_path: str = "figures/creep_fatigue_damage_accumulation.png",
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cycles = sim_data["cycles"]
    d_fatigue = sim_data["d_fatigue"]
    d_creep = sim_data["d_creep"]
    d_total = sim_data["d_total"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Damage vs Cycles
    ax1.plot(cycles, d_creep, color="#d95f02", linewidth=2.2, label=r"Creep Damage $D_{\mathrm{creep}}$")
    ax1.plot(cycles, d_fatigue, color="#7570b3", linewidth=2.2, label=r"Fatigue Damage $D_{\mathrm{fatigue}}$")
    ax1.plot(cycles, d_total, color="#e7298a", linewidth=2.5, label=r"Total Damage $D_{\mathrm{total}}$")
    ax1.axhline(1.0, color="red", linestyle="--", label="Critical Rupture Limit ($D = 1.0$)")
    ax1.set_xlabel("Operational Daily Cycles (Starts)")
    ax1.set_ylabel("Accumulated Continuum Damage $D$")
    ax1.set_title("(a) Damage Evolution over 6,000 Peaking Cycles")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Creep-Fatigue Bilinear Interaction Diagram (ASME Section III NH)
    ax2.plot([0.0, 0.3, 1.0], [1.0, 0.3, 0.0], "r--", linewidth=2.0, label="Bilinear Allowable Boundary")
    ax2.plot(d_fatigue, d_creep, color="#1f77b4", linewidth=2.5, label="Operational Trajectory (Basra CCGT)")
    ax2.scatter(d_fatigue[-1], d_creep[-1], color="red", s=80, zorder=5, label="16-Year Operating Point")
    ax2.set_xlabel(r"Fatigue Damage Fraction $\sum (N / N_f)$")
    ax2.set_ylabel(r"Creep Damage Fraction $\sum (t / t_r)$")
    ax2.set_title("(b) ASME Creep-Fatigue Interaction Envelope")
    ax2.set_xlim(-0.05, 1.1)
    ax2.set_ylim(-0.05, 1.1)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Creep-Fatigue Interaction & End-of-Life Prediction", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_remaining_life_vs_ramping(
    save_path: str = "figures/remaining_life_vs_ramping_rate.png",
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ramp_rates = np.linspace(2.0, 15.0, 30)  # C/min steam ramping rate

    # Equivalent operating hours and years before crack initiation
    life_hours = 1.2e5 / (1.0 + 0.35 * (ramp_rates / 5.0) ** 1.9)
    life_years = life_hours / (365.25 * 14.0)

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    color1 = "#1f77b4"
    line1 = ax1.plot(ramp_rates, life_hours / 1e3, color=color1, linewidth=2.5, label="Operating Hours ($10^3$ h)")
    ax1.set_xlabel(r"Turbine Start-Up Ramping Rate ($^\circ\mathrm{C/min}$)")
    ax1.set_ylabel(r"Service Life to Crack Initiation ($10^3$ Hours)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "#d62728"
    line2 = ax2.plot(ramp_rates, life_years, color=color2, linestyle="--", linewidth=2.5, label="Calendar Years (14h/day Peaking)")
    ax2.set_ylabel("Estimated Operational Years", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.grid(False)

    # Shaded operational zones
    ax1.axvspan(2.0, 5.0, color="green", alpha=0.12, label="Recommended Cold/Warm Ramp")
    ax1.axvspan(9.0, 15.0, color="red", alpha=0.12, label="Severe Thermal Shock Peaking")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    plt.title("HRSG Header Life Degradation vs Gas Turbine Start-Up Ramping", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_pinn_convergence(
    epochs: List[int],
    total_loss: List[float],
    pde_loss: List[float],
    bc_loss: List[float],
    save_path: str = "figures/pinn_creep_stress_convergence.png",
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.semilogy(epochs, total_loss, color="#1f77b4", linewidth=2.2, label="Total Composite Loss")
    ax.semilogy(epochs, pde_loss, color="#d62728", linestyle="--", linewidth=2.0, label="Thermal Conduction Residual Loss")
    ax.semilogy(epochs, bc_loss, color="#2ca02c", linestyle="-.", linewidth=2.0, label="Boundary & Stress Equilibrium Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Residual")
    ax.set_title("HRSG Creep-Fatigue Multi-Physics PINN Optimization Convergence", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
