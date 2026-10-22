"""
Evaluation and Benchmarking Script for Iraqi CCGT HRSG Creep PINN.
"""

from typing import Dict, Tuple
import os
import argparse
import time
import numpy as np
import torch

from physics.creep_fatigue import HRSGSuperheaterPhysics
from models.creep_pinn import HRSGCreepPINN


def evaluate_benchmark(
    checkpoint_path: str = "checkpoints/best_hrsg_pinn_model.pt",
    device_name: str = "cpu",
) -> Dict[str, float]:
    device = torch.device(device_name if torch.cuda.is_available() and device_name != "cpu" else "cpu")
    print(f"Loading checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found. Run train.py first.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    physics = HRSGSuperheaterPhysics()
    model = HRSGCreepPINN(
        in_dim=2, hidden_dim=64, num_layers=4, ri=physics.ri, ro=physics.ro, t_max=3600.0
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    r_test = np.linspace(physics.ri, physics.ro, 100).astype(np.float32)
    t_test = np.full_like(r_test, 1800.0)  # Mid-cycle (30 min)

    r_t = torch.tensor(r_test).unsqueeze(1).to(device)
    t_t = torch.tensor(t_test).unsqueeze(1).to(device)

    start_eval = time.time()
    with torch.no_grad():
        T_pred, sig_pred, D_pred = model(r_t, t_t)
    eval_time = (time.time() - start_eval) * 1e3

    # Analytical Lamé mechanical stress
    _, sig_th_mech, _ = physics.lame_mechanical_stresses(r_test)
    stress_rmse = float(np.sqrt(np.mean((sig_pred.cpu().numpy().flatten() - sig_th_mech) ** 2)))

    results = {
        "mean_temperature_k": float(torch.mean(T_pred).item()),
        "hoop_stress_rmse_mpa": stress_rmse,
        "mean_accumulated_damage": float(torch.mean(D_pred).item()),
        "eval_time_ms": float(eval_time),
    }

    print("\n" + "=" * 60)
    print("      IRAQI CCGT HRSG CREEP-FATIGUE PINN BENCHMARK      ")
    print("=" * 60)
    print(f" Mean Header Temperature:          {results['mean_temperature_k']:.1f} K ({results['mean_temperature_k'] - 273.15:.1f} °C)")
    print(f" Hoop Stress RMSE vs Lamé:         {results['hoop_stress_rmse_mpa']:.2f} MPa")
    print(f" Continuum Creep Damage Fraction:  {results['mean_accumulated_damage']:.4f}")
    print(f" Evaluation Inference Latency:     {results['eval_time_ms']:.2f} ms")
    print("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_hrsg_pinn_model.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    evaluate_benchmark(checkpoint_path=args.checkpoint, device_name=args.device)
