"""
Continuum Damage Mechanics & Creep-Fatigue Interaction for CCGT HRSG Superheaters.

Models P91/P92 high-temperature martensitic steel header tubes in combined cycle
power plants (Besmaya & Rumaila CCGT) subjected to severe daily start-up cycling:
- Radial transient thermal conduction
- Lamé & thermal hoop/radial stresses in thick cylinders
- Norton-Bailey creep strain & Kachanov-Rabotnov damage evolution:
    dD/dt = ( sigma_eq / (A_D * (1 - D)) )^r
- Coffin-Manson low-cycle fatigue (LCF) interaction
"""

from typing import Dict, Tuple
import numpy as np


class HRSGSuperheaterPhysics:
    """
    Thermo-mechanical stress and creep damage model for P91 steam headers.
    """

    def __init__(
        self,
        r_inner: float = 0.120,      # Inner radius (m) (240 mm ID)
        r_outer: float = 0.160,      # Outer radius (m) (320 mm OD, 40 mm wall)
        p_steam_mpa: float = 14.0,   # High-pressure steam (140 bar = 14.0 MPa)
        T_steam_c: float = 565.0,    # Main steam temperature (565 C)
        E_modulus_gpa: float = 165.0,# Elastic modulus at 565 C (GPa)
        nu_poisson: float = 0.30,    # Poisson's ratio
        alpha_th: float = 1.35e-5,   # Thermal expansion coeff (1/K)
        k_thermal: float = 28.5,     # Thermal conductivity (W/m*K)
        rho_density: float = 7750.0, # Density (kg/m^3)
        cp_specific: float = 590.0,  # Specific heat (J/kg*K)
    ) -> None:
        self.ri = r_inner
        self.ro = r_outer
        self.p_mpa = p_steam_mpa
        self.T_steam = T_steam_c
        self.E = E_modulus_gpa * 1e3  # In MPa
        self.nu = nu_poisson
        self.alpha = alpha_th
        self.k = k_thermal
        self.rho = rho_density
        self.cp = cp_specific
        self.thermal_diffusivity = self.k / (self.rho * self.cp)  # ~6.23e-6 m^2/s

        # P91 Creep parameters at 565 C (Kachanov-Rabotnov model)
        # dD/dt = (sigma / A_D)^r * (1 - D)^(-q)
        self.A_D = 185.0   # Stress coefficient (MPa)
        self.r_exp = 5.2   # Creep damage exponent
        self.n_norton = 6.4 # Norton creep exponent

    def lame_mechanical_stresses(self, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Lamé equations for internal pressure:
        sigma_r(r) = (p * ri^2) / (ro^2 - ri^2) * (1 - ro^2 / r^2)
        sigma_theta(r) = (p * ri^2) / (ro^2 - ri^2) * (1 + ro^2 / r^2)
        sigma_z(r) = (p * ri^2) / (ro^2 - ri^2) (closed ends)
        """
        r_clamped = np.clip(r, self.ri, self.ro)
        factor = (self.p_mpa * (self.ri ** 2)) / (self.ro ** 2 - self.ri ** 2)

        sigma_r = factor * (1.0 - (self.ro ** 2) / (r_clamped ** 2))
        sigma_theta = factor * (1.0 + (self.ro ** 2) / (r_clamped ** 2))
        sigma_z = np.full_like(r_clamped, factor)

        return sigma_r, sigma_theta, sigma_z

    def thermal_transient_stress(
        self, r: np.ndarray, dT_wall: float = 65.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quasi-steady thermal stress during rapid warm start-up:
        Inner surface heats up faster than outer surface (dT_wall = T_inner - T_outer).
        """
        r_clamped = np.clip(r, self.ri, self.ro)
        # Logarithmic steady-state temperature profile
        T_profile = self.T_steam - dT_wall * (np.log(r_clamped / self.ri) / np.log(self.ro / self.ri))

        # Mean temperature integral: (2 / (ro^2 - ri^2)) * integral (T * r) dr
        # Analytical approximation for thin/moderate cylinders
        T_mean = self.T_steam - 0.5 * dT_wall

        # Thermal hoop stress: sigma_th = (alpha * E / (1 - nu)) * (T_mean - T(r))
        sigma_th_theta = (self.alpha * self.E / (1.0 - self.nu)) * (T_mean - T_profile)
        # Thermal radial stress
        sigma_th_r = (self.alpha * self.E / (2.0 * (1.0 - self.nu))) * dT_wall * (
            -np.log(self.ro / r_clamped) + ((self.ri ** 2) / (self.ro ** 2 - self.ri ** 2)) * (1.0 - (self.ro ** 2) / (r_clamped ** 2)) * np.log(self.ro / self.ri)
        )

        return sigma_th_r, sigma_th_theta

    def von_mises_equivalent_stress(
        self, sig_r: np.ndarray, sig_th: np.ndarray, sig_z: np.ndarray
    ) -> np.ndarray:
        """
        sigma_eq = sqrt( 0.5 * [ (sig_r - sig_th)^2 + (sig_th - sig_z)^2 + (sig_z - sig_r)^2 ] )
        """
        return np.sqrt(
            0.5 * (
                (sig_r - sig_th) ** 2
                + (sig_th - sig_z) ** 2
                + (sig_z - sig_r) ** 2
            )
        )

    def simulate_damage_accumulation(
        self,
        num_cycles: int = 6000,           # 6000 daily cycles ~ 16 years of operation
        hold_hours_per_cycle: float = 14.0, # Peaking load hold time per day
        ramp_rate_c_min: float = 5.0,     # Steam temperature ramp rate (C/min)
    ) -> Dict[str, np.ndarray]:
        """
        Calculates creep-fatigue life consumption over operational cycles.
        """
        # Thermal gradient proportional to start-up ramping rate
        dT_start = min(12.0 * ramp_rate_c_min, 110.0)

        # Stresses at critical inner bore (r = ri)
        r_bore = np.array([self.ri])
        sig_r_mech, sig_th_mech, sig_z_mech = self.lame_mechanical_stresses(r_bore)
        _, sig_th_th = self.thermal_transient_stress(r_bore, dT_wall=dT_start)

        # Peak start-up stress
        sig_th_peak = sig_th_mech[0] + sig_th_th[0]
        sig_eq_peak = self.von_mises_equivalent_stress(
            sig_r_mech[0], sig_th_peak, sig_z_mech[0]
        )

        # Steady hold stress
        sig_eq_hold = self.von_mises_equivalent_stress(
            sig_r_mech[0], sig_th_mech[0], sig_z_mech[0]
        )

        # Fatigue cycles to failure (Coffin-Manson for P91 steel):
        # Delta_eps = sig_eq_peak / E -> N_f
        delta_eps = max((sig_eq_peak - 120.0) / self.E, 0.001)
        N_failure_fatigue = int(0.5 * (delta_eps / 0.42) ** (-1.85))
        N_failure_fatigue = max(N_failure_fatigue, 2500)

        # Rupture time under hold stress (Larson-Miller parameter / Norton creep)
        # t_rupture in hours
        t_rupture_hold = 1.0e5 * (self.A_D / max(sig_eq_hold, 10.0)) ** self.n_norton
        t_rupture_hold = max(t_rupture_hold, 20000.0)

        cycles = np.linspace(1, num_cycles, 200)
        d_fatigue = cycles / N_failure_fatigue
        total_hold_time = cycles * hold_hours_per_cycle
        d_creep = total_hold_time / t_rupture_hold

        # Bilinear creep-fatigue interaction envelope (ASME Section III NH / RCC-MRx)
        d_total = d_fatigue + d_creep

        return {
            "cycles": cycles,
            "d_fatigue": np.clip(d_fatigue, 0.0, 1.2),
            "d_creep": np.clip(d_creep, 0.0, 1.2),
            "d_total": np.clip(d_total, 0.0, 1.5),
            "sig_peak_mpa": sig_eq_peak,
            "sig_hold_mpa": sig_eq_hold,
            "n_failure_fatigue": N_failure_fatigue,
            "t_rupture_hours": t_rupture_hold,
        }
