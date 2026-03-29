"""
Parameter sensitivity analysis for the DAC NaOH absorber.

Sweeps key parameters (L_vol_flow, Z, G_vol, a_p, c_NaOH_in)
around the baseline design, with auto-sized column diameter for
each case.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from absorber import Absorber


def run_sensitivity_analysis():
    baseline = dict(
        Z=15.0, D_col=None, a_p=250.0, epsilon=0.97, d_p=0.016,
        C_BG=0.0338, C_BL=1.334,
        sigma_c=0.075, packing_type="structured",
        max_liq_load=40.0,
        y_CO2_in=420e-6, T_G_in=293.15, G_vol=35.0, RH_in=0.60,
        c_Na_tot=2.0, c_NaOH_in=2.0, C_DIC_in=0.0, T_L_in=293.15,
        L_vol_flow=40.0, kL_model="Higbie", disc_z=151,
    )

    sweeps = {
        "L_vol_flow": [0.5, 1, 2, 5, 10, 20, 40],
        "Z":          [3, 5, 8, 10, 15, 20],
        "G_vol":      [15, 25, 35, 50, 70],
        "a_p":        [125, 250, 350, 500],
        "c_NaOH_in":  [0.5, 1.0, 2.0, 3.0, 5.0],
    }

    print(f"\n{'=' * 110}")
    print(f"  PARAMETER SENSITIVITY ANALYSIS (V6.7.4, auto-sized D_col)")
    print(f"{'=' * 110}")

    for param_name, values in sweeps.items():
        print(f"\n  --- Sweep: {param_name} ---")
        print(
            f"  {'Value':>10s}  {'D[cm]':>5s}  {'eta[%]':>6s}  "
            f"{'kG_e3':>6s}  {'kL_e5':>6s}  {'Ha':>6s}  {'E':>6s}  "
            f"{'HTU[m]':>6s}  {'NTU':>6s}  {'NaOH%':>6s}  {'pH_bot':>5s}"
        )
        print(f"  {'-' * 90}")

        for val in values:
            params = baseline.copy()
            params[param_name] = val
            if param_name == "c_NaOH_in":
                params["c_Na_tot"] = val
            try:
                ab = Absorber(**params)
                ab.solve_column()
                eta = ab.capture_efficiency()
                ut = ab.NaOH_utilization()
                ab.calculate_hydraulics()
                print(
                    f"  {str(val):>10s}  "
                    f"{ab.D_col * 100:>5.0f}  "
                    f"{eta:>6.2f}  "
                    f"{np.mean(ab.kG_profile) * 1e3:>6.2f}  "
                    f"{np.mean(ab.kL_profile) * 1e5:>6.2f}  "
                    f"{ab.Ha.max():>6.1f}  "
                    f"{ab.E.max():>6.1f}  "
                    f"{ab.HTU:>6.3f}  "
                    f"{ab.NTU:>6.3f}  "
                    f"{ut * 100:>6.2f}  "
                    f"{ab.pH[0]:>5.2f}"
                )
            except Exception as e:
                print(f"  {str(val):>10s}  FAILED: {e}")


if __name__ == "__main__":
    run_sensitivity_analysis()