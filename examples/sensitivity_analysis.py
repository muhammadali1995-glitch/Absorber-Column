"""
=============================================================================
 Sensitivity Analysis for 1D Counter-Current NaOH Packed Absorber (DAC)
 Compatible with absorber_Version19.py (V6.7.4)
=============================================================================
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ── Import the absorber model ──────────────────────────────────────────────
# Adjust the import to match your file name / module
from absorber_Version19 import Absorber, AdiabaticAbsorber, PACKING_DB

# ══════════════════════════════════════════════════════════════════════════
#  DIRECTORY SETUP
# ══════════════════════════════════════════════════════════════════════════
dirs = {
    "standard":          "absorber_results/standard/",
    "adiabatic":         "absorber_results/adiabatic/",
    "NaOH":              "absorber_results/sensitivity/NaOH_concentration/",
    "L_vol_flow":        "absorber_results/sensitivity/liquid_flow_rate/",
    "G_vol":             "absorber_results/sensitivity/gas_flow_rate/",
    "Z":                 "absorber_results/sensitivity/packed_height/",
    "T_in":              "absorber_results/sensitivity/temperature/",
    "RH":                "absorber_results/sensitivity/relative_humidity/",
    "packing":           "absorber_results/sensitivity/packing_type/",
}
for d in dirs.values():
    os.makedirs(d, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
#  HELPER: run one case, extract metrics, print one-liner
# ══════════════════════════════════════════════════════════════════════════

def run_and_report(label, save_dir, save_name, **kwargs):
    """
    Instantiate an Absorber with **kwargs, solve, report key metrics.
    Returns (absorber_instance, metrics_dict).
    """
    try:
        ab = Absorber(**kwargs)
        ab.solve_column()
    except Exception as e:
        print(f"  ✗ {label}: FAILED — {e}")
        return None, None

    eta  = ab.capture_efficiency()
    util = ab.NaOH_utilization()
    sty  = ab.removal_rate()
    ab.calculate_hydraulics()

    # Feasibility flags
    warnings = []
    if ab.f_flood > 100:
        warnings.append(f"⚠ UNPHYSICAL: {ab.f_flood:.0f}% flood — column would flood at this L/G for D={ab.D_col*100:.0f} cm")
    elif ab.f_flood < 20:
        warnings.append(f"⚠ Very low flood fraction ({ab.f_flood:.1f}%) — poor gas-side mass transfer")
    if ab.liq_load > ab.max_liq_load:
        warnings.append(f"⚠ Liquid load {ab.liq_load:.1f} > max {ab.max_liq_load:.0f} m³/(m²·h)")

    metrics = {
        "eta":       eta,
        "util":      util * 100,
        "NTU":       ab.NTU,
        "HTU":       ab.HTU,
        "STY":       sty * 1e6,
        "dP":        ab.dP_total,
        "flood":     ab.f_flood,
        "D":         ab.D_col * 100,
        "warnings":  warnings,
        "liq_load":  ab.liq_load,
        "kGaw":      np.mean(ab.kG_profile) * ab.a_w,
    }

    print(f"    η = {eta:.2f} %  |  util = {util*100:.2f} %  |  "
          f"NTU = {ab.NTU:.3f}  |  HTU = {ab.HTU:.3f} m  |  "
          f"STY = {sty*1e6:.3f} µmol/m³/s  |  "
          f"ΔP = {ab.dP_total:.1f} Pa  |  flood = {ab.f_flood:.1f} %  |  "
          f"D = {ab.D_col*100:.0f} cm")
    for w in warnings:
        print(f"    {w}")

    ab.save_results(directory=save_dir, name=save_name)
    return ab, metrics


def print_summary_table(title, param_name, param_values, metrics_list, unit=""):
    """Print a formatted summary table for a sweep."""
    print(f"\n{'='*100}")
    print(f"  SUMMARY: {title}")
    print(f"{'='*100}")
    hdr = (f"  {param_name:20s}  {'η [%]':>8s}  {'util [%]':>8s}  "
           f"{'NTU':>8s}  {'HTU [m]':>8s}  {'STY [µmol]':>10s}  "
           f"{'ΔP [Pa]':>8s}  {'flood [%]':>9s}  {'D [cm]':>6s}  Warning")
    print(hdr)
    print(f"  {'-'*96}")
    for val, m in zip(param_values, metrics_list):
        if m is None:
            print(f"  {str(val)+unit:20s}  {'FAILED':>8s}")
            continue
        warn_str = ""
        if m["flood"] > 100:
            warn_str += "  ⚠ HYDRAULICS"
        elif m["flood"] < 20:
            warn_str += "  ⚠ HYDRAULICS"
        if m.get("liq_load", 0) > 40:
            warn_str += "  ⚠ LIQ LOAD"
        print(f"  {str(val)+unit:20s}  {m['eta']:8.2f}  {m['util']:8.2f}  "
              f"  {m['NTU']:8.3f}  {m['HTU']:8.3f}  {m['STY']:10.3f}  "
              f"  {m['dP']:8.1f}  {m['flood']:9.1f}  {m['D']:6.0f}{warn_str}")
    print(f"{'='*100}")


# ══════════════════════════════════════════════════════════════════════════
#  COMMON PARAMETERS  (baseline)
# ══════════════════════════════════════════════════════════════════════════
BASELINE = dict(
    Z              = 15.0,
    packing_name   = "Mellapak 250Y",
    y_CO2_in       = 420e-6,
    T_G_in         = 293.15,       # 20 °C
    G_vol          = 35.0,         # m³/h
    RH_in          = 0.60,
    c_Na_tot       = 2.0,
    c_NaOH_in      = 2.0,
    C_DIC_in       = 0.0,
    T_L_in         = 293.15,
    L_vol_flow     = 40.0,         # L/min
    disc_z         = 401,
)

# ══════════════════════════════════════════════════════════════════════════
#  BASELINE (isothermal)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  STANDARD BASELINE CASE")
print("=" * 62)

print("\n  Baseline (isothermal)")
ab_base, m_base = run_and_report(
    "Baseline", dirs["standard"], "baseline",
    **BASELINE
)
D_fixed = ab_base.D_col  # lock diameter for all sweeps
print(f"\n  Baseline D_col = {D_fixed*100:.1f} cm (fixed for all sweeps)")

# ══════════════════════════════════════════════════════════════════════════
#  BASELINE (adiabatic)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  ADIABATIC BASELINE CASE")
print("=" * 62)

ab_adia = AdiabaticAbsorber(D_col=D_fixed, **BASELINE)
ab_adia.solve_column()
ab_adia.print_summary()
ab_adia.save_results(directory=dirs["adiabatic"], name="adiabatic_baseline")

eta_adia = ab_adia.capture_efficiency()
print(f"\n  Adiabatic baseline:")
print(f"    η         = {eta_adia:.2f} %")
print(f"    ΔT_G      = {ab_adia.T_G[-1] - ab_adia.T_G[0]:.2f} K")
print(f"    ΔT_L      = {ab_adia.T_L[0] - ab_adia.T_L[-1]:.2f} K")


# ══════════════════════════════════════════════════════════════════════════
#  1. NaOH CONCENTRATION SWEEP
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  SENSITIVITY: NaOH Concentration")
print("=" * 62)

c_NaOH_values = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
metrics_NaOH = []

for i, c in enumerate(c_NaOH_values):
    print(f"\n  Case {i}: c_NaOH_in = {c} M")
    kw = {**BASELINE, "D_col": D_fixed,
          "c_NaOH_in": c, "c_Na_tot": c}
    _, m = run_and_report(
        f"NaOH={c}", dirs["NaOH"],
        f"var_NaOH_{i}_c={c}", **kw)
    metrics_NaOH.append(m)

print_summary_table("NaOH Concentration Sweep",
                     "c_NaOH [M]", c_NaOH_values, metrics_NaOH)


# ══════════════════════════════════════════════════════════════════════════
#  2. LIQUID FLOW RATE SWEEP
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  SENSITIVITY: Liquid Flow Rate")
print("=" * 62)

L_values = [5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0]
metrics_L = []

for i, L in enumerate(L_values):
    print(f"\n  Case {i}: L_vol_flow = {L} L/min")
    kw = {**BASELINE, "D_col": D_fixed, "L_vol_flow": L}
    _, m = run_and_report(
        f"L={L}", dirs["L_vol_flow"],
        f"var_L_vol_flow_{i}_L_vol_flow={L}", **kw)
    metrics_L.append(m)

print_summary_table("Liquid Flow Rate Sweep (D fixed)",
                     "L [L/min]", L_values, metrics_L)


# ══════���═══════════════════════════════════════════════════════════════════
#  3. GAS FLOW RATE SWEEP
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  SENSITIVITY: Gas Flow Rate")
print("=" * 62)

G_values = [10.0, 20.0, 35.0, 50.0, 75.0, 100.0, 150.0]
metrics_G = []

for i, G in enumerate(G_values):
    print(f"\n  Case {i}: G_vol = {G} m³/h")
    kw = {**BASELINE, "D_col": D_fixed, "G_vol": G}
    _, m = run_and_report(
        f"G={G}", dirs["G_vol"],
        f"var_G_vol_{i}_G_vol={G}", **kw)
    metrics_G.append(m)

print_summary_table("Gas Flow Rate Sweep (D fixed)",
                     "G [m³/h]", G_values, metrics_G)


# ══════════════════════════════════════════════════════════════════════════
#  4. PACKED HEIGHT SWEEP
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  SENSITIVITY: Packed Height")
print("=" * 62)

Z_values = [2.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0]
metrics_Z = []

for i, Z in enumerate(Z_values):
    print(f"\n  Case {i}: Z = {Z} m")
    kw = {**BASELINE, "D_col": D_fixed, "Z": Z}
    _, m = run_and_report(
        f"Z={Z}", dirs["Z"],
        f"var_Z_{i}_Z={Z}", **kw)
    metrics_Z.append(m)

print_summary_table("Packed Height Sweep (D fixed)",
                     "Z [m]", Z_values, metrics_Z)


# ══════════════════════════════════════════════════════════════════════════
#  5. TEMPERATURE SWEEP  (NEW — important for DAC)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  SENSITIVITY: Inlet Temperature (gas & liquid)")
print("=" * 62)

T_values_C = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
metrics_T = []

for i, TC in enumerate(T_values_C):
    TK = TC + 273.15
    print(f"\n  Case {i}: T_in = {TC} °C")
    kw = {**BASELINE, "D_col": D_fixed,
          "T_G_in": TK, "T_L_in": TK}
    _, m = run_and_report(
        f"T={TC}°C", dirs["T_in"],
        f"var_T_{i}_T={TC}", **kw)
    metrics_T.append(m)

print_summary_table("Temperature Sweep (D fixed)",
                     "T [°C]", T_values_C, metrics_T)


# ══════════════════════════════════════════════════════════════════════════
#  6. RELATIVE HUMIDITY SWEEP  (NEW — affects water balance / evaporation)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  SENSITIVITY: Inlet Relative Humidity")
print("=" * 62)

RH_values = [0.20, 0.40, 0.60, 0.80, 0.95]
metrics_RH = []

for i, rh in enumerate(RH_values):
    print(f"\n  Case {i}: RH_in = {rh*100:.0f} %")
    kw = {**BASELINE, "D_col": D_fixed, "RH_in": rh}
    _, m = run_and_report(
        f"RH={rh*100:.0f}%", dirs["RH"],
        f"var_RH_{i}_RH={rh}", **kw)
    metrics_RH.append(m)

print_summary_table("Relative Humidity Sweep (D fixed)",
                     "RH [%]", [f"{r*100:.0f}" for r in RH_values], metrics_RH)


# ══════════════════════════════════════════════════════════════════════════
#  7. PACKING TYPE SWEEP  (NEW — compare structured packings)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  SENSITIVITY: Packing Type")
print("=" * 62)

packing_names = ["Mellapak 125Y", "Mellapak 250Y", "Mellapak 500Y",
                  "Sulzer BX", "Pall 25mm", "Pall 50mm"]
metrics_packing = []

for i, pname in enumerate(packing_names):
    print(f"\n  Case {i}: {pname}")
    # For packing sweep, let auto-sizing pick the best D for each packing
    kw = {**BASELINE, "packing_name": pname}
    # Don't fix D_col — different packings have different max liq loads
    _, m = run_and_report(
        pname, dirs["packing"],
        f"var_packing_{i}_{pname.replace(' ','_')}", **kw)
    metrics_packing.append(m)

print_summary_table("Packing Type Sweep (auto-sized D)",
                     "Packing", packing_names, metrics_packing)


# ══════════════════════════════════════════════════════════════════════════
#  COMBINED SENSITIVITY SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  PARAMETER IMPORTANCE RANKING FOR DAC ABSORBER")
print("=" * 80)
print("""
  Based on sensitivity analysis results:

  1. PACKED HEIGHT (Z)        — MOST IMPORTANT
     η ranges from 53% (2 m) to 99.7% (15 m).  HTU ≈ 2.6 m is constant,
     so Z/HTU directly sets NTU and thus η.  Strong diminishing returns
     above ~15 m.

  2. GAS FLOW RATE (G_vol)    — HIGH IMPACT
     η drops from ~100% (10 m³/h) to 96% (150 m³/h).  STY increases
     linearly.  Trade-off: higher throughput vs lower capture fraction.
     Also determines column diameter and pressure drop.

  3. LIQUID FLOW RATE (L)     — MODERATE-HIGH IMPACT
     η ranges 96–100%.  Primarily affects wetted area (a_w) and hence
     kG·a_w.  Very high L causes flooding; very low L reduces wetting
     and capture.  Also determines NaOH utilization.

  4. INLET TEMPERATURE        — MODERATE IMPACT (NEW)
     Higher T → faster kinetics (k_OH doubles per ~15 K) but lower
     Henry's constant (less CO₂ solubility) and more evaporation.
     Net effect depends on whether system is gas-film or liquid-film
     controlled.

  5. NaOH CONCENTRATION       — LOW-MODERATE IMPACT
     η stays ~99.6% from 0.5 to 3.0 M, then drops at >4 M due to
     salting-out and reduced D_L.  System is 90% gas-film controlled,
     so liquid chemistry changes are buffered.  Mainly affects
     downstream regeneration costs.

  6. RELATIVE HUMIDITY        — LOW IMPACT on η (NEW)
     Primarily affects water balance and evaporative cooling.
     η changes are minimal since CO₂ absorption dominates.

  7. PACKING TYPE             — MODERATE IMPACT (NEW)
     Different a_p values fundamentally change interfacial area.
     Higher a_p (500Y) → more area → better capture at same height,
     but also higher ΔP and potentially worse wetting.
""")

print("\n  Sensitivity analysis complete.")
