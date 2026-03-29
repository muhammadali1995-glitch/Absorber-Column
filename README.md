# DAC NaOH Packed Absorber Model

A 1D steady-state counter-current packed column model for CO₂ absorption from ambient air into aqueous NaOH solution — Direct Air Capture (DAC).

**Version:** 6.7.4

## Overview

This model simulates the mass transfer, chemical reaction, and (optionally) heat transfer in a packed absorption column for removing CO₂ from air using NaOH solution. It is designed for Direct Air Capture (DAC) applications operating at atmospheric CO₂ concentrations (~420 ppm).

### Key Features

- **Isothermal and adiabatic** operation modes
- **Rigorous carbonate speciation** (CO₂(aq), HCO₃⁻, CO₃²⁻, OH⁻) via charge-balance solver
- **Three liquid-side mass transfer models**: Billet-Schultes, Higbie penetration, Onda (1968)
- **Enhancement factor** via DeCoursey (1974) approximation with finite E∞
- **Auto-sizing** of column diameter from gas flooding, liquid loading, and velocity constraints
- **Water evaporation** coupling (humidity balance)
- **Built-in packing database** (Mellapak 250Y/500Y/125Y, Sulzer BX, Pall rings)
- **Comprehensive balance verification** (CO₂, H₂O, energy)
- **16-panel axial profile plots**

### Physics Summary

| Module | Correlation / Source |
|---|---|
| Gas-side kG | Billet & Schultes (1999) |
| Liquid-side kL | Billet-Schultes / Higbie / Onda (selectable) |
| Wetted area | Onda et al. (1968) |
| Enhancement factor | DeCoursey (1974) |
| CO₂ + OH⁻ kinetics | Pohorecki & Moniuk (2001), k = 5.985×10¹³ exp(−55400/RT) |
| Henry's law | Sander (2015) with Sechenov ionic correction |
| Carbonate equilibria | Millero (1995) |
| Flooding velocity | Billet & Schultes (1999) with holdup correction |
| Gas diffusivity | Fuller correlation |
| Water vapour | Antoine equation with NaOH activity correction |

## Installation

```bash
git clone https://github.com/muhammadarslanaslam01-ui/absorber-column.git
cd absorber-column
pip install -r requirements.txt
```

## Quick Start

```bash
# Run the default case (isothermal + adiabatic)
python absorber.py
```

This runs a 15 m column with Mellapak 250Y-equivalent structured packing, 35 m³/h air at 420 ppm CO₂, and 40 L/min of 2 M NaOH at 20 °C. The column diameter is auto-sized to 70% of flooding velocity.

### Using as a Library

```python
from absorber import Absorber, AdiabaticAbsorber

# Custom isothermal case
ab = Absorber(
    Z=10.0,                    # packed height [m]
    G_vol=50.0,                # gas flow [m³/h]
    L_vol_flow=20.0,           # liquid flow [L/min]
    c_NaOH_in=1.0,             # NaOH concentration [mol/L]
    c_Na_tot=1.0,              # total Na⁺ [mol/L]
    packing_name="Mellapak 250Y",
)
ab.solve_column()
ab.print_summary()

# Adiabatic case with coupled energy balance
ad = AdiabaticAbsorber(
    Z=10.0,
    G_vol=50.0,
    L_vol_flow=20.0,
    c_NaOH_in=1.0,
    c_Na_tot=1.0,
)
ad.solve_column()
ad.print_summary()
ad.plot_profiles()
```

## Input Parameters

### Column Geometry

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `Z` | 15.0 | m | Packed height |
| `D_col` | None (auto) | m | Column internal diameter |
| `D_min` | 0.15 | m | Minimum column diameter (auto-sizer) |
| `u_G_max` | 2.0 | m/s | Maximum gas velocity (auto-sizer) |
| `flood_fraction` | 0.70 | — | Target fraction of flooding velocity |

### Packing

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `packing_name` | None | — | Look up from PACKING_DB (e.g., "Mellapak 250Y") |
| `a_p` | 250 | m²/m³ | Specific surface area |
| `epsilon` | 0.97 | — | Void fraction |
| `d_p` | 0.016 | m | Nominal packing size |
| `max_liq_load` | 40.0 | m³/(m²·h) | Maximum liquid loading |

### Gas Inlet (z = 0, bottom)

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `y_CO2_in` | 420×10⁻⁶ | — | CO₂ mole fraction |
| `T_G_in` | 293.15 | K | Temperature |
| `G_vol` | 35.0 | m³/h | Volumetric flow rate |
| `RH_in` | 0.60 | — | Relative humidity |

### Liquid Inlet (z = Z, top)

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `c_NaOH_in` | 2.0 | mol/L | NaOH concentration |
| `c_Na_tot` | 2.0 | mol/L | Total Na⁺ concentration |
| `C_DIC_in` | 0.0 | mol/L | Dissolved inorganic carbon |
| `T_L_in` | 293.15 | K | Temperature |
| `L_vol_flow` | 40.0 | L/min | Volumetric flow rate |

### Mass Transfer

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `kL_model` | "Higbie" | — | Options: "BS", "Higbie", "Onda", "min", "mean" |
| `kG_override` | None | mol/(m²·s·atm) | Override gas-side coefficient |
| `kL_override` | None | m/s | Override liquid-side coefficient |

### Design Mode

| Parameter | Default | Description |
|---|---|---|
| `design_mode` | False | Auto-calculate liquid flow for target utilization |
| `target_NaOH_utilization` | 0.30 | Target NaOH utilization fraction |

## Output

### Console Output

The model prints:
1. **Auto-sizing result** — diameter, flooding %, liquid loading
2. **BVP convergence** — bracket, residuals, solution status
3. **Summary table** — geometry, mass transfer, performance, hydraulics
4. **Outlet streams** — full inlet/outlet molar flowrates
5. **Balance verification** — CO₂, H₂O, and energy closure

### Key Performance Metrics

| Metric | Description |
|---|---|
| η | CO₂ capture efficiency [%] |
| NTU | Number of transfer units |
| HTU | Height of a transfer unit [m] |
| STY | Space-time yield [µmol/(m³·s)] |
| NaOH utilization | Fraction of NaOH consumed [%] |
| % flood | Operating fraction of flooding velocity |
| ΔP | Total pressure drop [Pa] |

## Examples

### Parameter Sensitivity Analysis

```bash
python examples/sensitivity_analysis.py
```

Sweeps `L_vol_flow`, `Z`, `G_vol`, `a_p`, and `c_NaOH_in` around the baseline design.

### Design Mode Demonstration

```bash
python examples/design_mode_demo.py
```

Runs an adiabatic case with auto-calculated liquid flow targeting 30% NaOH utilization. This stress-tests the model's speciation and thermal physics.

## Testing

```bash
python -m pytest tests/ -v
```

## Available Packings

| Name | a_p [m²/m³] | ε | Type | Max Liq Load [m³/(m²·h)] |
|---|---|---|---|---|
| Mellapak 250Y | 250 | 0.97 | Structured | 40 |
| Mellapak 500Y | 500 | 0.97 | Structured | 25 |
| Mellapak 125Y | 125 | 0.97 | Structured | 60 |
| Sulzer BX | 492 | 0.90 | Structured | 20 |
| Pall 25mm | 220 | 0.94 | Random | 80 |
| Pall 50mm | 112 | 0.95 | Random | 120 |

## References

1. Seithümmer, V., et al. (2025). "Modelling CO₂ absorption in NaOH solution." *Chem. Ing. Tech.* 97(5), 554–559.
2. Ghaffari, A., et al. (2023). "Direct air capture of CO₂ using aqueous NaOH." *Ind. Eng. Chem. Res.* 62(19), 7566–7579.
3. Knuutila, H., et al. (2010). "Kinetics of the reaction of CO₂ with aqueous NaOH." *Chem. Eng. Sci.* 65, 6077–6088.
4. Sander, R. (2015). "Compilation of Henry's law constants." *Atmos. Chem. Phys.* 15, 4399–4981.
5. Millero, F. J. (1995). "Thermodynamics of the carbon dioxide system in the oceans." *Geochim. Cosmochim. Acta* 59, 661–677.
6. Billet, R. & Schultes, M. (1999). "Prediction of mass transfer columns with dumped and arranged packings." *Trans. IChemE* 77A, 498–504.
7. Onda, K., Takeuchi, H. & Okumoto, Y. (1968). "Mass transfer coefficients between gas and liquid phases in packed columns." *J. Chem. Eng. Jpn.* 1(1), 56–62.
8. Pohorecki, R. & Moniuk, W. (2001). "Kinetics of reaction between CO₂ and hydroxyl ions." *Chem. Eng. Sci.* 43, 1677–1684.
9. DeCoursey, W. J. (1974). "Absorption with chemical reaction: development of a new relation for the Danckwerts model." *Chem. Eng. Sci.* 29, 1867–1872.

## License

This project is provided for academic and research purposes.