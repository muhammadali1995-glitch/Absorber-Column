"""
Basic validation tests for the absorber model.

Run with: python -m pytest tests/test_absorber.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from absorber import Absorber, AdiabaticAbsorber, PACKING_DB


class TestAbsorberCreation:
    """Test that Absorber objects can be created with various parameters."""

    def test_default_creation(self):
        ab = Absorber()
        assert ab.D_col > 0
        assert ab.A_col > 0
        assert ab.u_G > 0

    def test_fixed_diameter(self):
        ab = Absorber(D_col=0.30)
        assert ab.D_col == 0.30
        assert ab._auto_sized is False

    def test_packing_database(self):
        for name in PACKING_DB:
            ab = Absorber(packing_name=name, disc_z=51)
            assert ab.packing_name == name

    def test_design_mode(self):
        ab = Absorber(design_mode=True, target_NaOH_utilization=0.30, disc_z=51)
        assert ab.L_vol_flow < 40.0  # Should be much less than default


class TestPhysicalProperties:
    """Test physical property correlations."""

    def test_gas_density(self):
        ab = Absorber(disc_z=51)
        rho = ab.density_gas(293.15)
        assert 1.1 < rho < 1.3  # ~1.21 kg/m3 at 20C

    def test_henry_constant(self):
        ab = Absorber(disc_z=51)
        H = ab.henry_CO2_pure_water(298.15)
        assert 0.03 < H < 0.04  # ~0.034 mol/(L*atm)

    def test_reaction_rate(self):
        ab = Absorber(disc_z=51)
        k = ab.reaction_rate_constant(293.15)
        assert 5000 < k < 12000  # ~8051 L/(mol*s) at 20C

    def test_water_saturation_pressure(self):
        ab = Absorber(disc_z=51)
        p = ab.p_sat_water(373.15)
        assert 95000 < p < 110000  # ~101325 Pa at 100C


class TestSpeciation:
    """Test carbonate speciation solver."""

    def test_high_pH_fresh_NaOH(self):
        ab = Absorber(c_Na_tot=2.0, disc_z=51)
        c_OH, _, _, _, pH, _ = ab.solve_speciation(293.15, 1e-6)
        assert pH > 14.0

    def test_speciation_with_DIC(self):
        ab = Absorber(c_Na_tot=2.0, disc_z=51)
        c_OH, c_HCO3, c_CO3, c_CO2aq, pH, c_NaOH_free = \
            ab.solve_speciation(293.15, 0.5)
        assert c_CO3 > c_HCO3  # At high pH, carbonate dominates
        assert c_NaOH_free < 2.0
        assert pH > 12.0


class TestIsothermalSolve:
    """Test isothermal column solution."""

    def test_solve_converges(self):
        ab = Absorber(Z=5.0, disc_z=101)
        ab.solve_column()
        eta = ab.capture_efficiency()
        assert eta > 0
        assert eta <= 100

    def test_co2_balance_closes(self):
        ab = Absorber(Z=10.0, disc_z=151)
        ab.solve_column()
        n_gas = ab.G_inert * ab.A_col * (ab.Y_CO2_in - ab.Y_CO2[-1])
        u_L_out = ab._local_u_L(ab.C_DIC[0])
        n_liq = (u_L_out * ab.C_DIC[0] * 1000.0 * ab.A_col
                 - ab.u_L * ab.C_DIC_top * 1000.0 * ab.A_col)
        err = abs(n_gas - n_liq) / max(abs(n_gas), 1e-20)
        assert err < 1e-3

    def test_co2_decreases_upward(self):
        ab = Absorber(Z=5.0, disc_z=101)
        ab.solve_column()
        assert ab.y_CO2[-1] < ab.y_CO2[0]  # CO2 removed going up


class TestAdiabaticSolve:
    """Test adiabatic column solution."""

    def test_adiabatic_converges(self):
        ab = AdiabaticAbsorber(Z=5.0, disc_z=101)
        ab.solve_column()
        eta = ab.capture_efficiency()
        assert eta > 0

    def test_temperature_profiles_exist(self):
        ab = AdiabaticAbsorber(Z=5.0, disc_z=101)
        ab.solve_column()
        assert len(ab.T_G) == 101
        assert len(ab.T_L) == 101
        assert np.all(ab.T_G > 250)
        assert np.all(ab.T_L > 250)


class TestHydraulics:
    """Test hydraulic calculations."""

    def test_flooding_in_range(self):
        ab = Absorber(disc_z=51)
        ab.solve_column()
        ab.calculate_hydraulics()
        assert 0 < ab.f_flood < 100

    def test_pressure_drop_positive(self):
        ab = Absorber(disc_z=51)
        ab.solve_column()
        ab.calculate_hydraulics()
        assert ab.dP_total > 0

    def test_auto_size_respects_D_min(self):
        ab = Absorber(D_min=0.20, L_vol_flow=0.1, disc_z=51)
        assert ab.D_col >= 0.20

    def test_auto_size_respects_u_G_max(self):
        ab = Absorber(u_G_max=1.0, disc_z=51)
        assert ab.u_G <= 1.0 + 0.01  # small tolerance for rounding


if __name__ == "__main__":
    pytest.main([__file__, "-v"])