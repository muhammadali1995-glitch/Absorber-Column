"""
Design-mode demonstration -- stress-tests the model's full physics.

Automatically calculates liquid flow for 30% NaOH utilization target.
This exercises pH gradients, speciation shifts, thermal effects, and
varying Ha/E along the column -- physics that are dormant at the default
high-L/G operating point.

NOTE: At 420 ppm CO2 DAC conditions, achieving high NaOH utilization
AND good column hydraulics simultaneously is physically challenging.
The warnings produced (poor wetting, low flooding) reflect real
engineering constraints, not code bugs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from absorber import AdiabaticAbsorber

if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  DESIGN MODE -- 30% NaOH UTILIZATION TARGET")
    print("=" * 62)
    design_case = AdiabaticAbsorber(
        design_mode=True,
        target_NaOH_utilization=0.30,
    )
    design_case.solve_column()
    design_case.print_summary()
    design_case.plot_profiles()