"""
Generate all required plots for the comparative analysis (Section 4.4.1).

This script loads training data from both agents and generates:
1. Reward Curve (both agents overlaid)
2. Loss Curves (both agents overlaid)
3. Per-agent detailed plots
4. Entropy plot (PPO only)

Run this after training both agents:
    python generate_plots.py
"""

import utils


def main():
    """Generate all required plots."""
    print("\n" + "=" * 70)
    print("Generating Required Plots for Comparative Analysis (Section 4.4.1)")
    print("=" * 70)

    success = utils.generate_comparative_plots()

    if not success:
        print("\n✗ Error: Could not generate plots")
        print("  Make sure you've trained both agents and they saved their results")


if __name__ == "__main__":
    main()
