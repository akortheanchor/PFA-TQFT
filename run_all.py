#!/usr/bin/env python3
"""
run_all.py
==========
Single entry point: run all experiments and generate all figures + GIFs.

Usage
-----
    python run_all.py                   # everything
    python run_all.py --audit           # scientific audit only
    python run_all.py --experiments     # experiments only (no figures)
    python run_all.py --figures         # figures only
    python run_all.py --quick           # fast mode (fewer phases/shots)

Output
------
    figures/    — 6 publication-quality PNG figures (300 DPI)
    gifs/       — 5 animated GIFs
    results/    — CSV data from experiments
"""

import argparse
import sys
import os
import time

# Ensure package is importable from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def section(title: str) -> None:
    w = 64
    print("\n" + "═" * w)
    print(f"  {title}")
    print("═" * w)


def run_audit() -> bool:
    """Scientific audit: verify all theorems and corrected values."""
    section("SCIENTIFIC AUDIT")
    from pfa_tqft import (
        exact_qft_matrix, phase_input_state, tqft_circuit,
        total_variation_distance, tvd_upper_bound,
        optimal_d_star, gate_count_tqft, gate_count_full_qft,
        tfim_ground_energy, noise_crossover_threshold,
        PLATFORMS,
    )
    import numpy as np
    import math

    issues = []
    passes = []

    # 1. Theorem 1
    print("\n[1] Theorem 1: TV(P_φ, P_φ^d) ≤ π(m-d)/2^d")
    rng = np.random.default_rng(42)
    for m in [4, 5]:
        F = exact_qft_matrix(m)
        for d in range(1, m + 1):
            phis  = rng.uniform(0, 1, 300)
            bound = tvd_upper_bound(m, d)
            tvds  = []
            for phi in phis:
                inp  = phase_input_state(m, phi)
                Pf   = np.abs(F @ inp) ** 2
                Pt   = np.abs(tqft_circuit(inp, m, d)) ** 2
                tvds.append(total_variation_distance(Pf, Pt))
            ok = all(v <= bound + 1e-10 for v in tvds)
            tag = "PASS" if ok else "FAIL"
            if ok:
                passes.append(f"Theorem 1: m={m},d={d}")
            else:
                issues.append(f"Theorem 1 VIOLATED: m={m},d={d}")
    print(f"    {len([p for p in passes if 'Theorem' in p])} / "
          f"{sum(1 for i in [4,5] for _ in range(i))} checks passed")

    # 2. PFA Criterion
    print("\n[2] PFA Criterion: d* = floor(log2(2π/ε₂q))")
    expected = [("IBM Eagle r3", 3e-3, 11), ("IBM Heron r2", 5e-4, 13),
                ("IonQ Aria",    3e-4, 14), ("IQM Garnet",   2e-3, 11)]
    for name, eps, exp_d in expected:
        got = optimal_d_star(eps)
        ok  = (got == exp_d)
        print(f"    {name:<16}: ε={eps:.0e}, d*={got} ({'✓' if ok else f'✗ expected {exp_d}'})")
        if ok: passes.append(f"d* correct: {name}")
        else:  issues.append(f"d* wrong: {name}")

    # 3. Gate count
    print("\n[3] Gate count G(m,d) by exact enumeration")
    checks = [(30,11,245),(30,13,282),(30,14,299),(5,3,7)]
    for m, d, exp in checks:
        got = gate_count_tqft(m, d)
        ok  = (got == exp)
        print(f"    G({m},{d}) = {got}  {'✓' if ok else f'✗ expected {exp}'}")
        if ok: passes.append(f"G({m},{d})={exp}")
        else:  issues.append(f"G({m},{d}) wrong: got {got}, expected {exp}")

    # 4. TFIM E0
    print("\n[4] TFIM ground energy (n=4, J=1, h=0.5, open BC)")
    E0, _, _ = tfim_ground_energy(4)
    ok = abs(E0 - (-3.427034)) < 1e-4
    print(f"    E0 = {E0:.6f} J  {'✓' if ok else '✗ expected -3.427034'}")
    if ok: passes.append("TFIM E0")
    else:  issues.append(f"TFIM E0 wrong: {E0:.4f}")

    # 5. Cross-over threshold
    print("\n[5] Noise-truncation cross-over threshold (analytical model)")
    eps_cross = noise_crossover_threshold(16, optimal_d_star(3e-3))
    ok = 1.5e-3 < eps_cross < 3.5e-3
    print(f"    ε× = {eps_cross:.3e}  "
          f"{'✓ (in range [1.5e-3, 3.5e-3])' if ok else '✗'}")
    if ok: passes.append("cross-over threshold")
    else:  issues.append(f"Cross-over threshold out of range: {eps_cross:.3e}")

    print("\n" + "═" * 64)
    print(f"  AUDIT: {len(passes)} PASS / {len(issues)} FAIL")
    if issues:
        for i in issues: print(f"  ✗ {i}")
    else:
        print("  ALL CHECKS PASSED ✓")
    print("═" * 64)
    return len(issues) == 0


def run_all_experiments(quick: bool = False) -> None:
    """Run the four paper experiments."""
    section("EXPERIMENTS")
    os.makedirs("results", exist_ok=True)
    n_phases = 200 if quick else 500
    n_shots  = 500 if quick else 1000

    from experiments.exp1_tvd_validation import run_tvd_experiment, save_csv
    from experiments.exp2_platform_rules import run_platform_analysis, print_design_table
    from experiments.exp3_tfim_qpe import run_rmse_table, run_noise_sweep
    from experiments.exp4_fidelity_cliff import run_cliff_analysis

    print("\n── Experiment 1: TVD bound validation ──")
    results1 = run_tvd_experiment([4, 5], n_phases=n_phases, verbose=True)
    save_csv(results1, "results/exp1_tvd_validation.csv")

    print("\n── Experiment 2: Platform design rules ──")
    results2 = run_platform_analysis(m=30)
    print_design_table(results2)

    print("\n── Experiment 3: TFIM RMSE analysis ──")
    rows = run_rmse_table(m=16, eps_2q=1e-3)
    print(f"{'Method':<30} {'Gates':>6} {'RMSE (×10⁻³)':>14} {'ΔRMSE':>8}")
    for r in rows:
        print(f"{r['method']:<30} {r['gates']:>6d} "
              f"{r['rmse']:>13.3f} {r['delta']:>+8.1f}%")

    print("\n── Experiment 4: Fidelity cliff ──")
    run_cliff_analysis([4, 5], n_shots=n_shots, verbose=True)


def generate_figures(quick: bool = False) -> None:
    """Generate all publication figures and animated GIFs."""
    section("FIGURES AND GIFs")
    # Import the figures module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "pfa_tqft_figures",
            os.path.join(os.path.dirname(__file__), "src", "pfa_tqft_figures.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print("\nGenerating static figures (300 DPI)...")
        mod.figure_circuit()
        mod.figure_tvd_bound()
        mod.figure_gate_count()
        mod.figure_fidelity_cliff()
        mod.figure_tfim_rmse()
        mod.figure_platform_analysis()
        if not quick:
            print("\nGenerating animated GIFs...")
            mod.gif_tvd_convergence()
            mod.gif_qpe_distribution()
            mod.gif_fidelity_cliff()
            mod.gif_noise_synergy()
            mod.gif_circuit_evolution()
        print("\n✓ All figures saved to figures/ and gifs/")
    except Exception as e:
        print(f"  [figures] {e}")
        print("  Run: python src/pfa_tqft_figures.py  for figures separately")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PFA-TQFT: run all experiments, figures, and audit"
    )
    parser.add_argument("--audit",       action="store_true",
                        help="Scientific audit only")
    parser.add_argument("--experiments", action="store_true",
                        help="Experiments only (no figures)")
    parser.add_argument("--figures",     action="store_true",
                        help="Figures only")
    parser.add_argument("--quick",       action="store_true",
                        help="Fast mode: fewer phases/shots")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║  PFA-TQFT — Phase-Fidelity-Aware Truncated QFT               ║
║  Akoramurthy S, Surendiran B — NIT Puducherry                 ║
║  Quantum journal (submitted 2025)                             ║
╚══════════════════════════════════════════════════════════════╝
""")

    t0 = time.time()
    ok = True

    if args.audit:
        ok = run_audit()
    elif args.experiments:
        run_all_experiments(args.quick)
    elif args.figures:
        generate_figures(args.quick)
    else:
        # Default: audit + experiments + figures
        ok = run_audit()
        run_all_experiments(args.quick)
        generate_figures(args.quick)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
