# PFA-TQFT: Phase-Fidelity-Aware Truncated Quantum Fourier Transform

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://python.org)
[![Journal: Quantum](https://img.shields.io/badge/Journal-Quantum-purple.svg)](https://quantum-journal.org)

Reproducibility code for the paper:

> **Phase-Fidelity-Aware Truncated Quantum Fourier Transform for Scalable Phase Estimation on NISQ Hardware**  
> Akoramurthy B, Surendiran B  
> Department of Computer Science & Engineering, NIT Puducherry  
> *Quantum* (submitted) · DOI: 10.22331/q-XXXX

---

## Overview

This repository provides exact quantum circuit simulation code that:

- **Validates Theorem 1** (TVD upper bound) against exact statevector simulation
- **Reproduces all 6 publication figures** at 300 DPI
- **Generates 5 animated GIFs** illustrating key results
- **Runs a full scientific audit** verifying every theorem and formula

All simulation uses **exact linear-algebra statevector methods** — no approximations, no external quantum SDK required.

---

## Quick Start

```bash
# Clone
git clone https://github.com/akortheanchor/PFA-TQFT.git
cd pfa-tqft

# Install dependencies
pip install -r requirements.txt

# Run scientific audit (verify all theorems)
python src/pfa_tqft_figures.py --audit

# Generate all figures + GIFs
python src/pfa_tqft_figures.py

# Figures only
python src/pfa_tqft_figures.py --figs-only

# GIFs only
python src/pfa_tqft_figures.py --gifs-only
```

---

## Repository Structure

```
pfa-tqft/
├── src/
│   └── pfa_tqft_figures.py    # All simulation + figure code (self-contained)
├── figures/                   # Generated PNG figures (300 DPI, white bg)
│   ├── fig1_circuit.png       # Quantum circuit: Full QFT vs PFA-TQFT
│   ├── fig2_tvd_bound.png     # Theorem 1 validation + hardware d*
│   ├── fig3_gate_count.png    # O(m²) → O(m log m) gate scaling
│   ├── fig4_fidelity_cliff.png # Fidelity cliff (exact simulation)
│   ├── fig5_tfim_rmse.png     # TFIM RMSE comparison (5 methods)
│   └── fig6_platform_analysis.png # Platform-specific design rules
├── gifs/                      # Generated animated GIFs
│   ├── gif1_tvd_convergence.gif   # TVD → 0 as d increases
│   ├── gif2_qpe_distribution.gif  # QPE measurement distribution
│   ├── gif3_fidelity_cliff.gif    # Fidelity cliff for m=3,4,5
│   ├── gif4_noise_synergy.gif     # Noise-truncation cross-over
│   └── gif5_circuit_evolution.gif # Gate-by-gate amplitude evolution
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Core API

All physics is implemented in pure NumPy — no external quantum SDKs.

```python
from src.pfa_tqft_figures import (
    exact_qft_matrix,       # Exact 2^m × 2^m QFT unitary
    tqft_circuit,           # Truncated QFT at depth d (exact statevector)
    total_variation_distance,
    tvd_upper_bound,        # Theorem 1: pi*(m-d)/2^d
    optimal_d_star,         # PFA Criterion: floor(log2(2*pi/eps_2q))
    gate_count_tqft,        # Exact gate count formula
    gate_count_full_qft,
    tfim_ground_energy,     # Exact TFIM diagonalisation (n=4: E0=-3.427 J)
)
```

### Example: Validate Theorem 1 for a single case

```python
import numpy as np
from src.pfa_tqft_figures import (
    exact_qft_matrix, phase_input_state,
    tqft_circuit, total_variation_distance, tvd_upper_bound
)

m, d = 5, 3
F    = exact_qft_matrix(m)
phi  = 0.375

inp    = phase_input_state(m, phi)
P_full = abs(F @ inp) ** 2
P_trunc= abs(tqft_circuit(inp, m, d)) ** 2
tvd    = total_variation_distance(P_full, P_trunc)
bound  = tvd_upper_bound(m, d)

print(f"TVD = {tvd:.4f}  ≤  bound = {bound:.4f}  →  {'PASS' if tvd <= bound else 'FAIL'}")
# TVD = 0.1196  ≤  bound = 0.7854  →  PASS
```

### Example: Get hardware-optimal d* for your device

```python
from src.pfa_tqft_figures import optimal_d_star, gate_count_tqft, gate_count_full_qft

eps_2q = 3e-3          # Your device's two-qubit gate error rate
m      = 30            # Number of control qubits

d_star      = optimal_d_star(eps_2q)
gates_tqft  = gate_count_tqft(m, d_star)
gates_full  = gate_count_full_qft(m)
reduction   = 100 * (1 - gates_tqft / gates_full)

print(f"d* = {d_star}")
print(f"Gates: PFA-TQFT = {gates_tqft},  Full QFT = {gates_full}")
print(f"Reduction: {reduction:.1f}%")
# d* = 11
# Gates: PFA-TQFT = 245,  Full QFT = 435
# Reduction: 43.7%
```

---

## Validated Platform Parameters

All d* values are derived from the PFA criterion `d* = floor(log2(2π/ε_2q))` and verified against exact gate-angle thresholds.

| Platform      | ε_2q    | d* | Gates (m=30) | Reduction |
|--------------|---------|-----|--------------|-----------|
| IBM Eagle r3 | 3×10⁻³  | 11  | 245 / 435    | 43.7%     |
| IBM Heron r2 | 5×10⁻⁴  | 13  | 282 / 435    | 35.2%     |
| IonQ Aria    | 3×10⁻⁴  | 14  | 299 / 435    | 31.3%     |
| IQM Garnet   | 2×10⁻³  | 11  | 245 / 435    | 43.7%     |

---

## TFIM Exact Ground Energy

The 1D transverse-field Ising model with open boundary conditions:

```
H = -J Σᵢ Zᵢ Zᵢ₊₁ - h Σᵢ Xᵢ
```

For n=4 sites, J=1, h=0.5: **E₀ = −3.427034 J** (exact diagonalisation).

```python
from src.pfa_tqft_figures import tfim_ground_energy
E0, evals, evecs = tfim_ground_energy(n=4, J=1.0, h=0.5)
print(f"E0 = {E0:.6f} J")   # E0 = -3.427034 J
```

---

## Scientific Audit Output

```
[1] Theorem 1: TV(P_phi, P_phi^d) <= pi*(m-d)/2^d
    m=3..5, d=1..m: ALL PASS ✓

[2] PFA Criterion: d* = floor(log2(2*pi/eps_2q))
    IBM Eagle r3, Heron r2, IonQ Aria, IQM Garnet: ALL PASS ✓

[3] Gate-count formula: sum_j max(0, min(d-1, m-j-1))
    m in {5,10,20,30}: ALL PASS ✓

[4] TFIM exact ground energy (n=4, J=1, h=0.5, open BC)
    E0 = -3.427034 J: PASS ✓

[5] Corollary: d >= log2(pi*m/alpha)  (alpha=0.05)
    m=10,20,30: ALL PASS ✓

AUDIT RESULT: ALL CHECKS PASSED ✓
```

---

## Dependencies

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
pillow>=9.0
```

---

## Citation

```bibtex
@article{akoramurthy2025pfatqft,
  title   = {Phase-Fidelity-Aware Truncated Quantum Fourier Transform
             for Scalable Phase Estimation on NISQ Hardware},
  author  = {Akoramurthy, B. and Surendiran, B.},
  journal = {Quantum},
  year    = {2025},
  doi     = {10.22331/q-XXXX},
  url     = {https://quantum-journal.org}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).

Copyright © 2025 Akoramurthy S, Surendiran B, NIT Puducherry.
