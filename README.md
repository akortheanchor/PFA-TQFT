# PFA-TQFT: Phase-Fidelity-Aware Truncated Quantum Fourier Transform

[![CI](https://github.com/akortheanchor/PFA-TQFT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Journal: Quantum](https://img.shields.io/badge/Journal-Quantum-purple)](https://quantum-journal.org)

Reproducibility repository for:

> **Phase-Fidelity-Aware Truncated Quantum Fourier Transform  
> for Scalable Phase Estimation on NISQ Hardware**  
> Akoramurthy B, Surendiran B  
> Department of CSE, NIT Puducherry, India  
> *Quantum* (submitted 2026)

---
<img width="2720" height="1383" alt="PFA_TQFT_Illustration" src="https://github.com/user-attachments/assets/54efca73-7685-4334-976b-e09b341d16ac" />


## Overview

The quantum Fourier transform (QFT) underpins quantum phase estimation (QPE), yet its
O(m²) gate count exceeds the coherence budget of NISQ hardware beyond m ≈ 20 qubits.

This work introduces **PFA-TQFT**: a hardware-calibrated approximate QFT that

1. **Proves** TV(P_φ, P_φ^d) ≤ π(m−d)/2^d (Theorem 1, exact simulation validated)
2. **Derives** the optimal truncation depth d\* = ⌊log₂(2π/ε₂q)⌋ from device calibration
3. **Achieves** 31–44% gate-count reduction on IBM Eagle/Heron, IonQ Aria, IQM Garnet
4. **Reveals** a noise-truncation synergy: PFA-TQFT outperforms full QFT above ε× ≈ 2.3×10⁻³

All code uses **pure NumPy/SciPy** — no external quantum SDK required.

---

## Quick Start

```bash
git clone https://github.com/nit-ece-qamp/pfa-tqft.git
cd pfa-tqft

pip install -e .                   # install package
python run_all.py --audit          # verify all theorems (< 2 min)
python run_all.py --experiments    # run all paper experiments
python run_all.py --figures        # generate all figures + GIFs
python run_all.py --quick          # fast demo (reduced shots)
```

Run the test suite:

```bash
pytest tests/ -v
```

---

## Repository Structure

```
pfa-tqft/
│
├── pfa_tqft/                  # Core library (importable package)
│   ├── __init__.py            # Public API
│   ├── core.py                # Simulation + formulas (corrected values)
│   └── platforms.py           # Validated platform data
│
├── experiments/               # Paper experiments (reproduce all results)
│   ├── exp1_tvd_validation.py # Theorem 1 exact simulation (Fig 2a)
│   ├── exp2_platform_rules.py # Platform design rules (Table 1, Fig 5)
│   ├── exp3_tfim_qpe.py       # TFIM RMSE analysis (Table 2, Fig 6)
│   └── exp4_fidelity_cliff.py # Fidelity cliff (Fig 4)
│
├── tests/                     # Pytest suite (all theorems + formulas)
│   └── test_theorems.py
│
├── src/
│   └── pfa_tqft_figures.py    # Figure + GIF generation (300 DPI)
│
├── figures/                   # Generated PNG figures (300 DPI)
├── gifs/                      # Generated animated GIFs
│
├── run_all.py                 # Single entry point
├── setup.py
├── requirements.txt
└── LICENSE
```

---

## Core API

```python
from pfa_tqft import (
    exact_qft_matrix,          # 2^m × 2^m QFT unitary
    tqft_circuit,              # PFA-TQFT_d statevector simulation
    tvd_upper_bound,           # Theorem 1: π(m-d)/2^d
    optimal_d_star,            # d* = floor(log2(2π/ε₂q))
    gate_count_tqft,           # G(m,d) exact formula
    tfim_ground_energy,        # TFIM exact diagonalisation
    rmse_model,                # Analytical RMSE model (Appendix E)
    noise_crossover_threshold, # ε× (analytical, model-derived)
    PLATFORMS,                 # Validated platform dictionary
)
```

### Theorem 1 — quick check

```python
import numpy as np
from pfa_tqft import exact_qft_matrix, phase_input_state, tqft_circuit
from pfa_tqft import total_variation_distance, tvd_upper_bound

m, d, phi = 5, 3, 0.375
F    = exact_qft_matrix(m)
inp  = phase_input_state(m, phi)
tvd  = total_variation_distance(np.abs(F @ inp)**2,
                                np.abs(tqft_circuit(inp, m, d))**2)
bound = tvd_upper_bound(m, d)
print(f"TVD = {tvd:.4f}  ≤  bound = {bound:.4f}  →  {'PASS' if tvd <= bound else 'FAIL'}")
# TVD = 0.1196  ≤  bound = 0.7854  →  PASS
```

### PFA criterion — get d\* for your device

```python
from pfa_tqft import optimal_d_star, gate_count_tqft, gate_count_full_qft

eps_2q = 3e-3          # your device's two-qubit gate error rate
m      = 30

d_star     = optimal_d_star(eps_2q)       # = 11
gates_pfa  = gate_count_tqft(m, d_star)  # = 245
gates_full = gate_count_full_qft(m)       # = 435
reduction  = 100 * (1 - gates_pfa / gates_full)

print(f"d* = {d_star}  |  gates: {gates_pfa}/{gates_full}  |  reduction: {reduction:.1f}%")
# d* = 11  |  gates: 245/435  |  reduction: 43.7%
```

---

## Validated Platform Table (m = 30)

| Platform      | ε₂q      | d\* | Gates   | Reduction |
|--------------|----------|-----|---------|-----------|
| IBM Eagle r3 | 3×10⁻³  | **11** | 245/435 | 43.7%  |
| IBM Heron r2 | 5×10⁻⁴  | **13** | 282/435 | 35.2%  |
| IonQ Aria    | 3×10⁻⁴  | **14** | 299/435 | 31.3%  |
| IQM Garnet   | 2×10⁻³  | **11** | 245/435 | 43.7%  |

> **Note:** d\* values use the correct formula d\* = ⌊log₂(2π/ε₂q)⌋.
> Earlier preprint versions incorrectly used ⌊log₂(1/ε₂q)⌋, giving values
> one unit too low (10, 12, 13, 10). All values here are verified.

---

## TFIM Ground Energy

```python
from pfa_tqft import tfim_ground_energy

E0, evals, evecs = tfim_ground_energy(n=4, J=1.0, h=0.5)
print(f"E0 = {E0:.6f} J")
# E0 = -3.427034 J   (correct; preprint stated -5.226 J — incorrect)
```

---

## Noise-Truncation Synergy

PFA-TQFT outperforms full QFT for ε₂q ≳ ε× ≈ 2.3×10⁻³ (IBM Eagle r3, m=16, d\*=11).
This threshold is derived from the **analytical RMSE model** (Appendix E) —
direct hardware validation is listed as future work.

```python
from pfa_tqft import noise_crossover_threshold, optimal_d_star

eps_cross = noise_crossover_threshold(m=16, d=optimal_d_star(3e-3))
print(f"ε× = {eps_cross:.3e}")   # ε× = 2.3e-03
```

---

## Scientific Audit

```
python run_all.py --audit
```

Expected output:
```
[1] Theorem 1: TV(P_φ, P_φ^d) ≤ π(m-d)/2^d
    18/18 checks passed

[2] PFA Criterion: d* = floor(log2(2π/ε₂q))
    IBM Eagle r3: d*=11  ✓
    IBM Heron r2: d*=13  ✓
    IonQ Aria:    d*=14  ✓
    IQM Garnet:   d*=11  ✓

[3] Gate count G(m,d) by exact enumeration
    G(30,11)=245  ✓   G(30,13)=282  ✓
    G(30,14)=299  ✓   G(5,3)=7      ✓

[4] TFIM ground energy (n=4, J=1, h=0.5, open BC)
    E0 = -3.427034 J  ✓

[5] Noise-truncation cross-over threshold
    ε× = 2.3e-03  ✓ (in range [1.5e-3, 3.5e-3])

ALL CHECKS PASSED ✓
```

---

## Dependencies

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
pillow>=9.0
```

No quantum SDK (Qiskit, Cirq, etc.) required.
All simulation is pure linear algebra.

---

## Citation

```bibtex
@article{akoramurthy2025pfatqft,
  title   = {Phase-Fidelity-Aware Truncated Quantum Fourier Transform
             for Scalable Phase Estimation on NISQ Hardware},
  author  = {Akoramurthy, B. and Surendiran, B.},
  journal = {Quantum},
  year    = {2025},
  url     = {https://quantum-journal.org},
  note    = {Code: https://github.com/nit-ece-qamp/pfa-tqft}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).

Copyright © 2025 Akoramurthy B, Surendiran B, NIT Puducherry.
