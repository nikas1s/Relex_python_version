# py-relex

Python implementation of the RELEX relativistic Coulomb-excitation model described by
C. A. Bertulani, based on and validated against the original RELEX Fortran implementation.
Earlier Python adaptation work by Emma Rice was consulted and is acknowledged.

## Prerequisites

- Python 3.9+

## Install (macOS/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Install (Windows PowerShell)

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Dependencies (`numpy`, `scipy`, `pandas`, `streamlit`) are installed automatically by `pip`.

Note: `-e` installs in editable (development) mode. For a standard install, use:

```bash
python -m pip install .
```

## Run

```bash
relex examples/relex_example.json --output-dir ./out
```

## Streamlit UI

```bash
streamlit run app.py
```

Outputs are written as:
- `relex.out` (Fortran-style text output)
- `relex2.out` (statistical tensors, if requested)
- `results.json` (machine-readable summary)

## Input JSON schema (minimal)

```json
{
  "system": {
    "ap": 25.01,
    "zp": 9.0,
    "at": 208.98,
    "zt": 83.0,
    "eca": 100.0,
    "iw": 0,
    "iout": 1
  },
  "integration": {
    "nb": 40,
    "accur": 0.001,
    "bmin": 10.0,
    "itot": 1
  },
  "options": {
    "iopw": 0,
    "iopnuc": 0
  },
  "grid": {
    "ngrid": 200
  },
  "states": [
    {"i": 1, "ex": 0.0, "spin": 2.5},
    {"i": 2, "ex": 3.3, "spin": 4.5}
  ],
  "matrix_elements": [
    {"i": 1, "j": 2, "mate1": 0.0, "mate2": 10.5, "matm1": 0.0}
  ]
}
```

### Optional: optical potential

If `iopw = 1`, add:

```json
"optical_potential": {
  "path": "examples/optw.in"
}
```

The file format matches the Fortran code:
```
NR
R  Real(U)  Imag(U)
...
```

### Optional: nuclear deformation

If `iopnuc = 1`, add entries for each excited state `j = 2..nst`:

```json
"nuclear_deformation": [
  {"j": 2, "delte0": 0.0, "delte1": 0.0, "delte2": 0.0}
]
```

## Notes

- The physics model follows the published RELEX formulation. Coupled-channel trajectories use
  SciPy's `solve_ivp` RK45 solver; cross sections use SciPy's natural `CubicSpline` interpolation
  and `quad` integration.
- For large `ngrid`, the folding step is expensive (O(ngrid^3)).

## Numerical-method replacement

This version no longer contains the earlier Numerical Recipes-style implementations named
`rkck`, `rkqs`, `odeint`, `spline`, `splint`, `trapzd`, and `qsimp`. They were replaced as follows:

- coupled-channel time evolution uses SciPy `solve_ivp(method="RK45")`;
- impact-parameter data uses SciPy `CubicSpline` with natural boundary conditions;
- cross-section integration uses SciPy `quad`.

The nuclear-physics model and published RELEX equations were not changed. Regression testing
against the documented original Fortran result remains included below.

## Tests

```bash
python -m unittest discover -v
```

The suite includes focused tests for the SciPy numerical adapters and the 42S-on-Au regression
against the documented original Fortran result.

## Provenance and attribution

- **Original RELEX:** C. A. Bertulani, “A Computer Program for Relativistic Multiple Coulomb
  and Nuclear Excitation,” *Computer Physics Communications* **116** (1999), 345–352,
  [doi:10.1016/S0010-4655(98)00141-6](https://doi.org/10.1016/S0010-4655(98)00141-6),
  [arXiv:nucl-ex/9803009](https://arxiv.org/abs/nucl-ex/9803009).
- **Earlier Python work:** an adaptation by Emma Rice was consulted and is acknowledged.
- **Current Python implementation:** implementation, packaging, interface, and maintenance by
  Stylianos Nikas.
- **Source distribution:** the original RELEX Fortran source is not redistributed in this
  repository.
- **Numerical software:** NumPy is used for array calculations. SciPy supplies the adaptive
  Runge–Kutta solver, cubic-spline interpolation, and quadrature used by the Python solver.

## Licensing status and disclaimer

This repository does not currently declare a project-wide software license while its provenance
and source history are being reviewed. No ownership of the original RELEX Fortran implementation
is claimed. The software is provided "as is," without warranty or responsibility for its use.

## Verification and Paper Comparison

This code implements the relativistic, straight-line coupled-channels Coulomb-excitation model
described for RELEX, with optional nuclear-excitation and optical-potential effects. It does
**not** implement the Coulomb-trajectory or `exp(-pi xi_a)` correction curves discussed in
Esbensen (arXiv:0808.0361v1); those are separate approximations in that paper.

### Direct Fortran vs Python check (42S on Au)

The Python implementation is validated against the original Fortran result using the 42S to 2+
quadrupole excitation on Au from Esbensen 2008 (Table I / Fig. 2 inputs):

- `Ex = 0.890 MeV`, `B(E2) = 397 e^2 fm^4` -> `M(E2) = sqrt(B(E2)) = 19.9249 e fm^2`
- `bmin = 14 fm`, `iopw = 0`, `iopnuc = 0`
- `nb = 30`, `ngrid = 200`

Result for `State 2` cross section:

- **Fortran**: `266.9 mb`
- **Python**: `266.901 mb`

This matches to better than `0.01 mb`.

### Repro (Fortran, external)

The original RELEX Fortran source is not included in this repository. If you have it
from the original distribution, you can reproduce the comparison by running the same
inputs and checking the `State 2` cross section in `relex.out`.

### Repro (Python)

```bash
.venv/bin/python - <<'PY'
import math
from relex.core import run_relex

inputs = {
    "system": {"ap": 42.0, "zp": 16.0, "at": 197.0, "zt": 79.0, "eca": 20.0, "iw": 0, "iout": 0},
    "integration": {"nb": 30, "accur": 0.001, "bmin": 14.0, "itot": 0},
    "options": {"iopw": 0, "iopnuc": 0},
    "grid": {"ngrid": 200},
    "states": [
        {"i": 1, "ex": 0.0, "spin": 0.0},
        {"i": 2, "ex": 0.890, "spin": 2.0},
    ],
    "matrix_elements": [
        {"i": 1, "j": 2, "mate1": 0.0, "mate2": math.sqrt(397.0), "matm1": 0.0}
    ],
}

res = run_relex(inputs)
print(res.cross_sections)
PY
```
