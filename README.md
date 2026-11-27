# Electronic Supporting File Repository

Supporting data and code for the article:

**Bridging Molecular Dynamics and Continuum Flow: A Slip Length Correlation for Water in a Nanoslit**

## Contents

- `data/`: slip-length database from 223 molecular dynamics cases.
- `code/`: correlation fitting, machine-learning benchmarks, and symbolic regression.
- `md_example/`: representative LAMMPS input, model, potential, and output files.
- `results/`: final correlation and model-comparison results.

## Requirements

Install the Python packages with:

```bash
python -m pip install -r requirements.txt
```

LAMMPS is required only to rerun the MD example. Operon symbolic regression
additionally requires `pyoperon`.

## Usage

Run from the repository root:

```bash
python code/correlation_fitting.py
python code/ml_benchmark.py
python code/symbolic_regression.py
```

Generated files are written to `results/`.

## Slip-length calculation

The MD mass flow rate was obtained by integrating the product of local density
and streamwise velocity across the channel. The slip length was determined by
comparison with the no-slip Navier-Stokes solution under the same conditions.
Complete equations and derivations are provided in the associated article.

## Reference

The Cu EAM potential in `md_example/Cu_u3.eam` follows:

Foiles, S. M.; Baskes, M. I.; Daw, M. S. Embedded-atom-method functions for the
fcc metals Cu, Ag, Au, Ni, Pd, Pt, and their alloys. *Physical Review B* **1986**,
*33* (12), 7983-7991. https://doi.org/10.1103/PhysRevB.33.7983
