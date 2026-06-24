# GPmp-contrib

`gpmp-contrib` extends `gpmp` with computer-experiment objects, multi-output
model containers, Matérn model classes, sequential design procedures, set
estimation tools, plots, and relaxed Gaussian-process utilities.

Use `gpmp` directly for core GP models, covariance functions, numerical backend
operations, and low-level parameter selection. Use `gpmp-contrib` when a script
needs a `ComputerExperiment`, a `ModelContainer`, a sequential strategy, a test
problem, or reGP.

## Main components

- Model containers and Matérn classes:
  - `Model_ConstantMean_Maternp_ML`
  - `Model_ConstantMean_Maternp_REML`
  - `Model_ConstantMean_Maternp_REMAP`
  - `Model_ConstantMean_Maternp_REMAP_logsigma2`
  - `Model_ConstantMean_Maternp_REMAP_logsigma2_and_logrho_prior`
  - `Model_Noisy_ConstantMean_Maternp_REML`
- Prior access on REMAP classes with priors:
  - `get_prior(...)`
  - `set_prior(...)`
- Sequential strategies:
  - fixed candidate sets with `SequentialStrategyGridSearch`
  - SMC particle sets with `SequentialStrategySMC`
  - BSS-style particle sets with `SequentialStrategyBSS`
- Optimization and set-estimation modules:
  - expected improvement in `gpmpcontrib.optim.expectedimprovement`
  - excursion sets in `gpmpcontrib.optim.excursionset`
  - set inversion and Pareto utilities in `gpmpcontrib.optim`
- reGP utilities in `gpmpcontrib.regp`.
- Parameter posterior sampling through `ModelContainer.sample_parameters(...)`.

## Package layout

- `gpmpcontrib/models/`: Matérn model container classes.
- `gpmpcontrib/modelcontainer.py`: multi-output model container.
- `gpmpcontrib/sequentialprediction.py`: observation storage and prediction updates.
- `gpmpcontrib/sequentialstrategy.py`: sequential decision strategies.
- `gpmpcontrib/optim/`: EI, excursion-set, set-inversion, and Pareto tools.
- `gpmpcontrib/regp/`: relaxed Gaussian-process utilities.
- `examples/`: scripts using the public objects.
- `docs/`: Sphinx documentation.

## Requirements

- Python `>=3.9`
- `gpmp >= 0.9.36`
- `numpy`
- `scipy>=1.12.0`
- `matplotlib`

## Installation

Install the released package from PyPI:

```bash
pip install gpmp-contrib
```

This installs `gpmp` and the other runtime dependencies declared in
`pyproject.toml`.

For development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/gpmp-dev/gpmp-contrib.git
cd gpmp-contrib
pip install -e .
```

When testing against a local `gpmp` checkout, install `gpmp` first, then install
`gpmp-contrib` in editable mode.

## Minimal example

```python
import gpmpcontrib as gpc

problem = gpc.ComputerExperiment(
    1,
    [[-1.0], [1.0]],
    single_function=lambda x: x**2,
)
```

The full documentation starts with `docs/source/getting_started.rst` and then
continues through the user guide. The examples section documents model
construction, noisy observations, expected improvement, excursion sets, set
inversion, and reGP.

## Documentation

The documentation is available at
<https://gpmp-dev.github.io/gpmp-contrib/>.

To build it locally, install the documentation dependencies and build the HTML
pages:

```bash
pip install -r docs/requirements.txt
cd docs
sphinx-build -M html source _build -E
```

Generate the static example figures with:

```bash
cd docs
python make_example_results.py
```

## Authors

See `AUTHORS.md`.

## How to cite

If you use GPmp-contrib in your research, please cite it as follows:

```bibtex
@software{gpmpcontrib2026,
  author       = {Emmanuel Vazquez},
  title        = {GPmp-contrib},
  year         = {2026},
  url          = {https://github.com/gpmp-dev/gpmp-contrib},
  note         = {Version 0.9.36},
}
```

Update the version number when citing another release.

## Copyright

Copyright (C) 2022-2026 CentraleSupelec

## License

GPmp-contrib is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

GPmp-contrib is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
GPmp-contrib. If not, see http://www.gnu.org/licenses/.
