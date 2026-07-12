<div align="left">
  <img src="https://raw.githubusercontent.com/gpmp-dev/gpmp-contrib/main/docs/source/_static/logo.png" width="120" alt="GPmp-contrib logo"/>
</div>

# GPmp-contrib: computer experiments and sequential design

[![Python](https://img.shields.io/pypi/pyversions/gpmp-contrib.svg)](https://pypi.org/project/gpmp-contrib/)
[![PyPI](https://img.shields.io/pypi/v/gpmp-contrib.svg)](https://pypi.org/project/gpmp-contrib/)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE.txt)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://gpmp-dev.github.io/gpmp-contrib/)
[![Docs build](https://github.com/gpmp-dev/gpmp-contrib/actions/workflows/docs.yml/badge.svg)](https://github.com/gpmp-dev/gpmp-contrib/actions/workflows/docs.yml)

[Website and documentation](https://gpmp-dev.github.io/gpmp-contrib/)
| [Examples](https://gpmp-dev.github.io/gpmp-contrib/examples/index.html)
| [API reference](https://gpmp-dev.github.io/gpmp-contrib/api/index.html)
| [PyPI](https://pypi.org/project/gpmp-contrib/)
| [GPmp](https://github.com/gpmp-dev/gpmp)

GPmp-contrib provides computer-experiment objects, multi-output model
containers, Matérn model classes, sequential-design procedures, set-estimation
methods, plots, and relaxed Gaussian-process utilities. It builds these objects
on the models, kernels, parameter-selection functions, diagnostics, and
numerical backends provided by [GPmp](https://github.com/gpmp-dev/gpmp).

## When to use GPmp-contrib

Use `gpmp-contrib` when you need a `ComputerExperiment`, a `ModelContainer`,
provided Matérn model classes, sequential strategies, Bayesian optimization,
excursion-set estimation, set inversion, or reGP.

Use `gpmp` directly when you need the numerical GP model, covariance functions,
parameter-selection functions, diagnostics, posterior samplers, or plotting
helpers without computer-experiment or sequential-design objects.

## Core features

- **Computer experiments:** input boxes, normalized inputs, objectives,
  constraints, and multi-output evaluations.
- **Model containers:** one `gpmp.core.Model` per output, with parameter
  selection, prediction, diagnosis, and readable parameter objects.
- **Matérn model classes:** fixed half-integer regularity (`Maternp`) or jointly
  selected regularity `nu > 0` (`Matern`). The available classes cover ML,
  REML, REMAP, and noisy observations.
- **REMAP priors:** inspect and modify resolved prior parameters with
  `get_prior(...)` and `set_prior(...)`.
- **Sequential strategies:** fixed candidate grids, SMC particle sets, and
  BSS-style particle sets.
- **Optimization and set estimation:** expected improvement, excursion sets,
  set inversion, and Pareto utilities.
- **Relaxed Gaussian processes:** reGP procedures for threshold-oriented
  prediction problems.
- **Posterior parameter sampling:** access GPmp's MH, HMC, NUTS, and SMC
  samplers through `ModelContainer.sample_parameters(...)`.

## Numerical backends

GPmp-contrib uses the numerical backend selected by GPmp:

- **NumPy:** often fast for small-to-medium exact GP computations.
- **PyTorch:** provides automatic differentiation and is useful when gradient
  information is needed, especially in higher-dimensional parameter settings.

Set the backend before importing either package:

```bash
export GPMP_BACKEND=torch
export GPMP_DTYPE=float64
```

Model and kernel computations use `gpmp.num` backend objects. High-level
`ModelContainer` methods convert inputs and outputs where documented in the
[API reference](https://gpmp-dev.github.io/gpmp-contrib/api/index.html).

## Package split

```text
gpmp
  core GP model
  covariance functions
  parameter objects and selection
  diagnostics
  posterior samplers
  plotting helpers

gpmp-contrib
  computer experiments
  model containers and provided Matérn classes
  sequential strategies
  Bayesian optimization
  excursion-set estimation
  set inversion
  reGP utilities
```

## Install

```bash
pip install gpmp-contrib
python -c "print(__import__('gpmpcontrib').__version__)"
```

The verification command prints the installed GPmp-contrib version. The
installation also installs `gpmp` and the other runtime dependencies declared
in `pyproject.toml`.

For local development:

```bash
git clone https://github.com/gpmp-dev/gpmp-contrib.git
cd gpmp-contrib
pip install -e .
```

Use `pip install -e ".[docs]"` for documentation tools. When testing against a
local GPmp checkout, install that checkout first and then install gpmp-contrib
in editable mode.

GPmp-contrib requires Python 3.9 or later, GPmp 0.9.38 or later, NumPy, SciPy,
and Matplotlib. PyTorch is optional and is installed separately when automatic
differentiation is needed.

## Documentation

The documentation is available at
<https://gpmp-dev.github.io/gpmp-contrib/>. It includes a complete Hartmann4
example, model construction and state, parameter selection, diagnostics,
sequential design, excursion sets, set inversion, reGP, and the public API.

To build it locally:

```bash
pip install -e ".[docs]"
python docs/make_example_results.py
sphinx-build -M html docs/source docs/_build -E
```

## Public API

The intended public API is organized around:

- `gpmpcontrib.ComputerExperiment`
- `gpmpcontrib.modelcontainer`
- the model classes exported by `gpmpcontrib`
- `gpmpcontrib.SequentialPrediction`
- `gpmpcontrib.SequentialStrategyGridSearch`
- `gpmpcontrib.SequentialStrategySMC`
- `gpmpcontrib.SequentialStrategyBSS`
- `gpmpcontrib.samplingcriteria`
- `gpmpcontrib.optim`
- `gpmpcontrib.regp`
- `gpmpcontrib.test_problems`

## How to cite

If you use GPmp-contrib in research, please cite:

```bibtex
@software{gpmpcontrib2026,
  author       = {Emmanuel Vazquez},
  title        = {GPmp-contrib},
  year         = {2026},
  url          = {https://github.com/gpmp-dev/gpmp-contrib},
  note         = {Version 0.9.38},
}
```

Update the version number when citing another release.

## Minimal example

The basic sequence is: choose a computer experiment, build a model container,
select covariance parameters, predict, and inspect the result.

```python
import gpmp as gp
import gpmp.num as gnp
import gpmpcontrib as gpc

gnp.set_seed(1234)

problem = gpc.test_problems.hartmann4
xi = gp.misc.designs.ldrandunif(problem.input_dim, 40, problem.input_box)
zi = problem(xi)
xt = gp.misc.designs.ldrandunif(problem.input_dim, 300, problem.input_box)

model = gpc.Model_ConstantMean_Maternp_REML(
    "hartmann4",
    output_dim=problem.output_dim,
    mean_specification={"type": "constant"},
    covariance_specification={"p": 3},
)

model.select_params(xi, zi)
zpm, zpv = model.predict(xi, zi, xt)
model.run_diagnosis(xi, zi)
```

The final call prints parameter-selection, parameter, and observation
summaries. See the
[getting-started example](https://gpmp-dev.github.io/gpmp-contrib/getting_started.html)
for prediction checks, performance measures, stored model state, and the
corresponding figure.

## Authors

See [AUTHORS.md](AUTHORS.md) for details.

## License

GPmp-contrib is free software released under the GNU General Public License
v3.0 or later. See [LICENSE](LICENSE.txt) for details.
