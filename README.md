# GPmp-contrib

`gpmp-contrib` provides higher-level workflows, models, and sequential design
algorithms built on top of `gpmp`.

It is intended for practical Gaussian-process pipelines where you want to
compose:
- model containers and ready-to-use Matérn model classes,
- sequential strategies for optimization / excursion / set inversion,
- diagnostics and visualization helpers,
- specialized procedures such as relaxed GP (reGP).

## Highlights

- Model classes:
  - `Model_ConstantMean_Maternp_ML`
  - `Model_ConstantMean_Maternp_REML`
  - `Model_ConstantMean_Maternp_REMAP`
  - `Model_Noisy_ConstantMean_Maternp_REML`
- Sequential strategies:
  - grid-search based (`SequentialStrategyGridSearch`)
  - SMC-adaptive (`SequentialStrategySMC`)
  - BSS-style (`SequentialStrategyBSS`)
- Optimization / design modules:
  - expected improvement (`gpmpcontrib.optim.expectedimprovement`)
  - excursion set estimation (`gpmpcontrib.optim.excursionset`)
  - set inversion and Pareto utilities
- reGP utilities:
  - `gpmpcontrib.regp`

## Package Layout

- `gpmpcontrib/models/`: preconfigured Matérn model classes
- `gpmpcontrib/modelcontainer.py`: multi-output container and wrappers
- `gpmpcontrib/sequentialprediction.py`: prediction/update orchestration
- `gpmpcontrib/sequentialstrategy.py`: sequential decision strategies
- `gpmpcontrib/optim/`: EI, excursion, set inversion, Pareto tools
- `gpmpcontrib/regp/`: relaxed GP utilities
- `examples/`: runnable scripts

## Requirements

- Python `>=3.9`
- `gpmp >= 0.9.34`
- `numpy`, `scipy>=1.12.0`, `matplotlib`

## Installation

Clone the repository:

```bash
git clone https://github.com/gpmp-dev/gpmp-contrib.git
cd gpmp-contrib
```

Install in dev mode:

```bash
pip install -e .
```

## Quick Start

```python
import gpmpcontrib as gpc

# See examples/ for full workflows
problem = gpc.ComputerExperiment(
    1,
    [[-1.0], [1.0]],
    single_function=lambda x: x**2,
)
```

## Examples

The `examples/` directory includes:
- model construction and prediction (`example01` to `example05`)
- expected improvement optimization (`example10`, `example11`, `example12`)
- relaxed GP workflow (`example20`)
- excursion set estimation (`example30`, `example31`)
- set inversion workflows (`example40`, `example41`)

## Authors

See `AUTHORS.md`.

## How to Cite

If you use GPmp-contrib in your research, please cite it as follows:

```bibtex
@software{gpmpcontrib2026,
  author       = {Emmanuel Vazquez},
  title        = {GPmp-contrib},
  year         = {2026},
  url          = {https://github.com/gpmp-dev/gpmp-contrib},
  note         = {Version x.y},
}
```

*Please update the version number as appropriate.*

## Copyright

Copyright (C) 2022-2026 CentraleSupelec

## License

GPmp contrib is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GPmp contrib is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
License for more details.

You should have received a copy of the GNU General Public License
along with gpmp. If not, see http://www.gnu.org/licenses/.
