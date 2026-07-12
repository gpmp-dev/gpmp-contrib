import importlib.metadata
from pathlib import Path

from .computerexperiment import ComputerExperiment
from . import test_problems
from .models import Model_ConstantMean_Matern_REML
from .models import Model_ConstantMean_Maternp_REML
from .models import Model_ConstantMean_Maternp_REMAP
from .models import Model_ConstantMean_Maternp_REMAP_logsigma2
from .models import Model_ConstantMean_Maternp_REMAP_logsigma2_and_logrho_prior
from .models import Model_ConstantMean_Maternp_ML
from .models import Model_Noisy_ConstantMean_Maternp_REML
from . import modelcontainer
from .sequentialprediction import SequentialPrediction
from .sequentialstrategy import (
    SequentialStrategyGridSearch,
    SequentialStrategySMC,
    SequentialStrategyBSS,
)
from . import samplingcriteria
from . import optim
from . import regp


def _read_version():
    version_file = Path(__file__).resolve().parent.parent / "VERSION"
    try:
        version = version_file.read_text(encoding="utf-8").strip()
        return version if version else "0.0.0"
    except OSError:
        pass
    try:
        return importlib.metadata.version("gpmp-contrib")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


__version__ = _read_version()
