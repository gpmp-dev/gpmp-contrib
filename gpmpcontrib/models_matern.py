# gpmpcontrib/models_matern.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""Backward-compatible exports for Matern model classes."""

from .models import (
    Model_ConstantMean_Maternp_ML,
    Model_ConstantMean_Maternp_REMAP,
    Model_ConstantMean_Maternp_REMAP_gaussian_logsigma2,
    Model_ConstantMean_Maternp_REMAP_gaussian_logsigma2_and_logrho_prior,
    Model_ConstantMean_Maternp_REML,
    Model_Noisy_ConstantMean_Maternp_REML,
)

__all__ = [
    "Model_ConstantMean_Maternp_REML",
    "Model_ConstantMean_Maternp_REMAP",
    "Model_ConstantMean_Maternp_REMAP_gaussian_logsigma2",
    "Model_ConstantMean_Maternp_REMAP_gaussian_logsigma2_and_logrho_prior",
    "Model_ConstantMean_Maternp_ML",
    "Model_Noisy_ConstantMean_Maternp_REML",
]
