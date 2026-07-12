"""Compare fixed-p and free-nu Matérn REML models.

``Model_ConstantMean_Maternp_REML`` fixes ``nu = p + 1/2``.
``Model_ConstantMean_Matern_REML`` selects ``nu`` with the variance and
lengthscale parameters.
"""

import gpmp.num as gnp

import gpmpcontrib as gpc


def main():
    xi = gnp.linspace(0.0, 1.0, 16).reshape(-1, 1)
    zi = gnp.sin(2.0 * gnp.pi * xi[:, 0]) + 0.3 * gnp.abs(xi[:, 0] - 0.45)
    xt = gnp.linspace(0.0, 1.0, 100).reshape(-1, 1)
    p = 2

    maternp_model = gpc.Model_ConstantMean_Maternp_REML(
        "fixed_regularity",
        1,
        mean_specification={"type": "constant"},
        covariance_specification={"p": p},
    )
    matern_model = gpc.Model_ConstantMean_Matern_REML(
        "selected_regularity",
        1,
        mean_specification={"type": "constant"},
    )

    maternp_model.select_params(xi, zi)
    matern_model.select_params(xi, zi)

    maternp_model.predict(xi, zi, xt)
    matern_model.predict(xi, zi, xt)
    matern_model.run_diagnosis(xi, zi)

    param = matern_model[0].get_param()
    lognu = param.get_by_path(["covparam", "regularity"])
    nu = gnp.to_scalar(gnp.exp(gnp.asarray(lognu).reshape(-1)[0]))
    print(f"Fixed Matérn regularity: nu = {p + 0.5:g}")
    print(f"Selected Matérn regularity: nu = {nu:.6g}")


if __name__ == "__main__":
    main()
