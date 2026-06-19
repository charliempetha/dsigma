"""Module for stacking lensing results after pre-computation."""

import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.units import UnitConversionError
from . import surveys
from .physics import mpc_per_degree, lens_magnification_shear_bias

# try:
#     import mlx.core as mx
#     import mlx.optimizers as optim

#     BACKEND = "mlx"

# except ImportError:
try:
    import jax
    import jax.numpy as jnp
    import jaxopt

    BACKEND = "jax"

except ImportError:
    import numpy as np
    from scipy.optimize import minimize

    BACKEND = "scipy"

__all__ = [
    "number_of_pairs",
    "raw_tangential_shear",
    "raw_excess_surface_density",
    "photo_z_dilution_factor",
    "boost_factor",
    "scalar_shear_response_factor",
    "matrix_shear_response_factor",
    "shear_responsivity_factor",
    "mean_lens_redshift",
    "mean_source_redshift",
    "mean_critical_surface_density",
    "lens_magnification_bias",
    "tangential_shear",
    "excess_surface_density",
    "get_boost",
    "get_pz",
]


def number_of_pairs(table_l):
    """Compute the number of lens-source pairs per bin.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    n_pairs : numpy.ndarray
        The number of lens-source pairs in each radial bin.

    """
    return np.sum(table_l["sum 1"].data, axis=0)


def raw_tangential_shear(table_l):
    """Compute the average tangential shear for a catalog.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    delta_sigma : numpy.ndarray
        The raw, uncorrected tangential shear in each radial bin.

    """
    return np.sum(
        table_l["sum w_ls e_t"].data * table_l["w_sys"].data[:, None], axis=0
    ) / np.sum(table_l["sum w_ls"].data * table_l["w_sys"].data[:, None], axis=0)


def raw_cross_shear(table_l):
    """Compute the average tangential shear for a catalog.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    delta_sigma : numpy.ndarray
        The raw, uncorrected tangential shear in each radial bin.

    """
    return np.sum(
        table_l["sum w_ls e_x"].data * table_l["w_sys"].data[:, None], axis=0
    ) / np.sum(table_l["sum w_ls"].data * table_l["w_sys"].data[:, None], axis=0)


def raw_excess_surface_density(table_l):
    """Compute the raw, uncorrected excess surface density for a catalog.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    delta_sigma : numpy.ndarray
        The raw, uncorrected excess surface density in each radial bin.

    """
    return np.sum(
        table_l["sum w_ls e_t sigma_crit"].data * table_l["w_sys"].data[:, None], axis=0
    ) / np.sum(table_l["sum w_ls"].data * table_l["w_sys"].data[:, None], axis=0)


def excess_surface_density_cross(table_l):
    """Compute the cross excess surface density for a catalog.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    delta_sigma : numpy.ndarray
        The raw, uncorrected excess surface density in each radial bin.

    """
    return np.sum(
        table_l["sum w_ls e_x sigma_crit"].data * table_l["w_sys"].data[:, None], axis=0
    ) / np.sum(table_l["sum w_ls"].data * table_l["w_sys"].data[:, None], axis=0)


def photo_z_dilution_factor(table_l):
    r"""Compute the photometric redshift bias averaged over the entire catalog.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    f_bias : float
        Photometric redshift bias :math:`f_{\mathrm{bias}}`.

    """
    return np.sum(
        table_l["sum w_ls e_t sigma_crit f_bias"].data * table_l["w_sys"].data[:, None],
        axis=0,
    ) / np.sum(
        table_l["sum w_ls e_t sigma_crit"].data * table_l["w_sys"].data[:, None], axis=0
    )


def boost_factor(table_l, table_r):
    """Compute the boost factor.

    Boost factor is computed by comparing the number of lens-source pairs
    in real lenses and random lenses.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    table_r : astropy.table.Table, optional
        Precompute results for random lenses.

    Returns
    -------
    b : numpy.ndarray
        Boost factor in each radial bin.

    """
    return (
        np.sum(table_l["sum w_ls"].data * table_l["w_sys"].data[:, None], axis=0)
        / np.sum(table_l["w_sys"].data)
        / np.sum(table_r["sum w_ls"].data * table_r["w_sys"].data[:, None], axis=0)
        * np.sum(table_r["w_sys"].data)
    )


def scalar_shear_response_factor(table_l):
    r"""Compute the mean shear response.

    The shear response factor :math:`m` is defined such that
    :math:`\gamma_{\mathrm obs} = (1 + m) \gamma_{\mathrm intrinsic}`.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    m : numpy.ndarray
        Multiplicative shear bias in each radial bin.

    """
    return np.sum(
        table_l["sum w_ls m"].data * table_l["w_sys"].data[:, None], axis=0
    ) / np.sum(table_l["sum w_ls"].data * table_l["w_sys"].data[:, None], axis=0)


def matrix_shear_response_factor(table_l):
    r"""Compute the mean tangential response.

    The tangential shear response factor:math:`R_t` is defined such that
    :math:`\gamma_{\mathrm obs} = R_t \gamma_{\mathrm intrinsic}`.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    r_t : numpy.ndarray
        Tangential shear response factor in each radial bin.

    """
    return np.sum(table_l["sum w_ls R_T"] * table_l["w_sys"][:, None], axis=0) / np.sum(
        table_l["sum w_ls"] * table_l["w_sys"][:, None], axis=0
    )


def shear_responsivity_factor(table_l):
    """Compute the shear responsitivity factor.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    r : numpy.ndarray
        Shear responsitivity factor in each radial bin.

    """
    return np.sum(
        table_l["sum w_ls (1 - e_rms^2)"] * table_l["w_sys"][:, None], axis=0
    ) / np.sum(table_l["sum w_ls"] * table_l["w_sys"][:, None], axis=0)


def mean_lens_redshift(table_l):
    """Compute the weighted-average lens redshift.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    z_l : numpy.ndarray
        Mean lens redshift in each bin.

    """
    return np.sum(table_l["sum w_ls z_l"] * table_l["w_sys"][:, None], axis=0) / np.sum(
        table_l["sum w_ls"] * table_l["w_sys"][:, None], axis=0
    )


def mean_source_redshift(table_l):
    """Compute the weighted-average source redshift.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    z_s : numpy.ndarray
        Mean source redshift in each bin.

    """
    return np.sum(table_l["sum w_ls z_s"] * table_l["w_sys"][:, None], axis=0) / np.sum(
        table_l["sum w_ls"] * table_l["w_sys"][:, None], axis=0
    )


def mean_critical_surface_density(table_l, photo_z_dilution_correction=False):
    """Compute the weighted-average (effective) critical surface density.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    photo_z_dilution_correction : boolean, optional
        If True, correct for photo-z biases. This can only be done if a
        calibration catalog has been provided in the Precomputation phase.
        Default is False.

    Returns
    -------
    sigma_crit : numpy.ndarray
        Mean (effective) critical surface density.

    """
    if photo_z_dilution_correction:
        key = "sum w_ls sigma_crit f_bias"
    else:
        key = "sum w_ls sigma_crit"
    return np.sum(table_l[key] * table_l["w_sys"][:, None], axis=0) / np.sum(
        table_l["sum w_ls"] * table_l["w_sys"][:, None], axis=0
    )


def lens_magnification_bias(
    table_l, alpha_l, camb_results, photo_z_dilution_correction=False, shear=False
):
    """Estimate the additive lens magnification bias.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    alpha_l : float
        The response of the lenses to magnification.
    camb_results : camb.results.CAMBdata
        CAMB results object that contains information on cosmology and the
        matter power spectrum.
    photo_z_dilution_correction : boolean, optional
        If True, correct the mean critical surface density for photo-z biases.
        Not used if `shear` is True. This should be consistent with what is
        used for calculating the total excess surface density. Default is
        False.
    shear : boolean, optional
        If True, return bias of the mean tangential shear. Otherwise, return
        an estimate for the bias of the excess surface density. Default is
        False.

    Returns
    -------
    ds_lm : numpy.ndarray
        The lens magnification bias in each radial bin.

    """
    cosmology = FlatLambdaCDM(H0=table_l.meta["H0"], Om0=table_l.meta["Om0"])

    z_l = mean_lens_redshift(table_l)
    z_s = mean_source_redshift(table_l)
    bins = table_l.meta["bins"]
    d = 2.0 / 3.0 * np.diff(bins**3) / np.diff(bins**2)

    try:
        theta = d.to(u.rad).value
    except UnitConversionError:
        theta = np.deg2rad(
            d.to(u.Mpc).value
            / mpc_per_degree(
                z_l, cosmology=cosmology, comoving=table_l.meta["comoving"]
            )
        )

    gt = np.array(
        [
            lens_magnification_shear_bias(
                theta[i], alpha_l, z_l[i], z_s[i], camb_results
            )
            for i in range(len(theta))
        ]
    )

    if shear:
        return gt
    else:
        return gt * mean_critical_surface_density(
            table_l, photo_z_dilution_correction=photo_z_dilution_correction
        )


def tangential_shear(
    table_l,
    table_r=None,
    boost_correction=False,
    scalar_shear_response_correction=False,
    matrix_shear_response_correction=False,
    shear_responsivity_correction=False,
    hsc_selection_bias_correction=False,
    random_subtraction=False,
    cross=False,
    return_table=False,
):
    """Compute the mean tangential shear with corrections, if applicable.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    table_r : astropy.table.Table, optional
        Precompute results for random lenses. Default is None.
    boost_correction : boolean, optional
        If True, calculate and apply a boost factor correction. This can only
        be done if a random catalog is provided. Default is False.
    scalar_shear_response_correction : boolean or string, optional
        Whether to correct for the multiplicative shear bias (scalar form).
        Default is False.
    matrix_shear_response_correction : boolean or string, optional
        Whether to correct for the multiplicative shear bias (tensor form).
        Default is False.
    shear_responsivity_correction : boolean, optional
        If True, correct for the shear responsivity. Default is False.
    hsc_selection_bias_correction : boolean, optional
        If True, correct for the multiplicative selection bias in HSC. Default
        is False.
    random_subtraction : boolean, optional
        If True, subtract the signal around randoms. This can only be done if
        a random catalog is provided. Default is False.
    return_table : boolean, optional
        If True, return a table with many intermediate steps of the
        computation. Otherwise, a simple array with just the final tangential
        shearis returned. Default is False.

    Returns
    -------
    e_t : numpy.ndarray or astropy.table.Table
        The tangential shear in each radial bin specified in the precomputation
        phase. If `return_table` is True, will return a table with detailed
        information for each radial bin. The final result is in the column
        `et`.

    Raises
    ------
    ValueError
        If boost or random subtraction correction are requested but no random
        catalog is provided.

    """
    result = Table()

    result["rp_min"] = table_l.meta["bins"][:-1]
    result["rp_max"] = table_l.meta["bins"][1:]
    result["n_pairs"] = number_of_pairs(table_l)
    result["rp"] = np.sqrt(result["rp_min"] * result["rp_max"])
    result["et_raw"] = raw_tangential_shear(table_l)
    result["et"] = raw_tangential_shear(table_l)
    result["ex"] = raw_cross_shear(table_l)
    result["z_l"] = mean_lens_redshift(table_l)
    result["z_s"] = mean_source_redshift(table_l)

    if not return_table:
        if cross:
            return result["ex"].data

    if boost_correction:
        if table_r is None:
            raise ValueError(
                "Cannot compute boost factor correction without"
                + " results from a random catalog."
            )
        result["b"] = boost_factor(table_l, table_r)
        result["et"] *= result["b"]

    if scalar_shear_response_correction:
        result["1+m"] = 1 + scalar_shear_response_factor(table_l)
        result["et"] /= result["1+m"]

    if matrix_shear_response_correction:
        result["R_t"] = matrix_shear_response_factor(table_l)
        result["et"] /= result["R_t"]

    if shear_responsivity_correction:
        result["2R"] = 2 * shear_responsivity_factor(table_l)
        result["et"] /= result["2R"]

    if hsc_selection_bias_correction:
        result["1+m_sel"] = 1 + surveys.hsc.selection_bias_factor(table_l)
        result["et"] *= result["1+m_sel"]

    if random_subtraction:
        if table_r is None:
            raise ValueError(
                "Cannot subtract random results without "
                + "results from a random catalog."
            )
        result["et_r"] = tangential_shear(
            table_r,
            boost_correction=False,
            scalar_shear_response_correction=scalar_shear_response_correction,
            matrix_shear_response_correction=matrix_shear_response_correction,
            shear_responsivity_correction=shear_responsivity_correction,
            hsc_selection_bias_correction=hsc_selection_bias_correction,
            random_subtraction=False,
            return_table=False,
        )
        result["et"] -= result["et_r"]

    if not return_table:
        return result["et"].data

    return result


def excess_surface_density(
    table_l,
    table_r=None,
    photo_z_dilution_correction=False,
    boost_correction=False,
    scalar_shear_response_correction=False,
    matrix_shear_response_correction=False,
    shear_responsivity_correction=False,
    hsc_selection_bias_correction=False,
    random_subtraction=False,
    return_table=False,
    cross=False,
):
    """Compute the mean excess surface density with corrections, if applicable.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    table_r : astropy.table.Table, optional
        Precompute results for random lenses. Default is None.
    photo_z_dilution_correction : boolean, optional
        If True, correct for photo-z biases. This can only be done if a
        calibration catalog has been provided in the precomputation phase.
        Default is False.
    boost_correction : boolean, optional
        If true, calculate and apply a boost factor correction. This can only
        be done if a random catalog is provided. Default is False.
    scalar_shear_response_correction : boolean or string, optional
        Whether to correct for the multiplicative shear bias (scalar form).
        Default is False.
    matrix_shear_response_correction : boolean or string, optional
        Whether to correct for the multiplicative shear bias (tensor form).
        Default is False.
    shear_responsivity_correction : boolean, optional
        If True, correct for the shear responsivity. Default is False.
    hsc_selection_bias_correction : boolean, optional
        If True, correct for the multiplicative selection bias in HSC. Default
        is False.
    random_subtraction : boolean, optional
        If True, subtract the signal around randoms. This can only be done if
        a random catalog is provided. Default is False.
    return_table : boolean, optional
        If True, return a table with many intermediate steps of the
        computation. Otherwise, a simple array with just the final excess
        surface density is returned. Default is False.

    Returns
    -------
    delta_sigma : numpy.ndarray or astropy.table.Table
        The excess surface density in each radial bin specified in the
        precomputation phase. If `return_table` is True, will return a table
        with detailed information for each radial bin. The final result is in
        the column `ds`.

    Raises
    ------
    ValueError
        If boost or random subtraction correction are requested but no random
        catalog is provided.

    """
    result = Table()

    result["rp_min"] = table_l.meta["bins"][:-1]
    result["rp_max"] = table_l.meta["bins"][1:]
    result["theta_min"] = table_l.meta["thetas"][:-1]
    result["theta_max"] = table_l.meta["thetas"][1:]
    result["n_pairs"] = number_of_pairs(table_l)
    result["rp"] = np.sqrt(result["rp_min"] * result["rp_max"])
    result["ds_raw"] = raw_excess_surface_density(table_l)
    result["ds"] = raw_excess_surface_density(table_l)
    result["ds_x"] = excess_surface_density_cross(table_l)
    result["z_l"] = mean_lens_redshift(table_l)
    result["z_s"] = mean_source_redshift(table_l)

    if not return_table:
        if cross:
            return result["ds_x"].data

    if boost_correction:
        if table_r is None:
            raise ValueError(
                "Cannot compute boost factor correction without"
                + " results from a random catalog."
            )
        result["b"] = boost_factor(table_l, table_r)
        result["ds"] *= result["b"]

    if scalar_shear_response_correction:
        result["1+m"] = 1 + scalar_shear_response_factor(table_l)
        result["ds"] /= result["1+m"]

    if matrix_shear_response_correction:
        result["R_t"] = matrix_shear_response_factor(table_l)
        result["ds"] /= result["R_t"]

    if shear_responsivity_correction:
        result["2R"] = 2 * shear_responsivity_factor(table_l)
        result["ds"] /= result["2R"]

    if hsc_selection_bias_correction:
        result["1+m_sel"] = 1 + surveys.hsc.selection_bias_factor(table_l)
        result["ds"] *= result["1+m_sel"]

    if photo_z_dilution_correction:
        result["f_bias"] = photo_z_dilution_factor(table_l)
        result["ds"] *= result["f_bias"]

    if random_subtraction:
        if table_r is None:
            raise ValueError(
                "Cannot subtract random results without "
                + "results from a random catalog."
            )
        result["ds_r"] = excess_surface_density(
            table_r,
            photo_z_dilution_correction=photo_z_dilution_correction,
            boost_correction=False,
            scalar_shear_response_correction=scalar_shear_response_correction,
            matrix_shear_response_correction=matrix_shear_response_correction,
            shear_responsivity_correction=shear_responsivity_correction,
            hsc_selection_bias_correction=hsc_selection_bias_correction,
            random_subtraction=False,
            return_table=False,
        )
        result["ds"] -= result["ds_r"]

    if not return_table:
        return result["ds"].data

    return result


def _get_pz(table):
    # Redshift bin centres from edges stored in table metadata
    zmid = 0.5 * (table.meta["zbins_pdf"][1:] + table.meta["zbins_pdf"][:-1])
    # Stack p(z) across all lens-source pairs, weighted by w_ls, then normalise
    pzbf = np.sum(table["sum_pzbf"], axis=0) / np.sum(table["sum w_ls"])
    pzbf = pzbf / np.trapezoid(pzbf, zmid, axis=1).reshape(
        -1, 1
    )  # unit-normalise each rp bin
    return pzbf


def get_pz(table_l, table_r=None):
    result = Table()
    result["pz_l"] = _get_pz(table_l)  # p(z) for lenses
    if table_r is not None:
        result["pz_r"] = _get_pz(table_r)  # p(z) for randoms
    result.meta["z_mid"] = 0.5 * (
        table_l.meta["zbins_pdf"][1:] + table_l.meta["zbins_pdf"][:-1]
    )
    return result


if BACKEND == "mlx":

    def gaussian(x, mean, std):
        return mx.exp(-0.5 * ((x - mean) / std) ** 2) / (
            std * mx.sqrt(mx.array(2.0 * np.pi))
        )

    def model_fn(mean, std, f, z_mid, background):
        g = gaussian(z_mid, mean, std)
        return f[:, None] * g[None, :] + (1.0 - f)[:, None] * background

    def chi2_fn(mean, std, f, data, z_mid, background):
        model = model_fn(mean, std, f, z_mid, background)
        residuals = model - data
        mask = data > 0
        return mx.sum((residuals**2) * mask)

    _chi2_grad = mx.value_and_grad(chi2_fn, argnums=(0, 1, 2))

    def _project(mean, std, f):
        mean = mx.clip(mean, 0.0, 2.0)
        std = mx.clip(std, 1e-4, 0.15)
        f = mx.clip(f, 0.0, 1.0)
        return mean, std, f

    def optimize(table_l, table_r=None, rp=None, n_iter=10000, lr=1e-3, tol=1e-5):
        resultpz = get_pz(table_l, table_r)

        data = mx.array(resultpz["pz_l"].astype(np.float64))
        z_mid = mx.array(resultpz.meta["z_mid"].astype(np.float64))

        if "pz_r" in resultpz.colnames:
            background = mx.array(resultpz["pz_r"].astype(np.float64))
        else:
            background = data[-1:, :]

        mean = mx.array(0.4)
        std = mx.array(0.05)
        f = mx.full((len(rp),), 0.3)

        scheduler = optim.cosine_decay(init=lr, decay_steps=n_iter)
        optimizer = optim.Adam(learning_rate=scheduler)

        prev_loss = float("inf")

        for i in range(n_iter):
            loss, (g_mean, g_std, g_f) = _chi2_grad(
                mean, std, f, data, z_mid, background
            )

            params = {"mean": mean, "std": std, "f": f}
            grads = {"mean": g_mean, "std": g_std, "f": g_f}

            updated = optimizer.apply_gradients(grads, params)

            mean, std, f = updated["mean"], updated["std"], updated["f"]
            mean, std, f = _project(mean, std, f)
            mx.eval(mean, std, f)

            loss_val = float(loss)

            if abs(loss_val - prev_loss) < tol:
                break
            prev_loss = loss_val

        return float(mean), float(std), np.array(f), resultpz

elif BACKEND == "jax":

    def gaussian(x, mean, std):
        return (
            1.0
            / jnp.sqrt(2.0 * jnp.pi)
            / std
            * jnp.exp(-((x - mean) ** 2) / (2.0 * std**2))
        )

    def model_fn(params, resultpz):
        mean = params[0]
        std = params[1]
        f = params[2:]

        if "pz_r" in resultpz.colnames:
            background = resultpz["pz_r"]
        else:
            background = resultpz["pz_l"][-1].reshape(1, -1)

        g = gaussian(resultpz.meta["z_mid"], mean, std)
        return f[:, None] * g.reshape(1, -1) + (1 - f)[:, None] * background

    def chi2(params, resultpz):
        data = resultpz["pz_l"]
        model = model_fn(params, resultpz)
        return jnp.sum(((model - data)[data > 0]) ** 2)

    def optimize(table_l, table_r=None, rp=None, **kwargs):
        resultpz = get_pz(table_l, table_r)

        params_init = jnp.array([0.5, 0.01] + [0.5] * len(rp))

        bounds = [(0.0, 2), (1e-4, 0.15)]
        bounds.extend([(0.0, 1.0)] * len(rp))

        fun = lambda x: chi2(x, resultpz)

        solver = jaxopt.ScipyBoundedMinimize(
            fun=fun,
            method="L-BFGS-B",
            tol=1e-6,
            maxiter=600,
        )

        res = solver.run(params_init, bounds=jnp.array(bounds).T)

        params = np.array(res.params)
        return params[0], params[1], params[2:], resultpz

else:

    def gaussian(x, mean, std):
        return (
            1.0
            / np.sqrt(2.0 * np.pi)
            / std
            * np.exp(-((x - mean) ** 2) / (2.0 * std**2))
        )

    def model_fn(params, resultpz):
        mean = params[0]
        std = params[1]
        f = params[2:]

        if "pz_r" in resultpz.colnames:
            background = resultpz["pz_r"]
        else:
            background = resultpz["pz_l"][-1].reshape(1, -1)

        g = gaussian(resultpz.meta["z_mid"], mean, std)
        return f[:, None] * g.reshape(1, -1) + (1 - f)[:, None] * background

    def chi2(params, resultpz):
        data = resultpz["pz_l"]
        model = model_fn(params, resultpz)
        return np.sum(((model - data)[data > 0]) ** 2)

    def optimize(table_l, table_r=None, rp=None, **kwargs):
        resultpz = get_pz(table_l, table_r)

        params_init = np.array([0.5, 0.01] + [0.5] * len(rp))

        bounds = [(0.0, 2), (1e-4, 0.15)]
        bounds.extend([(0.0, 1.0)] * len(rp))

        res = minimize(
            lambda x: chi2(x, resultpz),
            params_init,
            bounds=bounds,
            method="L-BFGS-B",
        )

        params = res.x
        return params[0], params[1], params[2:], resultpz


def get_boost(
    table_l,
    table_r=None,
    rp=None,
    returnfullparams=False,
    plot=False,
    **kwargs,
):
    print("Fitting boost factor model to p(z) data using backend:", BACKEND)
    mean, std, f, resultpz = optimize(
        table_l,
        table_r=table_r,
        rp=rp,
        **kwargs,
    )
    if plot:
        plot_boost_fit(mean, std, f, resultpz, rp)
    if returnfullparams:
        return np.array([mean, std] + list(f))

    return 1.0 / (1.0 - np.array(f))


def evaluate_boost_fit(
    mean,
    std,
    f,
    resultpz,
    rp,
):
    z_mid = np.array(resultpz.meta["z_mid"])
    data = np.array(resultpz["pz_l"])

    if "pz_r" in resultpz.colnames:
        background = np.array(resultpz["pz_r"])
    else:
        background = data[-1:, :]

    # Gaussian model
    g = np.exp(-0.5 * ((z_mid - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

    model = f[:, None] * g[None, :] + (1 - f)[:, None] * background

    frac_res = (model - data) / (data + 1e-8)
    boost = 1.0 / (1.0 - f)

    return z_mid, data, model, frac_res, boost


def plot_boost_fit(mean, std, f, resultpz, rp):
    import matplotlib.pyplot as plt

    z_mid, data, model, frac_res, boost = evaluate_boost_fit(mean, std, f, resultpz, rp)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # -------------------------
    # (1) p(z): data vs model
    # -------------------------
    ax = axes[0]
    for i in range(len(rp)):
        ax.plot(z_mid, data[i], label=f"rp={rp[i]:.2f}")
        ax.plot(z_mid, model[i], linestyle="--")

    ax.set_title("p(z): data vs model")
    ax.set_xlabel("z")
    ax.set_ylabel("p(z)")
    ax.legend(fontsize=7)

    # -------------------------
    # (2) contamination / boost
    # -------------------------
    ax = axes[1]
    ax.semilogx(rp, boost, "o-")
    ax.axhline(1.0, linestyle="--", color="k")
    ax.set_title("Boost factor")
    ax.set_xlabel("r_p")
    ax.set_ylabel("B(r_p)")

    # -------------------------
    # (3) residuals
    # -------------------------
    ax = axes[2]
    for i in range(len(rp)):
        ax.plot(z_mid, frac_res[i])

    ax.axhline(0, color="k", linestyle="--")
    ax.set_title("Fractional Residuals")
    ax.set_xlabel("z")
    ax.set_ylabel(" (model - data) / data")

    plt.tight_layout()
    plt.savefig(f"boost_fit_{BACKEND}.png", dpi=300)
    plt.show()
