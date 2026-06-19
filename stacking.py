"""Module for stacking lensing results after pre-computation."""

import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.units import UnitConversionError
from . import surveys
from .physics import mpc_per_degree, lens_magnification_shear_bias
from scipy.optimize import minimize
import mlx.core as mx
import mlx.optimizers as optim

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


def gaussian(x: mx.array, mean: mx.array, std: mx.array) -> mx.array:
    # Normalised Gaussian — models the redshift distribution of cluster contaminants
    return mx.exp(-0.5 * ((x - mean) / std) ** 2) / (
        std * mx.sqrt(mx.array(2.0 * np.pi))
    )


def model_fn(mean, std, f, z_mid, background):
    """
    mean, std : scalars — shared Gaussian centre and width across all rp bins
    f         : (n_rp,) — per-bin contamination fraction
    z_mid     : (n_z,)
    background: (1, n_z) or (n_rp, n_z)
    returns   : (n_rp, n_z)
    """
    g = gaussian(z_mid, mean, std)  # (n_z,) cluster p(z) template
    print(f"g shape: {g.shape}")
    print(f"f shape: {f.shape}")  # must be (n_rp,) = (17,)
    print(f"background shape: {background.shape}")  # must be (1, 899)
    # Each rp bin's p(z) = f * cluster + (1-f) * background
    return f[:, None] * g[None, :] + (1.0 - f)[:, None] * background


def chi2_fn(mean, std, f, data, z_mid, background):
    model = model_fn(mean, std, f, z_mid, background)
    residuals = model - data  # (n_rp, n_z) per-bin, per-z residuals
    # Mask empty redshift cells multiplicatively — avoids boolean scatter on GPU
    weights = mx.maximum(data, mx.zeros_like(data))
    return mx.sum((residuals**2) * (weights > 0))


# Compute loss and gradients w.r.t. mean, std, and f in one Metal pass
_chi2_grad = mx.value_and_grad(chi2_fn, argnums=(0, 1, 2))

# ── bounded gradient descent with Adam + projection ───────────────────────────


def _project(mean, std, f):
    # Hard-clip parameters back into physical bounds after each gradient step
    mean = mx.clip(mean, 0.0, 2.0)  # cluster redshift in [0, 2]
    std = mx.clip(std, 0.0001, 0.15)  # width must be positive and narrow
    f = mx.clip(f, 0.0, 1.0)  # contamination fraction in [0, 1]
    return mean, std, f


def get_boost(
    table_l,
    table_r=None,
    rp=None,
    n_iter=10000,
    lr=1e-3,
    tol=1e-5,
    returnfullparams=False,
    plot=False,
):
    n_rp = len(rp)
    resultpz = get_pz(table_l, table_r)

    # Transfer data to MLX arrays — from this point all ops run on the M4 Pro GPU
    data = mx.array(resultpz["pz_l"].astype(np.float64))  # (n_rp, n_z) observed p(z)
    z_mid = mx.array(resultpz.meta["z_mid"].astype(np.float64))  # (n_z,) redshift grid
    if "pz_r" in resultpz.colnames:
        background = mx.array(resultpz["pz_r"].astype(np.float64))[
            None, :
        ]  # explicit background catalogue
    else:
        background = data[-1:, :]  # fall back to outermost rp bin as background proxy

    # Initialise
    mean = mx.array(0.4)
    std = mx.array(0.05)
    f = mx.full((n_rp,), 0.3)

    # Cosine decay: fast early steps, fine-grained near convergence
    scheduler = optim.cosine_decay(init=lr, decay_steps=n_iter)
    optimizer = optim.Adam(learning_rate=scheduler)

    prev_loss = float("inf")

    def step(mean, std, f):
        (loss, (g_mean, g_std, g_f)) = _chi2_grad(mean, std, f, data, z_mid, background)
        params = {"mean": mean, "std": std, "f": f}
        grads = {"mean": g_mean, "std": g_std, "f": g_f}
        updated = optimizer.apply_gradients(grads, params)
        mean, std, f = updated["mean"], updated["std"], updated["f"]
        mean, std, f = _project(mean, std, f)
        return mean, std, f, loss

    for i in range(n_iter):
        mean, std, f, loss = step(mean, std, f)
        mx.eval(mean, std, f)

        loss_val = float(loss)
        if i % 200 == 0:
            print(
                f"iter {i:4d}  loss={loss_val:.6e}  mean={float(mean):.4f}  std={float(std):.5f}"
            )

        # Stop early if loss change is below tolerance
        if abs(loss_val - prev_loss) < tol:
            print(f"Converged at iter {i}  loss={loss_val:.6e}")
            break
        prev_loss = loss_val

    boost_factors = 1.0 / (1.0 - np.array(f))

    if plot:
        model = model_fn(mean, std, f, z_mid, background)
        mx.eval(model)

        # Materialise all MLX arrays once before plotting
        model_np = np.array(model)
        data_np = np.array(data)
        z_mid_np = np.array(z_mid)
        residuals_np = model_np - data_np
        mean_f = float(mean)
        std_f = float(std)

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(3 * 8.6 / 2.54, 8.6 / 2.54))

        ax = axes[0]
        for i in range(len(rp)):
            ax.plot(
                z_mid_np, data_np[i], color=f"C{i}", lw=1.5, label=f"rp={rp[i]:.2f}"
            )
            ax.plot(z_mid_np, model_np[i], color=f"C{i}", lw=1.5, ls="--")
        ax.set_xlabel("z")
        ax.set_ylabel("p(z)")
        ax.set_title("Solid=data, dashed=model")
        ax.legend(fontsize=6)

        ax = axes[1]
        ax.plot(rp, boost_factors, "o-")
        ax.axhline(1.0, color="k", ls="--", lw=0.8)
        ax.set_xlabel("rp")
        ax.set_ylabel("B(rp) = 1/(1-f)")
        ax.set_title(f"mean={mean_f:.4f}, std={std_f:.4f}")

        ax = axes[2]
        for i in range(len(rp)):
            ax.plot(
                z_mid_np,
                residuals_np[i],
                color=f"C{i}",
                lw=1.0,
                label=f"rp={rp[i]:.2f}",
            )
        ax.axhline(0.0, color="k", ls="--", lw=0.8)
        ax.set_xlabel("z")
        ax.set_ylabel("model - data")
        ax.set_title("Residuals per rp bin")
        ax.legend(fontsize=6)

        plt.tight_layout()
        plt.show()

    if returnfullparams:
        return np.array([float(mean), float(std)] + list(np.array(f)))
    return boost_factors
