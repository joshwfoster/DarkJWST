"""
Shared Gaussian-process likelihood for the JWST line analyses.

The initial zero-signal global search optimizes amp and logNerror, using
finite search bounds. Its result is converted to amp and Nerror, where
Nerror = 10**logNerror, before being returned as a local-fit seed.

The returned likelihood has three numerical parameters:

    amp
        Gaussian-process covariance amplitude in physical flux-squared units.
    Nerror
        Multiplicative rescaling of the quoted standard-deviation errors.
    gagg2
        Signed dimensionless line-amplitude coordinate. The reference line is
        evaluated at reference_coupling=1e-11 and multiplied by gagg2.

Local nuisance limits are amp >= 0 and Nerror >= 0, with no upper limits.
The signal coordinate gagg2 remains signed and unconstrained.
A constant residual mean is profiled analytically at every evaluation.
"""

import warnings

import george
import numpy as np
import scipy.optimize as opt
from george import kernels
from numpy.linalg import LinAlgError


INVALID_NLL = 1e30


################################
###   Validate Fit Data      ###
################################

def validate_likelihood_data(wavelength, flux, error):
    """Validate the arrays passed to the Gaussian-process likelihood."""

    wavelength = np.ascontiguousarray(wavelength, dtype=np.float64)
    flux = np.ascontiguousarray(flux, dtype=np.float64)
    error = np.ascontiguousarray(error, dtype=np.float64)

    if wavelength.ndim != 1 or flux.ndim != 1 or error.ndim != 1:
        raise ValueError(
            "The wavelength, flux, and error arrays must be one-dimensional."
        )

    if not (len(wavelength) == len(flux) == len(error)):
        raise ValueError(
            "Likelihood-array lengths do not agree:\n"
            f"  wavelength: {len(wavelength)}\n"
            f"  flux:       {len(flux)}\n"
            f"  error:      {len(error)}"
        )

    if len(wavelength) < 2:
        raise ValueError(
            "At least two wavelength bins are required to construct the likelihood."
        )

    if not np.all(np.diff(wavelength) > 0):
        raise ValueError("The wavelength bins must be strictly increasing.")

    return wavelength, flux, error


################################
###   Wavelength-Bin Edges   ###
################################

def build_dimensionless_bin_edges(wavelength, mass, hc):
    """Construct wavelength-bin edges in the dimensionless forward-model coordinate."""

    centers = wavelength * mass / hc

    edges = np.empty(len(centers) + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])

    return edges


################################
###   Profile Constant Mean  ###
################################

def profile_constant_mean(gp, residual):
    """Return the maximum-likelihood constant mean for the computed GP covariance."""

    ones = np.ones_like(residual)
    cinv_vectors = gp.apply_inverse(np.column_stack((residual, ones)))

    numerator = np.dot(ones, cinv_vectors[:, 0])
    denominator = np.dot(ones, cinv_vectors[:, 1])

    if not np.isfinite(denominator) or denominator <= 0.0:
        raise LinAlgError(
            "Invalid GLS normalization while profiling the constant mean."
        )

    return numerator / denominator


################################
###   Nuisance Bounds        ###
################################

def get_global_nuisance_limits(flux):
    """Return finite bounds for the global search in (amp, logNerror)."""

    amplitude = np.ptp(flux)

    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError(
            f"The fitted flux range must be finite and positive, but is {amplitude}."
        )

    return {
        "amp": (0.0, 100.0 * amplitude**2),
        "logNerror": (-3.0, 9.0),
    }


def get_nuisance_limits():
    """Return local-fit bounds in physical (amp, Nerror) coordinates."""

    return {
        "amp": (0.0, None),
        "Nerror": (0.0, None),
    }


################################
###   Initial Nuisance Fit   ###
################################

def find_initial_nuisance_seed(like, global_limits):
    """Search in log-error coordinates; return a physical-coordinate seed."""

    bounds = [global_limits["amp"], global_limits["logNerror"]]

    def objective(parameters):
        amp, logNerror = parameters
        return like(
            amp=amp,
            Nerror=10.0**logNerror,
            gagg2=0.0,
        )

    result = opt.differential_evolution(
        objective,
        bounds,
        maxiter=200,
        popsize=20,
        init="sobol",
        tol=1e-5,
        polish=False,  # All local optimization is done later in physical coordinates.
        seed=0,
    )

    if (
        not np.all(np.isfinite(result.x))
        or not np.isfinite(result.fun)
        or result.fun >= INVALID_NLL
    ):
        raise RuntimeError(
            "The initial nuisance search did not find a valid likelihood point. "
            f"Optimizer message: {result.message}"
        )

    if not result.success:
        warnings.warn(
            "The global nuisance search did not report convergence; "
            "using its best finite point as the local-fit seed. "
            f"Optimizer message: {result.message}",
            RuntimeWarning,
            stacklevel=2,
        )

    amp, logNerror = result.x

    return {
        "amp": float(amp),
        "Nerror": float(10.0**logNerror),
    }


################################
###   Build Gaussian Process ###
################################

def build_gp(wavelength, error, amp, Nerror, kernel_metric):
    """Construct and compute the GP using physical nuisance coordinates."""

    if not np.all(np.isfinite([amp, Nerror])) or amp < 0.0 or Nerror < 0.0:
        raise ValueError("amp and Nerror must be finite and nonnegative.")

    if amp == 0.0 and Nerror == 0.0:
        raise LinAlgError("amp and Nerror cannot both vanish: zero covariance.")

    # A missing kernel means zero GP covariance, avoiding log(0) at amp=0.
    kernel = None if amp == 0.0 else amp * kernels.ExpSquaredKernel(
        metric=kernel_metric
    )

    gp = george.GP(kernel)
    gp.compute(wavelength, error * Nerror)

    return gp


################################
###   Build Likelihood       ###
################################

def build_likelihood(
    wavelength,
    flux,
    error,
    mass,
    galactic_l,
    galactic_b,
    instrument_index,
    forward_model,
    velocity_parameters,
    reference_coupling=1e-11,
):
    """
    Construct the GP likelihood for one mass and dataset.

    The initial zero-signal search uses (amp, logNerror) with finite bounds.
    The returned likelihood, seed, and limits use physical (amp, Nerror)
    coordinates for all subsequent local fits.

    Parameters
    ----------
    wavelength, flux, error : array_like
        Spectral wavelength, flux, and uncertainty arrays in the fit window.
    mass : float
        Axion mass used to construct the line model.
    galactic_l, galactic_b : float
        Galactic longitude and latitude in degrees.
    instrument_index
        NIRSpec grating name or MIRI instrument index.
    forward_model
        Initialized JWST line forward-model object.
    velocity_parameters : dict
        Parameters passed to line_FWHM and forward_model.
    reference_coupling : float, optional
        Coupling used to generate the reference line model.

    Returns
    -------
    like : callable
        like(amp, Nerror, gagg2), returning -2 log L with a profiled mean.
    seed0 : dict
        Initial amp and Nerror values converted from the global search.
    nuisance_limits : dict
        Local limits: amp in [0, infinity), Nerror in [0, infinity).
        gagg2 is not bounded by this dictionary.
    """

    wavelength, flux, error = validate_likelihood_data(
        wavelength,
        flux,
        error,
    )

    bin_edges = build_dimensionless_bin_edges(
        wavelength,
        mass,
        forward_model.hc,
    )

    line_width = forward_model.line_FWHM(
        mass,
        instrument_index,
        **velocity_parameters,
    )

    kernel_metric = (3.0 * line_width) ** 2

    reference_line = forward_model.forward_model(
        bin_edges,
        galactic_l,
        galactic_b,
        reference_coupling,
        mass,
        instrument_index,
        **velocity_parameters,
    )

    if reference_line.shape != flux.shape:
        raise RuntimeError(
            "The reference-line shape does not match the fitted flux:\n"
            f"  reference line: {reference_line.shape}\n"
            f"  flux:           {flux.shape}"
        )

    def like(amp, Nerror, gagg2):
        if not np.isfinite(gagg2):
            return INVALID_NLL

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                residual = flux - gagg2 * reference_line

                gp = build_gp(
                    wavelength,
                    error,
                    amp,
                    Nerror,
                    kernel_metric,
                )

                mean = profile_constant_mean(gp, residual)
                value = -2.0 * gp.lnlikelihood(residual - mean)

        except (ValueError, LinAlgError, FloatingPointError, OverflowError):
            return INVALID_NLL

        return float(value) if np.isfinite(value) else INVALID_NLL

    global_limits = get_global_nuisance_limits(flux)
    seed0 = find_initial_nuisance_seed(like, global_limits)
    nuisance_limits = get_nuisance_limits()

    return like, seed0, nuisance_limits
