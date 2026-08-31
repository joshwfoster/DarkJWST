"""
Shared Gaussian-process likelihood for the JWST line analyses.

The baseline likelihood has three numerically optimized parameters:

    logamp
        Base-10 logarithm of the Gaussian-process kernel amplitude.

    logNerror
        Base-10 logarithm of the multiplicative error rescaling.

    gagg2
        Signed dimensionless line-amplitude coordinate. The reference line is
        evaluated at reference_coupling=1e-11 and multiplied by gagg2.

A constant residual mean is profiled analytically at every likelihood
calculation using generalized least squares. For the fixed-error systematic,
logNerror is removed and the quoted spectral errors are used directly.
"""


import george
import numpy as np
import scipy.optimize as opt
from george import kernels
from numpy.linalg import LinAlgError

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

def get_nuisance_limits(flux):
    """Return the nuisance-parameter limits used in the analysis."""

    amplitude = np.ptp(flux)

    if not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError(
            f"The fitted flux range must be finite and positive, but is {amplitude}."
        )

    return {
        "logamp": (-15.0, np.log10(amplitude**2)),
        "logNerror": (-2.0, 7.0),
    }


################################
###   Initial Nuisance Fit   ###
################################

def find_initial_nuisance_seed(like, nuisance_limits):
    """Find an initial nuisance solution with the signal fixed to zero."""

    nuisance_names = list(nuisance_limits)
    bounds = [nuisance_limits[name] for name in nuisance_names]

    def objective(parameters):
        return like(gagg2=0.0, **dict(zip(nuisance_names, parameters)))

    result = opt.differential_evolution(objective, bounds, maxiter=200, popsize=20,
                                        init="sobol", tol=1e-5, polish=True, seed=0)

    if not np.all(np.isfinite(result.x)):
        raise RuntimeError(
            f"The initial nuisance fit returned nonfinite parameters: {result.x}"
        )

    return {
        name: float(value)
        for name, value in zip(nuisance_names, result.x)
    }


################################
###   Build Likelihood       ###
################################

def build_gp(wavelength, error, logamp, logNerror, kernel_metric):
    """Construct and compute the Gaussian process."""

    kernel = 10.0**logamp * kernels.ExpSquaredKernel(metric=kernel_metric)
    gp = george.GP(kernel)
    gp.compute(wavelength, error * 10.0**logNerror)

    return gp




def build_likelihood(wavelength, flux, error, mass, galactic_l, galactic_b, instrument_index,
                     forward_model, velocity_parameters, reference_coupling=1e-11):
    """
    Construct the Gaussian-process likelihood for one mass and dataset.

    Parameters
    ----------
    wavelength, flux, error : array_like
        Spectral wavelength, flux, and uncertainty arrays over the local fitting window.

    mass : float
        Axion mass used to construct the line model.

    galactic_l, galactic_b : float
        Galactic longitude and latitude in degrees.

    instrument_index
        NIRSpec grating name or MIRI instrument index.

    forward_model
        Initialized JWST line forward-model object.

    velocity_parameters : dict
        Parameters passed to ``line_FWHM`` and ``forward_model``.

    reference_coupling : float, optional
        Coupling used to generate the reference line model.

    Returns
    -------
    like : callable
        Likelihood function returning -2 log L as a function of ``logamp``,
        ``logNerror``, and ``gagg2``.

    seed0 : dict
        Initial nuisance-parameter values obtained from the zero-signal fit.

    nuisance_limits : dict
        Bounds used for the numerically optimized nuisance parameters.
    """
    
    wavelength, flux, error = validate_likelihood_data(wavelength, flux, error)

    bin_edges = build_dimensionless_bin_edges(wavelength, mass, forward_model.hc)

    line_width = forward_model.line_FWHM(mass, instrument_index, **velocity_parameters)
    kernel_metric = (3.0 * line_width) ** 2

    reference_line = forward_model.forward_model(
        bin_edges, galactic_l, galactic_b, reference_coupling, mass,
        instrument_index, **velocity_parameters
    )

    if reference_line.shape != flux.shape:
        raise RuntimeError(
            "The reference-line shape does not match the fitted flux:\n"
            f"  reference line: {reference_line.shape}\n"
            f"  flux:           {flux.shape}"
        )

    def like(logamp, logNerror, gagg2):
        residual = flux - gagg2 * reference_line
    
        try:
            gp = build_gp(wavelength, error, logamp, logNerror, kernel_metric)
            mean = profile_constant_mean(gp, residual)
            return -2.0 * gp.lnlikelihood(residual - mean)
        except (ValueError, LinAlgError):
            return 1e30

    nuisance_limits = get_nuisance_limits(flux)
    seed0 = find_initial_nuisance_seed(like, nuisance_limits)

    return like, seed0, nuisance_limits