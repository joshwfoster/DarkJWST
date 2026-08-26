"""
Shared Gaussian-process likelihood for the JWST line analyses.

The baseline likelihood has three free parameters:

    logamp
        Base-10 logarithm of the Gaussian-process kernel amplitude.

    logNerror
        Base-10 logarithm of the multiplicative error rescaling.

    gagg2
        Signed dimensionless line-amplitude coordinate. The reference line is
        evaluated at reference_coupling=1e-11 and multiplied by gagg2.

For the fixed-error systematic, logNerror is removed from the likelihood and
the quoted spectral errors are used without rescaling.
"""

from numpy.linalg import LinAlgError

import george
import numpy as np
import scipy.optimize as opt
from george import kernels


################################
###   Validate Fit Data      ###
################################

def validate_likelihood_data(
    wavelength,
    flux,
    error,
):
    """Validate the arrays passed to the Gaussian-process likelihood."""

    wavelength = np.ascontiguousarray(
        wavelength,
        dtype=np.float64,
    )

    flux = np.ascontiguousarray(
        flux,
        dtype=np.float64,
    )

    error = np.ascontiguousarray(
        error,
        dtype=np.float64,
    )

    if (
        wavelength.ndim != 1
        or flux.ndim != 1
        or error.ndim != 1
    ):
        raise ValueError(
            "The wavelength, flux, and error arrays "
            "must be one-dimensional."
        )

    if not (
        len(wavelength)
        == len(flux)
        == len(error)
    ):
        raise ValueError(
            "Likelihood-array lengths do not agree:\n"
            f"  wavelength: {len(wavelength)}\n"
            f"  flux:       {len(flux)}\n"
            f"  error:      {len(error)}"
        )

    if len(wavelength) < 2:
        raise ValueError(
            "At least two wavelength bins are required "
            "to construct the likelihood."
        )

    if not np.all(
        np.diff(wavelength) > 0
    ):
        raise ValueError(
            "The wavelength bins must be strictly increasing."
        )

    return wavelength, flux, error


################################
###   Wavelength-Bin Edges   ###
################################

def build_dimensionless_bin_edges(
    wavelength,
    mass,
    hc,
):
    """
    Construct wavelength-bin edges in the dimensionless coordinate used by
    the line forward model.
    """

    centers = (
        wavelength
        * mass
        / hc
    )

    edges = np.empty(
        len(centers) + 1,
        dtype=np.float64,
    )

    edges[1:-1] = (
        0.5
        * (
            centers[:-1]
            + centers[1:]
        )
    )

    edges[0] = (
        centers[0]
        - 0.5
        * (
            centers[1]
            - centers[0]
        )
    )

    edges[-1] = (
        centers[-1]
        + 0.5
        * (
            centers[-1]
            - centers[-2]
        )
    )

    return edges


################################
###   Nuisance Bounds        ###
################################

def get_nuisance_limits(
    flux,
    *,
    float_error_scale=True,
):
    """Return the nuisance-parameter limits used in the analysis."""

    amplitude = (
        np.max(flux)
        - np.min(flux)
    )

    if (
        not np.isfinite(amplitude)
        or amplitude <= 0
    ):
        raise ValueError(
            "The fitted flux range must be finite "
            f"and positive, but is {amplitude}."
        )

    max_logamp = np.log10(
        amplitude**2
    )

    nuisance_limits = {
        "logamp": (
            -15.0,
            max_logamp,
        ),
    }

    if float_error_scale:
        nuisance_limits[
            "logNerror"
        ] = (
            -2.0,
            5.0,
        )

    return nuisance_limits


################################
###   Initial Nuisance Fit   ###
################################

def find_initial_nuisance_seed(
    like,
    nuisance_limits,
):
    """
    Find an initial nuisance-parameter solution with the signal fixed to zero.

    The differential-evolution settings reproduce the production analysis.
    """

    nuisance_names = list(
        nuisance_limits
    )

    bounds = [
        nuisance_limits[name]
        for name in nuisance_names
    ]

    def objective(parameters):

        likelihood_parameters = {
            name: parameters[index]
            for index, name
            in enumerate(nuisance_names)
        }

        return like(
            gagg2=0.0,
            **likelihood_parameters,
        )

    result = opt.differential_evolution(
        objective,
        bounds,
        maxiter=200,
        popsize=20,
        init="sobol",
        tol=1e-5,
        polish=True,
        workers=1,
        seed=0,
    )

    if not np.all(
        np.isfinite(result.x)
    ):
        raise RuntimeError(
            "The initial nuisance fit returned "
            f"nonfinite parameters: {result.x}"
        )

    return {
        name: float(
            result.x[index]
        )
        for index, name
        in enumerate(nuisance_names)
    }


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
    float_error_scale=True,
):
    """
    Construct the Gaussian-process likelihood for one mass and dataset.

    Parameters
    ----------
    wavelength, flux, error
        Spectral arrays after selecting the local fitting window.

    mass
        Axion mass used by the line forward model.

    galactic_l, galactic_b
        Galactic longitude and latitude in degrees.

    instrument_index
        NIRSpec grating name or MIRI instrument index 0.

    forward_model
        Initialized JWST Line_Forward_Model object.

    velocity_parameters
        Dictionary passed to line_FWHM() and forward_model().

    reference_coupling
        Physical coupling used to generate the reference line.

    float_error_scale
        If True, profile over the multiplicative spectral-error rescaling
        parameter logNerror. If False, use the quoted spectral errors directly
        and remove logNerror from the nuisance-parameter fit.

    Returns
    -------
    like
        Likelihood callable. Its nuisance-parameter signature depends on
        float_error_scale.

    seed0
        Dictionary containing the initial nuisance-parameter values.

    nuisance_limits
        Dictionary containing the Minuit bounds for the nuisance parameters.
    """

    wavelength, flux, error = (
        validate_likelihood_data(
            wavelength,
            flux,
            error,
        )
    )

    bin_edges = (
        build_dimensionless_bin_edges(
            wavelength,
            mass,
            forward_model.hc,
        )
    )

    line_width = (
        forward_model.line_FWHM(
            mass,
            instrument_index,
            **velocity_parameters,
        )
    )

    kernel_metric = (
        3.0 * line_width
    ) ** 2

    reference_line = (
        forward_model.forward_model(
            bin_edges,
            galactic_l,
            galactic_b,
            reference_coupling,
            mass,
            instrument_index,
            **velocity_parameters,
        )
    )

    reference_line = np.asarray(
        reference_line,
        dtype=np.float64,
    )

    if (
        reference_line.shape
        != flux.shape
    ):
        raise RuntimeError(
            "The reference-line shape does not match "
            "the fitted flux:\n"
            f"  reference line: "
            f"{reference_line.shape}\n"
            f"  flux:           "
            f"{flux.shape}"
        )


    ################################
    ###   Floating Error Scale   ###
    ################################

    if float_error_scale:

        def like(
            logamp,
            logNerror,
            gagg2,
        ):
            signal_model = (
                gagg2
                * reference_line
            )

            residual = (
                flux
                - signal_model
            )

            flux_mean = np.mean(
                residual
            )

            kernel = (
                10.0**logamp
                * kernels.ExpSquaredKernel(
                    metric=kernel_metric
                )
            )

            gp = george.GP(
                kernel,
                mean=flux_mean,
            )

            try:
                gp.compute(
                    wavelength,
                    error
                    * 10.0**logNerror,
                )

                return (
                    -2.0
                    * gp.lnlikelihood(
                        residual
                    )
                )

            except (
                ValueError,
                LinAlgError,
            ):
                return 1e30


    ################################
    ###   Fixed Error Scale      ###
    ################################

    else:

        def like(
            logamp,
            gagg2,
        ):
            signal_model = (
                gagg2
                * reference_line
            )

            residual = (
                flux
                - signal_model
            )

            flux_mean = np.mean(
                residual
            )

            kernel = (
                10.0**logamp
                * kernels.ExpSquaredKernel(
                    metric=kernel_metric
                )
            )

            gp = george.GP(
                kernel,
                mean=flux_mean,
            )

            try:
                gp.compute(
                    wavelength,
                    error,
                )

                return (
                    -2.0
                    * gp.lnlikelihood(
                        residual
                    )
                )

            except (
                ValueError,
                LinAlgError,
            ):
                return 1e30


    ################################
    ###   Initial Nuisance Fit   ###
    ################################

    nuisance_limits = (
        get_nuisance_limits(
            flux,
            float_error_scale=(
                float_error_scale
            ),
        )
    )

    seed0 = (
        find_initial_nuisance_seed(
            like,
            nuisance_limits,
        )
    )

    return (
        like,
        seed0,
        nuisance_limits,
    )