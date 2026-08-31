"""
Forward model for Galactic axion-decay lines observed with JWST.
"""

from pathlib import Path

import mwdust
import numpy as np

from astropy import units as u
from astropy.io import fits
from dust_extinction.parameter_averages import F99, G23
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import erf


################################
###   Physical Constants     ###
################################

SPEED_OF_LIGHT = 299_792.0  # km/s
CM_PER_KPC = 3.086e21
GEV_TO_EV = 1e9
EV_TO_ERG = 1.60218e-12

HC = 1.23984  # eV micron
RHO_CRITICAL = 4.8e-6  # GeV/cm^3
GAUSSIAN_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


################################
###   Line Forward Model     ###
################################

class Line_Forward_Model:
    """Forward model for a Galactic axion-decay line."""

    def __init__(self, IRFs_file_loc, N_g=2):

        self.IRFs_file_loc = Path(IRFs_file_loc)
        self.N_g = N_g
        self.hc = HC

        self.set_NFW_params()

        # Solar velocity in Galactic UVW coordinates.
        self.v_Sun = np.array([11.0, 247.0, 8.0]) / SPEED_OF_LIGHT

        # Extinction laws.
        self.f99 = F99(Rv=3.1)
        self.g23 = G23(Rv=3.1)
        self.AlAV_at_0p75_um = self.f99(0.75 * u.micron)

        self.extinction_law = None
        self.set_extinction_law("PowerLaw")

        # Dust maps.
        self._ext_maps = {}
        self.ext_map = None
        self.ext_map_name = None
        self.set_ext_map("Drimmel")

        # Instrument response.
        self.inst = None
        self.gratings = None


    ################################
    ###   NFW Halo               ###
    ################################

    def set_NFW_params(self, r_s=24.0, rho_E=0.30375, r_E=8.0):
        """Set the NFW halo parameters and determine r200."""

        self.r_s = r_s
        self.rho_E = rho_E
        self.r_E = r_E
        self.rho_s = rho_E / self.NFW_norm_1(r_E)

        def overdensity_equation(c):
            mass_norm = np.log1p(c) - c / (1.0 + c)
            return 3.0 * self.rho_s * mass_norm / c**3 - 200.0 * RHO_CRITICAL

        self.c200 = brentq(overdensity_equation, 1.0, 100.0)
        self.r200 = self.c200 * self.r_s


    def NFW_norm_1(self, r):
        """Dimensionless NFW density profile."""

        x = np.asarray(r) / self.r_s
        return 1.0 / (x * (1.0 + x)**2)


    def NFW(self, r):
        """NFW density in GeV/cm^3."""

        return self.rho_s * self.NFW_norm_1(r)


    def r_GC(self, d_E, l, b):
        """Galactocentric radius for heliocentric distance d_E [kpc]."""

        return np.sqrt(
            (d_E * np.cos(b) * np.cos(l) - self.r_E)**2
            + (d_E * np.cos(b) * np.sin(l))**2
            + (d_E * np.sin(b))**2
        )


    def max_los_distance(self, l, b):
        """Distance [kpc] at which a line of sight exits r200."""

        l = np.deg2rad(l)
        b = np.deg2rad(b)
        mu = np.cos(b) * np.cos(l)

        return self.r_E * mu + np.sqrt(self.r200**2 - self.r_E**2 * (1.0 - mu**2))


    ################################
    ###   Velocity Profile       ###
    ################################

    def velocity_profile(self, ml_arr, l, b, v_0):
        """
        Standard Halo Model line profile.

        f(v_los) ∝ exp(-v_los^2 / v_0^2), so sigma_v = v_0 / sqrt(2).
        """

        l = np.deg2rad(l)
        b = np.deg2rad(b)

        n_hat = np.array([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)])

        ml_arr = np.asarray(ml_arr)
        dml = np.diff(ml_arr)

        doppler_factor = 1.0 + np.dot(n_hat, self.v_Sun)
        ml_shifted = ml_arr * doppler_factor

        v_lo = 1.0 - 2.0 / ml_shifted[:-1]
        v_hi = 1.0 - 2.0 / ml_shifted[1:]

        return (erf(v_hi / v_0) - erf(v_lo / v_0)) / (2.0 * dml)


    ################################
    ###   Dust Extinction        ###
    ################################

    def set_ext_map(self, ext_map_str="Drimmel"):
        """Set the Galactic dust map."""

        name = ext_map_str.lower()

        if name == "drimmel":
            canonical_name = "Drimmel"
            constructor = mwdust.Drimmel03
        elif name == "sfd":
            canonical_name = "sfd"
            constructor = mwdust.SFD
        elif name == "combined":
            canonical_name = "Combined"
            constructor = mwdust.Combined19
        else:
            raise ValueError("ext_map_str must be 'Drimmel', 'sfd', or 'Combined'.")

        if canonical_name not in self._ext_maps:
            self._ext_maps[canonical_name] = constructor(filter="Landolt V")

        self.ext_map = self._ext_maps[canonical_name]
        self.ext_map_name = canonical_name


    def set_extinction_law(self, extinction_law="PowerLaw"):
        """Set the wavelength-dependent extinction law."""

        name = extinction_law.lower()

        if name == "powerlaw":
            self.extinction_law = "PowerLaw"
        elif name == "g23":
            self.extinction_law = "G23"
        elif name in ["none", "nodust", "no_dust"]:
            self.extinction_law = "None"
        else:
            raise ValueError("extinction_law must be 'PowerLaw', 'G23', or 'None'.")


    def compute_AlAV(self, wl):
        """Return A_lambda/A_V for wavelength wl [micron]."""

        wl = np.asarray(wl, dtype=float)

        if self.extinction_law == "PowerLaw":
            return self.AlAV_at_0p75_um * (wl / 0.75)**-1.85

        if self.extinction_law == "G23":
            return np.asarray(self.g23(wl * u.micron), dtype=float)

        if self.extinction_law == "None":
            return np.zeros_like(wl, dtype=float)

        raise RuntimeError("Extinction law has not been set.")


    def compute_A_lambda(self, l, b, d, wl):
        """Return A_lambda along a line of sight at distance d [kpc]."""

        scalar_input = np.ndim(d) == 0
        d = np.atleast_1d(d).astype(float)

        if self.extinction_law == "None":
            result = np.zeros_like(d)
        else:
            result = self.ext_map(l, b, d) * self.compute_AlAV(wl)

        return result[0] if scalar_input else result


    ################################
    ###   Wavelength Conversion  ###
    ################################

    def ml2wl(self, ml_arr, ma):
        """Convert dimensionless m_a lambda to wavelength [micron]."""

        return np.asarray(ml_arr) * self.hc / ma


    ################################
    ###   Effective D-Factor     ###
    ################################

    def dDeffdml(self, ml_arr, l, b, ma, v_0, method="quad", **integration_kwargs):
        """Return the differential effective D-factor."""

        velocity_profile = self.velocity_profile(ml_arr, l, b, v_0)
        effective_D = self.effective_D_factor(l, b, ma, method=method, **integration_kwargs)

        return velocity_profile * effective_D


    def effective_D_factor(self, l, b, ma, method="quad", n_grid=10_001, softening=None):
        """
        Compute the extinction-weighted decay D-factor.

        ``quad`` uses adaptive quadrature. ``grid`` uses a fixed line-of-sight
        grid with a softened NFW cusp. The result is returned in eV/cm^2.
        """

        if method == "quad":
            return self._effective_D_quad(l, b, ma)

        if method == "grid":
            return self._effective_D_grid(l, b, ma, n_grid, softening)

        raise ValueError("method must be 'quad' or 'grid'.")


    def _effective_D_quad(self, l, b, ma):

        s_max = self.max_los_distance(l, b)
        l_rad = np.deg2rad(l)
        b_rad = np.deg2rad(b)
        wl_axion = 2.0 * self.hc / ma

        def integrand(s):
            density = self.NFW(self.r_GC(s, l_rad, b_rad))
            A_lambda = self.compute_A_lambda(l, b, s, wl_axion)
            return float(density * 10.0**(-0.4 * A_lambda))

        s_closest = self.r_E * np.cos(b_rad) * np.cos(l_rad)
        points = [s_closest] if 0.0 < s_closest < s_max else None

        effective_D = quad(
            integrand, 0.0, s_max, points=points,
            epsabs=0.0, epsrel=1e-6
        )[0]

        return effective_D * GEV_TO_EV * CM_PER_KPC


    def _effective_D_grid(self, l, b, ma, n_grid, softening):

        s_max = self.max_los_distance(l, b)
        distances = np.linspace(0.0, s_max, n_grid)

        l_rad = np.deg2rad(l)
        b_rad = np.deg2rad(b)
        galactocentric_radii = self.r_GC(distances, l_rad, b_rad)

        if softening is None:
            softening = distances[1] - distances[0]

        softened_radii = np.sqrt(galactocentric_radii**2 + softening**2)
        density = self.NFW(softened_radii)

        wl_axion = 2.0 * self.hc / ma
        A_lambda = self.compute_A_lambda(l, b, distances, wl_axion)
        attenuation = 10.0**(-0.4 * A_lambda)

        effective_D = np.trapezoid(density * attenuation, distances)

        return effective_D * GEV_TO_EV * CM_PER_KPC


    ################################
    ###   Axion Decay Spectrum   ###
    ################################

    def compute_decay_spectrum(self, ml_arr, l, b, gagg, ma, v_0,
                               integration_method="quad", **integration_kwargs):
        """Compute the unconvolved axion-decay spectrum in MJy/sr."""

        ml_arr = np.asarray(ml_arr)
        ml_centers = 0.5 * (ml_arr[:-1] + ml_arr[1:])
        wl_centers = self.ml2wl(ml_centers, ma)

        # |d lambda / d nu| [micron/Hz].
        dldnu = 3.33564e-15 * wl_centers**2

        # Decay rate [s^-1] for gagg [GeV^-1] and ma [eV].
        decay_rate = 0.00151927 * gagg**2 * ma**3 / (64.0 * np.pi)

        dDeffdml = self.dDeffdml(
            ml_arr, l, b, ma, v_0,
            method=integration_method, **integration_kwargs
        )

        prefactor = self.N_g * decay_rate / (4.0 * np.pi * wl_centers)
        luminosity_per_solid_angle = dldnu * prefactor
        flux = EV_TO_ERG * luminosity_per_solid_angle * dDeffdml

        # erg/cm^2/s/sr/Hz -> MJy/sr.
        return 1e17 * flux


    ################################
    ###   Instrument Setup        ###
    ################################

    def set_instrument(self, instrument_str):
        """Set the JWST instrument."""

        instrument = instrument_str.upper()

        if instrument == "NIRSPEC":
            self.inst = "NIRSpec"
        elif instrument == "MIRI":
            self.inst = "MIRI"
        else:
            raise ValueError("instrument_str must be 'NIRSpec' or 'MIRI'.")


    def set_gratings(self):
        """Load the NIRSpec wavelength-dependent resolving powers."""

        if self.inst != "NIRSpec":
            raise RuntimeError("set_gratings() is only used for NIRSpec.")

        grating_names = [
            "G140H", "G140M", "G235H", "G235M",
            "G395H", "G395M", "PRISM",
        ]

        self.gratings = {}

        for grating in grating_names:
            path = self.IRFs_file_loc / self.inst / f"jwst_nirspec_{grating.lower()}_disp.fits"

            with fits.open(path) as f:
                wavelength = np.asarray(f[1].data["WAVELENGTH"])
                resolving_power = np.asarray(f[1].data["R"])

            self.gratings[grating] = (wavelength, resolving_power)


    ################################
    ###   Instrument Resolution  ###
    ################################

    def compute_resolving_power(self, wl, gr):
        """Return the instrumental resolving power at wavelength wl [micron]."""

        if self.inst == "NIRSpec":
            if self.gratings is None:
                raise RuntimeError("Call set_gratings() first.")

            wavelength, resolving_power = self.gratings[gr]
            return np.interp(wl, wavelength, resolving_power)

        if self.inst == "MIRI":
            return 4603.0 - 128.0 * wl

        raise RuntimeError("Call set_instrument() first.")


    ################################
    ###   Instrument Convolution ###
    ################################

    def convolve_instrument_response(self, spectrum, ml_arr, ma, gr):
        """Convolve a spectrum with the Gaussian instrumental response."""

        wl_edges = self.ml2wl(ml_arr, ma)
        wl_centers = 0.5 * (wl_edges[:-1] + wl_edges[1:])

        wl_axion = 2.0 * self.hc / ma
        resolving_power = self.compute_resolving_power(wl_axion, gr)

        instrumental_FWHM = wl_axion / resolving_power
        instrumental_sigma = instrumental_FWHM / GAUSSIAN_FWHM

        z = (wl_edges[None, :] - wl_centers[:, None]) / (
            np.sqrt(2.0) * instrumental_sigma
        )
        weights = 0.5 * (erf(z[:, 1:]) - erf(z[:, :-1]))

        return weights @ spectrum


    def forward_model(self, ml_arr, l, b, gagg, ma, gr, v_0,
                      integration_method="quad", **integration_kwargs):
        """Compute the instrument-convolved axion-decay spectrum."""

        input_spectrum = self.compute_decay_spectrum(
            ml_arr, l, b, gagg, ma, v_0,
            integration_method=integration_method, **integration_kwargs
        )

        return self.convolve_instrument_response(input_spectrum, ml_arr, ma, gr)


    ################################
    ###   Total Line Width       ###
    ################################

    def line_FWHM(self, ma, gr, v_0):
        """
        Approximate the total observed line FWHM.

        Since f(v_los) ∝ exp(-v_los^2 / v_0^2), sigma_v = v_0 / sqrt(2).
        """

        wl_axion = 2.0 * self.hc / ma
        resolving_power = self.compute_resolving_power(wl_axion, gr)

        instrumental_sigma = (wl_axion / resolving_power) / GAUSSIAN_FWHM
        doppler_sigma = wl_axion * v_0 / np.sqrt(2.0)
        total_sigma = np.sqrt(instrumental_sigma**2 + doppler_sigma**2)

        return GAUSSIAN_FWHM * total_sigma