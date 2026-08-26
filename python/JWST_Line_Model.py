from pathlib import Path

import mwdust
import numpy as np

from astropy import units as u
from astropy.io import fits
from dust_extinction.parameter_averages import F99, G23
from scipy.ndimage import gaussian_filter1d
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

    def __init__(self, IRFs_file_loc, N_g=2):

        self.IRFs_file_loc = Path(IRFs_file_loc)
        self.N_g = N_g
        self.hc = HC

        self.set_NFW_params()

        # Line-of-sight integration grid [kpc].
        #self.integration_radii = np.geomspace(1e-4, self.r200, 1_000_001)
        self.integration_radii = np.geomspace(1e-4, self.r200, 10_001)

        # Solar velocity in Galactic UVW coordinates.
        self.v_Sun = np.array([11.0, 247.0, 8.0]) / SPEED_OF_LIGHT
        self.DS_Bool = True

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

        self.r_s = r_s
        self.rho_E = rho_E
        self.r_E = r_E

        self.rho_0 = rho_E / self.NFW_norm_1(r_E)

        A = self.rho_0 / (200.0 * RHO_CRITICAL)
        root = 2.0 + 27.0 * A + 3.0 * np.sqrt(3.0) * np.sqrt(A * (4.0 + 27.0 * A))

        x200 = (
            -2.0
            + 2.0**(1.0 / 3.0) / root**(1.0 / 3.0)
            + root**(1.0 / 3.0) / 2.0**(1.0 / 3.0)
        ) / 3.0

        self.r200 = r_s * x200


    def NFW_norm_1(self, r):

        x = np.asarray(r) / self.r_s
        return 1.0 / (x * (1.0 + x)**2)


    def NFW(self, r):

        return self.rho_0 * self.NFW_norm_1(r)


    def r_GC(self, d_E, l, b):

        return np.sqrt(
            (d_E * np.cos(b) * np.cos(l) - self.r_E)**2
            + (d_E * np.cos(b) * np.sin(l))**2
            + (d_E * np.sin(b))**2
        )


    ################################
    ###   Velocity Profile       ###
    ################################

    def Doppler_Shift_On(self, DS_Bool):

        self.DS_Bool = bool(DS_Bool)


    def velocity_profile(self, ml_arr, l, b, v_0):
        """
        Standard Halo Model:

            f(v_los) ∝ exp(-v_los^2 / v_0^2)

        so sigma_v = v_0 / sqrt(2).
        """

        l = np.deg2rad(l)
        b = np.deg2rad(b)

        n_hat = np.array([
            np.cos(b) * np.cos(l),
            np.cos(b) * np.sin(l),
            np.sin(b),
        ])

        doppler_factor = 1.0 + self.DS_Bool * np.dot(n_hat, self.v_Sun)
        ml_arr = np.asarray(ml_arr) * doppler_factor

        ml_lo = ml_arr[:-1]
        ml_hi = ml_arr[1:]
        dml = np.diff(ml_arr)

        v_lo = 1.0 - 2.0 / ml_lo
        v_hi = 1.0 - 2.0 / ml_hi

        return (erf(v_hi / v_0) - erf(v_lo / v_0)) / (2.0 * dml)


    ################################
    ###   Dust Extinction        ###
    ################################

    def set_ext_map(self, ext_map_str="Drimmel"):

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

        name = extinction_law.lower()

        if name == "powerlaw":
            self.extinction_law = "PowerLaw"
        elif name == "g23":
            self.extinction_law = "G23"
        elif name in ["none", "nodust", "no_dust"]:
            self.extinction_law = "None"
        else:
            raise ValueError(
                "extinction_law must be 'PowerLaw', 'G23', or 'None'."
            )


    def compute_AlAV(self, wl):

        wl = np.asarray(wl, dtype=float)

        if self.extinction_law == "PowerLaw":
            return self.AlAV_at_0p75_um * (wl / 0.75)**-1.85

        if self.extinction_law == "G23":
            return np.asarray(self.g23(wl * u.micron), dtype=float)

        if self.extinction_law == "None":
            return np.zeros_like(wl, dtype=float)

        raise RuntimeError("Extinction law has not been set.")


    def compute_extinction_factor(self, l, b, d, wl):

        if self.extinction_law == "None":
            return np.zeros_like(np.asarray(d), dtype=float)

        return self.ext_map(l, b, np.asarray(d)) * self.compute_AlAV(wl)


    ################################
    ###   Wavelength Conversion  ###
    ################################

    def ml2wl(self, ml_arr, ma):

        return np.asarray(ml_arr) * self.hc / ma


    ################################
    ###   Effective D-Factor     ###
    ################################

    def dDeffdml(self, ml_arr, l, b, ma, v_0):
        """
        Compute the extinction-weighted differential D-factor.

        Extinction is evaluated at the central axion wavelength because
        the line is narrow compared with the wavelength scale over which
        the extinction law varies.
        """

        velocity_profile = self.velocity_profile(ml_arr, l, b, v_0)

        radii = self.integration_radii
        l_rad = np.deg2rad(l)
        b_rad = np.deg2rad(b)

        galactocentric_radii = self.r_GC(radii, l_rad, b_rad)
        density = self.NFW(galactocentric_radii)

        wl_axion = 2.0 * self.hc / ma
        A_lambda = self.compute_extinction_factor(l, b, radii, wl_axion)
        attenuation = 10.0**(-0.4 * A_lambda)

        effective_D = np.trapezoid(density * attenuation, radii)
        effective_D *= GEV_TO_EV * CM_PER_KPC

        return velocity_profile * effective_D


    ################################
    ###   Axion Decay Spectrum   ###
    ################################

    def compute_decay_spectrum(self, ml_arr, l, b, gagg, ma, v_0):

        ml_arr = np.asarray(ml_arr)
        ml_centers = 0.5 * (ml_arr[:-1] + ml_arr[1:])
        wl_centers = self.ml2wl(ml_centers, ma)

        # |d lambda / d nu| [micron/Hz].
        dldv = 3.33564e-15 * wl_centers**2

        # Decay rate [s^-1] for gagg [GeV^-1] and ma [eV].
        decay_rate = 0.00151927 * gagg**2 * ma**3 / (64.0 * np.pi)

        dDeffdml = self.dDeffdml(ml_arr, l, b, ma, v_0)

        prefactor = self.N_g * decay_rate / (4.0 * np.pi * wl_centers)
        luminosity_per_solid_angle = dldv * prefactor

        flux = EV_TO_ERG * luminosity_per_solid_angle * dDeffdml

        # erg/cm^2/s/sr/Hz -> MJy/sr.
        return 1e17 * flux


    ################################
    ###   Instrument Setup        ###
    ################################

    def set_instrument(self, instrument_str):

        instrument = instrument_str.upper()

        if instrument == "NIRSPEC":
            self.inst = "NIRSpec"
        elif instrument == "MIRI":
            self.inst = "MIRI"
        else:
            raise ValueError("instrument_str must be 'NIRSpec' or 'MIRI'.")


    def set_gratings(self):

        if self.inst != "NIRSpec":
            raise RuntimeError("set_gratings() is only used for NIRSpec.")

        grating_names = [
            "G140H", "G140M",
            "G235H", "G235M",
            "G395H", "G395M",
            "PRISM",
        ]

        self.gratings = {}

        for grating in grating_names:

            path = (
                self.IRFs_file_loc
                / self.inst
                / f"jwst_nirspec_{grating.lower()}_disp.fits"
            )

            with fits.open(path) as f:
                wavelength = np.asarray(f[1].data["WAVELENGTH"])
                resolving_power = np.asarray(f[1].data["R"])

            self.gratings[grating] = (wavelength, resolving_power)


    ################################
    ###   Instrument Resolution  ###
    ################################

    def compute_Gaussian_width(self, wl, gr):

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

    def forward_model(self, ml_arr, l, b, gagg, ma, gr, v_0):

        input_spectrum = self.compute_decay_spectrum(ml_arr, l, b, gagg, ma, v_0)

        ml_arr = np.asarray(ml_arr)
        ml_centers = 0.5 * (ml_arr[:-1] + ml_arr[1:])
        wl_centers = self.ml2wl(ml_centers, ma)

        wl_axion = 2.0 * self.hc / ma
        resolving_power = self.compute_Gaussian_width(wl_axion, gr)

        instrumental_FWHM = wl_axion / resolving_power
        instrumental_sigma = instrumental_FWHM / GAUSSIAN_FWHM
        sigma_pixels = instrumental_sigma / np.diff(wl_centers)[0]

        return gaussian_filter1d(input_spectrum, sigma_pixels)


    ################################
    ###   Total Line Width       ###
    ################################

    def line_FWHM(self, ma, gr, v_0):
        """
        Approximate total observed FWHM.

        Since f(v_los) ∝ exp(-v_los^2 / v_0^2),

            sigma_v = v_0 / sqrt(2).
        """

        wl_axion = 2.0 * self.hc / ma
        resolving_power = self.compute_Gaussian_width(wl_axion, gr)

        instrumental_sigma = (wl_axion / resolving_power) / GAUSSIAN_FWHM
        doppler_sigma = wl_axion * v_0 / np.sqrt(2.0)

        total_sigma = np.sqrt(instrumental_sigma**2 + doppler_sigma**2)

        return GAUSSIAN_FWHM * total_sigma