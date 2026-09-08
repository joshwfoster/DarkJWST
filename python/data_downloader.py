"""
Download and reprocess public JWST IFU spectroscopy from MAST.

Given a MAST observation ID, this module downloads the raw (UNCAL)
exposures and reruns the three JWST calibration-pipeline stages with the
settings used in the analysis:

    Detector1Pipeline
        UNCAL -> RATE. Standard ramp fitting with default settings.

    Spec2Pipeline
        RATE -> CAL. Pixel-based background subtraction is skipped so
        that the diffuse sky signal is preserved in the calibrated data.

    Spec3Pipeline
        ASN -> combined S3D cube and X1D spectrum. The master-background
        step is skipped. For point-like (or unknown) sources, the
        extract_1d local background subtraction is enabled.

CRDS configuration
------------------
The JWST pipeline requires a CRDS reference-file cache. Either set the
``CRDS_PATH`` and ``CRDS_SERVER_URL`` environment variables before
running, or call ``configure_crds`` with a writable cache directory.

Example
-------
>>> from data_downloader import configure_crds, process_observation
>>> configure_crds("/path/to/crds_cache")
>>> process_observation(
...     "jw03131-o001_t001_nirspec_g140h-f100lp",
...     "/path/to/download_directory",
...     "NIRSPEC/IFU",
... )
"""

import glob
import os
import shutil
from pathlib import Path

import numpy as np
from astropy.io import fits
from astroquery.mast import Observations


################################
###   CRDS Configuration     ###
################################

CRDS_SERVER_URL = "https://jwst-crds.stsci.edu"


def configure_crds(crds_path, crds_server_url=CRDS_SERVER_URL):
    """
    Point the JWST pipeline at a CRDS reference-file cache.

    Parameters
    ----------
    crds_path : str or Path
        Writable directory used to cache CRDS reference files. Several
        tens of GB may be downloaded on first use.

    crds_server_url : str, optional
        CRDS server used to populate the cache.
    """

    crds_path = Path(crds_path)
    crds_path.mkdir(parents=True, exist_ok=True)

    os.environ["CRDS_PATH"] = str(crds_path)
    os.environ["CRDS_SERVER_URL"] = crds_server_url


################################
###   File Handling          ###
################################

def safe_move(src, dst):
    """Move ``src`` to ``dst``, resolving duplicate or partial downloads."""

    if os.path.exists(dst):
        if os.path.getsize(dst) == os.path.getsize(src):
            # Duplicate download: discard the source.
            os.remove(src)
            return dst

        # Partial or corrupt file: replace it.
        os.remove(dst)

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    return shutil.move(src, dst)


################################
###   MAST Queries           ###
################################

def query_products(candidate_obs_id, instrument):
    """
    Query MAST for the products belonging to one public observation.

    Parameters
    ----------
    candidate_obs_id : str
        MAST observation ID, e.g. ``jw03131-o001_t001_nirspec_g140h-f100lp``.

    instrument : str
        MAST instrument name, e.g. ``NIRSPEC/IFU`` or ``MIRI/IFU``.

    Returns
    -------
    products : astropy.table.Table
        Full product list for the observation.
    """

    observations = Observations.query_criteria(
        obs_collection="JWST",
        instrument_name=instrument,
        provenance_name="CALJWST",
        dataRights="PUBLIC",
        obs_id=candidate_obs_id,
        intentType="science",
    )

    if len(observations) == 0:
        raise RuntimeError(
            f"No public JWST observations found for obs_id '{candidate_obs_id}' "
            f"with instrument '{instrument}'."
        )

    return Observations.get_product_list(observations)


def download_uncal_files(products, directory, folder):
    """
    Download the raw UNCAL exposures and collect them in one folder.

    Parameters
    ----------
    products : astropy.table.Table
        Product list returned by ``query_products``.

    directory : str
        Base download directory passed to astroquery.

    folder : str
        Destination folder for the collected UNCAL files.

    Returns
    -------
    obs_ids : ndarray of str
        Exposure-level observation IDs of the downloaded UNCAL files.
    """

    uncal_products = Observations.filter_products(
        products,
        productType="SCIENCE",
        productSubGroupDescription="UNCAL",
        calib_level=[1],
    )

    if len(uncal_products) == 0:
        raise RuntimeError("No UNCAL products found for this observation.")

    Observations.download_products(
        uncal_products,
        mrp_only=False,
        download_dir=directory,
    )

    os.makedirs(folder, exist_ok=True)

    # Astroquery downloads each exposure into its own subdirectory:
    # collect everything in a single folder.
    for exposure_id in uncal_products["obs_id"]:
        source = os.path.join(
            directory, "mastDownload/JWST", exposure_id, f"{exposure_id}_uncal.fits"
        )
        safe_move(source, os.path.join(folder, f"{exposure_id}_uncal.fits"))

        source_dir = os.path.dirname(source)
        if os.path.isdir(source_dir) and not os.listdir(source_dir):
            os.rmdir(source_dir)

    files = glob.glob(os.path.join(folder, "*_uncal.fits"))
    filenames = [os.path.basename(f) for f in files]

    return np.array([f.removesuffix("_uncal.fits") for f in filenames])


def download_asn_file(products, directory):
    """
    Download the level-3 association (ASN) file for the observation.

    Parameters
    ----------
    products : astropy.table.Table
        Product list returned by ``query_products``.

    directory : str
        Base download directory passed to astroquery.

    Returns
    -------
    asn_filename : str
        Filename of the downloaded association file.
    """

    asn_products = Observations.filter_products(
        products,
        productSubGroupDescription="ASN",
        calib_level=[3],
    )

    if len(asn_products) == 0:
        raise RuntimeError("No level-3 ASN products found for this observation.")

    Observations.download_products(
        asn_products,
        mrp_only=False,
        download_dir=directory,
    )

    return asn_products["productFilename"][0]


################################
###   Pipeline Stages        ###
################################

def run_detector1(folder, obs_ids):
    """Run the Detector1 pipeline (UNCAL -> RATE) on each exposure."""

    from jwst.pipeline import Detector1Pipeline

    for obs_id in obs_ids:
        if os.path.exists(os.path.join(folder, f"{obs_id}_rate.fits")):
            print(f"{obs_id}: Detector1 output found, skipping...")
            continue

        Detector1Pipeline.call(
            os.path.join(folder, f"{obs_id}_uncal.fits"),
            save_results=True,
            output_dir=folder,
        )

        # Remove unneeded per-integration output.
        rateints = os.path.join(folder, f"{obs_id}_rateints.fits")
        if os.path.exists(rateints):
            os.remove(rateints)


def run_spec2(folder, obs_ids):
    """
    Run the Spec2 pipeline (RATE -> CAL) on each exposure.

    Pixel-based background subtraction is skipped so that the diffuse
    sky emission is preserved in the calibrated data.
    """

    from jwst.pipeline import Spec2Pipeline

    for obs_id in obs_ids:
        if os.path.exists(os.path.join(folder, f"{obs_id}_cal.fits")):
            print(f"{obs_id}: Spec2 output found, skipping...")
            continue

        Spec2Pipeline.call(
            os.path.join(folder, f"{obs_id}_rate.fits"),
            save_results=True,
            steps={"bkg_subtract": {"skip": True}},
            output_dir=folder,
        )

        # Remove unneeded per-exposure level-2 products.
        for suffix in ("x1d", "s3d"):
            path = os.path.join(folder, f"{obs_id}_{suffix}.fits")
            if os.path.exists(path):
                os.remove(path)


def read_source_type(folder, obs_ids):
    """Return the proposal source type (SRCTYAPT) from the first CAL file."""

    cal_file = os.path.join(folder, f"{obs_ids[0]}_cal.fits")

    with fits.open(cal_file) as hdul:
        return hdul[0].header["SRCTYAPT"]


def run_spec3(folder, asn_filename, source_type):
    """
    Run the Spec3 pipeline on the association file.

    The master-background step is skipped. For point-like or unknown
    sources, the extract_1d local background subtraction is enabled.
    """

    from jwst.pipeline import Spec3Pipeline

    subtract_background = source_type in ("POINT")

    Spec3Pipeline.call(
        os.path.join(folder, asn_filename),
        steps={
            "master_background": {"skip": True},
            "extract_1d": {"subtract_background": subtract_background},
        },
        save_results=True,
        output_dir=folder,
    )


################################
###   Cleanup                ###
################################

def cleanup_intermediates(folder, obs_ids):
    """Remove per-exposure intermediate files, keeping the level-3 outputs."""

    for obs_id in obs_ids:
        for suffix in ("uncal", "rate", "cal", "o001_crf"):
            path = Path(folder) / f"{obs_id}_{suffix}.fits"
            path.unlink(missing_ok=True)


################################
###   Main Entry Point       ###
################################

def process_observation(candidate_obs_id, directory, instrument,
                        cleanup=True):
    """
    Download and reprocess one public JWST observation.

    Parameters
    ----------
    candidate_obs_id : str
        MAST observation ID, e.g. ``jw03131-o001_t001_nirspec_g140h-f100lp``.

    directory : str
        Base directory used for downloads and pipeline products. The
        final products are written to
        ``directory/mastDownload/JWST/candidate_obs_id/``.

    instrument : str
        MAST instrument name, e.g. ``NIRSPEC/IFU`` or ``MIRI/IFU``.

    cleanup : bool, optional
        Remove per-exposure intermediate files after the level-3
        pipeline completes, keeping only the combined products.

    Returns
    -------
    folder : str
        Folder containing the final level-3 products.
    """

    folder = os.path.join(directory, "mastDownload/JWST", candidate_obs_id, "")

    products = query_products(candidate_obs_id, instrument)

    obs_ids = download_uncal_files(products, directory, folder)

    run_detector1(folder, obs_ids)
    run_spec2(folder, obs_ids)

    source_type = read_source_type(folder, obs_ids)

    asn_filename = download_asn_file(products, directory)
    run_spec3(folder, asn_filename, source_type)

    if cleanup:
        cleanup_intermediates(folder, obs_ids)

    return folder
