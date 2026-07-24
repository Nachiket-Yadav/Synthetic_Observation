"""
analysis_skirt.py
==================

SKIRT-variant counterpart to ``analysis.py``. Fits the SKIRT-variant
observed images (``*_sim_observed_pbcor_SKIRT.fits``, written by
``casa_simulation_skirt.py``) with the exact same ``imfit``-based procedure
and dust-mass physics as ``analysis.py``, but writes to a **separate**
results file, ``fitting_results_skirt.json``, keyed identically to
``fitting_results.json`` (``{ snapshot: { region: { axis: {...} } } }``).

Kept as a separate file rather than a flag on ``analysis.py`` so the thin
pipeline is untouched -- in particular, an existing ``fitting_results.json``
can never be overwritten with SKIRT-derived numbers under the same
snapshot/region/axis keys just because a SKIRT pbcor image happened to be
sitting in the same ``pbcor_imgs/`` folder.

Like ``analysis.py``, the ``imfit``/``imhead``/``imstat`` calls mean this
must be run inside CASA:

    casa --nogui --nologger -c analysis_skirt.py \\
        --image-dir pbcor_imgs --results fitting_results_skirt.json \\
        --luminosity-file luminosities.json

``--luminosity-file`` is shared with the thin pipeline on purpose: the
protostellar luminosity used to set the dust temperature is a property of
the underlying simulation snapshot, not of which skymodel-generation method
produced the sky model.
"""

import os
import glob
import json
import argparse

import numpy as np


# ---------------------------------------------------------------------------
# Physical constants (CGS) -- identical to analysis.py.
# ---------------------------------------------------------------------------
H_PLANCK = 6.62607015e-27        # erg s
C_LIGHT = 2.99792458e10          # cm/s
K_BOLTZ = 1.380649e-16           # erg/K
NU_HZ = 3.38e11                  # 338 GHz observing frequency
KAPPA_NU = 1.84                  # cm^2/g dust opacity at 338 GHz
SOLAR_MASS_G = 1.989e33          # g
JY_TO_CGS = 1e-23                # 1 Jy = 1e-23 erg/s/cm^2/Hz

# Region distances in centimetres (Orion ~400 pc, Perseus ~300 pc).
REGION_DISTANCE_CM = {
    "Orion": 1.234e21,
    "Perseus": 9.257e20,
}
REGION_DISTANCE_PC = {
    "Orion": 400.0,
    "Perseus": 300.0,
}

T0_DUST_K = 43.0                 # reference dust temperature for L-scaling

SKIRT_SUFFIX = "_SKIRT"


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def load_dict(path):
    """Load a JSON dict from ``path``, returning {} if it does not exist."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    print("No file found at %s, starting fresh." % path)
    return {}


def save_dict(d, path):
    """Write ``d`` to ``path`` as indented JSON."""
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


# ---------------------------------------------------------------------------
# Physics -- identical to analysis.py.
# ---------------------------------------------------------------------------
def planck_bnu(temperature_k, nu_hz=NU_HZ):
    """Planck function B_nu(T) in erg/s/cm^2/Hz/sr."""
    x = (H_PLANCK * nu_hz) / (K_BOLTZ * temperature_k)
    return (2 * H_PLANCK * nu_hz**3 / C_LIGHT**2) / (np.exp(x) - 1)


def dust_mass_msun(flux_jy, distance_cm, dust_temp_k):
    """Optically thin dust mass in solar masses.

    M = F_nu * d^2 / (kappa_nu * B_nu(T))
    """
    flux_cgs = flux_jy * JY_TO_CGS                      # erg/s/cm^2/Hz
    bnu = planck_bnu(dust_temp_k)                        # erg/s/cm^2/Hz/sr
    mass_g = (flux_cgs * distance_cm**2) / (KAPPA_NU * bnu)
    return mass_g / SOLAR_MASS_G


def nan_result():
    """A measurement record full of NaNs (used when a fit fails)."""
    keys = ["Rmaj", "Rmaj_err", "Rmin", "Rmin_err", "pa", "pa_err",
            "flux", "flux_err", "flux_peak", "flux_peak_err", "inc",
            "radius_AU", "radius_AU_Tobin", "rms_residual", "peak_residual",
            "min_residual", "peak_residual_fraction", "snr"]
    return {k: float("nan") for k in keys}


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------
def fit_image(image_path, residual_dir, model_dir, fit_radius_frac=0.3):
    """Run imfit on one SKIRT image and return ``(snapshot, region, axis, record)``.

    Same as ``analysis.fit_image`` except the residual/model FITS exports
    keep the ``_SKIRT`` suffix so they land alongside (not on top of) the
    thin residual/model images in the shared ``residual_dir``/``model_dir``.
    """
    fname = os.path.basename(image_path)

    # Filename: ALMA_snapshot_<id>_axis_<axis>_<Region>_sim_observed_pbcor_SKIRT.fits
    # snapshot/axis/region sit at the same split() indices as the thin
    # filename -- the trailing _SKIRT suffix comes after all three, so this
    # positional parse (unlike the sky-model one in casa_simulation_skirt.py)
    # is unaffected by it.
    parts = fname.split("_")
    snapshot = parts[2]
    axis = parts[4]
    region = parts[5]

    # Image geometry -> central pixel and circular fit region.
    hdr = imhead(image_path)
    nx, ny = hdr["shape"][0], hdr["shape"][1]
    cx, cy = nx // 2, ny // 2
    fit_radius_pix = fit_radius_frac * nx
    region_string = "circle[[%dpix,%dpix],%fpix]" % (cx, cy, fit_radius_pix)

    print("  fitting region:", region_string)

    residual_image = "synthetic_disk_residual_skirt.image"
    model_image = "synthetic_disk_model_skirt.image"

    myfit = imfit(
        imagename=image_path,
        region=region_string,
        logfile="synthetic_disk_imfit_skirt.txt",
        residual=residual_image,
        model=model_image,
        overwrite=True,
    )

    # Residual statistics within the same region (for fit-quality metrics).
    residual_stats = imstat(imagename=residual_image, region=region_string)
    rms_residual = residual_stats["rms"][0]
    peak_residual = residual_stats["max"][0]
    min_residual = residual_stats["min"][0]

    # Export residual and model for inspection.
    if not os.path.isdir(residual_dir):
        os.makedirs(residual_dir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    # fname already ends in "..._pbcor_SKIRT" (extension stripped), so this
    # naturally produces "..._pbcor_SKIRT_residual.fits".
    exportfits(
        imagename=residual_image,
        fitsimage="%s/%s_residual.fits" % (residual_dir, fname.split(".")[0]),
        overwrite=True,
    )
    exportfits(
        imagename=model_image,
        fitsimage="%s/ALMA_snapshot_%s_axis_%s_%s_sim_observed_pbcor%s_model.fits"
                  % (model_dir, snapshot, axis, region, SKIRT_SUFFIX),
        overwrite=True,
    )

    try:
        comp = myfit["deconvolved"]["component0"]
        res = myfit["results"]["component0"]

        Rmaj = comp["shape"]["majoraxis"]["value"]
        Rmin = comp["shape"]["minoraxis"]["value"]
        Rmaj_err = comp["shape"]["majoraxiserror"]["value"]
        Rmin_err = comp["shape"]["minoraxiserror"]["value"]
        pa = comp["shape"]["positionangle"]["value"]
        pa_err = comp["shape"]["positionangleerror"]["value"]
        flux = comp["flux"]["value"][0]
        flux_err = comp["flux"]["error"][0]
        flux_peak = res["peak"]["value"]
        flux_peak_err = res["peak"]["error"]

        inc = np.degrees(np.arccos(Rmin / Rmaj))           # inclination, deg

        distance_pc = REGION_DISTANCE_PC.get(region, 400.0)
        radius_AU = Rmaj / 2.0 * distance_pc               # half-FWHM radius
        radius_AU_Tobin = Rmaj * 2.0 / 2.355 * distance_pc # Tobin-style radius

        peak_residual_fraction = abs(peak_residual) / flux_peak
        snr = flux_peak / rms_residual

        record = {
            "Rmaj": Rmaj, "Rmaj_err": Rmaj_err,
            "Rmin": Rmin, "Rmin_err": Rmin_err,
            "pa": pa, "pa_err": pa_err,
            "flux": flux, "flux_err": flux_err,
            "flux_peak": flux_peak, "flux_peak_err": flux_peak_err,
            "inc": inc,
            "radius_AU": radius_AU, "radius_AU_Tobin": radius_AU_Tobin,
            "rms_residual": rms_residual,
            "peak_residual": peak_residual,
            "min_residual": min_residual,
            "peak_residual_fraction": peak_residual_fraction,
            "snr": snr,
        }
    except Exception:
        print("  fit failed -> recording NaNs.")
        record = nan_result()

    return snapshot, region, axis, record


def add_dust_mass(snapshot, region, axis, master_dict, default_dust_temp):
    """Add a dust-mass estimate to an existing measurement record.

    Identical to ``analysis.add_dust_mass``.
    """
    record = master_dict[snapshot][region][axis]
    flux = record.get("flux")
    if flux is None or (isinstance(flux, float) and np.isnan(flux)):
        return  # nothing to do for failed fits

    distance_cm = REGION_DISTANCE_CM.get(region, REGION_DISTANCE_CM["Orion"])

    # Global (snapshot-level) luminosity-scaled temperature, if available.
    # A luminosity of 0 (no protostar yet) would give T = 0, which is
    # unphysical and divides by zero in the Planck function, so fall back to
    # the fixed default temperature in that case.
    snap_lum = master_dict[snapshot].get("total_luminosity")
    if snap_lum is not None and snap_lum > 0:
        dust_temp = T0_DUST_K * (snap_lum ** 0.25)
    else:
        dust_temp = default_dust_temp
    record["dust_mass_Msun"] = dust_mass_msun(flux, distance_cm, dust_temp)

    # Optional radial-luminosity-scaled temperature, if available.
    lum_radial = record.get("total_luminosity_radial")
    if lum_radial is not None and lum_radial > 0:
        dust_temp_radial = T0_DUST_K * (lum_radial ** 0.25)
        record["dust_mass_radial_Msun"] = dust_mass_msun(
            flux, distance_cm, dust_temp_radial)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fit SKIRT-variant observed images and predict disk dust masses."
    )
    parser.add_argument(
        "--image-dir", default="pbcor_imgs",
        help="Folder of *_sim_observed_pbcor_SKIRT.fits images (default: pbcor_imgs, "
             "the same folder the thin images live in).",
    )
    parser.add_argument(
        "--glob", default="*_SKIRT.fits",
        help="Glob used to select images within --image-dir (default: *_SKIRT.fits -- "
             "restricts to the SKIRT variant even when thin pbcor images live in the "
             "same folder).",
    )
    parser.add_argument(
        "--results", default="fitting_results_skirt.json",
        help="JSON file to read/write results (default: fitting_results_skirt.json, "
             "kept separate from the thin fitting_results.json).",
    )
    parser.add_argument(
        "--residual-dir", default="residual_imgs",
        help="Folder for exported residual FITS images.",
    )
    parser.add_argument(
        "--model-dir", default="model_imgs",
        help="Folder for exported model FITS images.",
    )
    parser.add_argument(
        "--fit-radius-frac", type=float, default=0.3,
        help="Fit-region radius as a fraction of image width (default: 0.3).",
    )
    parser.add_argument(
        "--dust-temp", type=float, default=43.0,
        help="Fallback dust temperature (K) when no luminosity is available.",
    )
    parser.add_argument(
        "--luminosity-file", default="luminosities.json",
        help="JSON of per-snapshot total luminosities written by "
             "skymodel_generation.py. Used to scale the dust temperature "
             "(T = 43 K * L^0.25). Shared with the thin pipeline -- "
             "luminosity is a property of the simulation snapshot, not of "
             "which skymodel variant was generated from it. Ignored if the "
             "file is missing.",
    )
    args, _unknown = parser.parse_known_args()

    images = sorted(glob.glob(os.path.join(args.image_dir, args.glob)))
    if not images:
        raise SystemExit("No SKIRT images found in %s matching %s" % (args.image_dir, args.glob))

    master_dict = load_dict(args.results)

    # Load per-snapshot luminosities (from stage 1) and merge them into the
    # results dict so add_dust_mass can pick them up. Keys are bare snapshot
    # IDs (e.g. "170"), matching how images are keyed below.
    luminosities = load_dict(args.luminosity_file)
    for snap_id, lum in luminosities.items():
        master_dict.setdefault(snap_id, {})["total_luminosity"] = lum
    if luminosities:
        print("Loaded luminosities for %d snapshot(s) from %s."
              % (len(luminosities), args.luminosity_file))

    print("Processing %d SKIRT image(s)." % len(images))

    for image_path in images:
        print("Image:", os.path.basename(image_path))
        snapshot, region, axis, record = fit_image(
            image_path, args.residual_dir, args.model_dir, args.fit_radius_frac)

        master_dict.setdefault(snapshot, {}).setdefault(region, {})[axis] = record
        add_dust_mass(snapshot, region, axis, master_dict, args.dust_temp)

        save_dict(master_dict, args.results)   # checkpoint after each image

    print("\nDone. Results written to %s" % args.results)


if __name__ == "__main__":
    main()
