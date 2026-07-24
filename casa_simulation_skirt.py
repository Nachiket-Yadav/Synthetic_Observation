"""
casa_simulation_skirt.py
=========================

SKIRT-variant counterpart to ``casa_simulation.py``. Runs the exact same
CASA pipeline (``simobserve`` -> ``tclean`` -> ``impbcor`` -> ``exportfits``)
with byte-identical observation/imaging parameters, but on sky models
produced by SKIRT Monte Carlo radiative transfer (dropped into the sky
model folder externally, already in the same FITS convention as the thin
sky models) instead of the yt-projection ("thin") sky models.

This is a separate file rather than a flag on ``casa_simulation.py`` on
purpose: the thin pipeline must stay untouched and behave exactly as before
for any existing invocation, and the two variants need to be run, resumed,
and debugged independently. The only intended difference between a thin and
a SKIRT run of the same snapshot/region/axis is the input sky model --
every ``OBS_SETTINGS`` value below is copied verbatim from
``casa_simulation.py``.

Naming convention
------------------
SKIRT sky models are named like the thin ones with a ``_SKIRT`` suffix
appended to the stem, immediately before the extension, e.g.::

    thin : skymodels/snapshot_170_Orion_flux_map_ALMA_axis_z.fits
    skirt: skymodels/snapshot_170_Orion_flux_map_ALMA_axis_z_SKIRT.fits

The exported pbcor image follows the same convention, in the *same*
``pbcor_imgs/`` folder as the thin images (they don't collide because of
the suffix)::

    pbcor_imgs/ALMA_snapshot_170_axis_z_Orion_sim_observed_pbcor_SKIRT.fits

Because the suffix sits at the very end of the sky-model filename, it lands
*after* the axis letter -- so the naive ``fname.split("_")[-1]`` trick used
in ``casa_simulation.py`` to pull out the axis (last token before the
extension) would read "SKIRT" instead of e.g. "z". ``parse_skirt_skymodel_filename``
below parses this properly with a regex instead of relying on that
position. (Filenames the *pbcor* stage parses -- axis at split index 4,
snapshot at index 2 -- are unaffected by a trailing suffix, since those
indices sit well before it; see SKIRT_PIPELINE_TASK.md for the full
positional-parsing audit.)

IMPORTANT -- this script must be run inside CASA, not plain Python, exactly
like ``casa_simulation.py``:

    casa --nogui --nologger -c casa_simulation_skirt.py \\
        --skymodel-dir skymodels --out-dir pbcor_imgs

    # or, from inside an interactive CASA session:
    CASA <1>: import sys
    CASA <2>: sys.argv = ['casa_simulation_skirt.py', '--skymodel-dir', 'skymodels',
                          '--out-dir', 'pbcor_imgs']
    CASA <3>: execfile('casa_simulation_skirt.py')
"""

import os
import re
import glob
import argparse


# ---------------------------------------------------------------------------
# Observation settings -- byte-identical to casa_simulation.py. If you change
# a value here, change it there too (or better, extract OBS_SETTINGS into a
# shared module) so the two variants stay comparable.
# ---------------------------------------------------------------------------
OBS_SETTINGS = {
    # simobserve
    "indirection": "J2000 03h29m10.0s +31d13m30s",  # pointing centre on sky
    "incenter": "338GHz",                           # central frequency
    "inwidth": "4.75GHz",                           # bandwidth
    "integration": "10s",                           # integration time per sample
    "totaltime": "30min",                           # total on-source time
    "antennalist": "alma.cycle5.6",                 # array configuration (no .cfg)
    "thermalnoise": "tsys-atm",                     # thermal-noise model

    # tclean
    "cell_tclean_arcsec": 0.03,                     # output image cell size
    "weighting": "natural",
    "niter": 1000,
    "deconvolver": "hogbom",
    "stokes": "I",

    # imsize selection: large image if the model pixel is coarse, else small.
    "npix_coarse": 720,                             # used when incell > 0.01 arcsec
    "npix_fine": 100,                               # used otherwise
    "incell_threshold_arcsec": 0.01,
}

SKIRT_SUFFIX = "_SKIRT"

SKYMODEL_RE = re.compile(
    r"^snapshot_(?P<snapshot>\d+)_(?P<region>[A-Za-z]+)_flux_map_(?P<telescope>[A-Za-z]+)"
    r"_axis_(?P<axis>[a-zA-Z])" + re.escape(SKIRT_SUFFIX) + r"\.fits$"
)


def parse_skirt_skymodel_filename(fname):
    """Parse a ``*_SKIRT.fits`` sky-model filename -> (snapshot, region, axis).

    Deliberately regex-based rather than positional splitting: the trailing
    ``_SKIRT`` suffix makes ``fname.split("_")[-1]`` (used for the thin
    sky-model filename in ``casa_simulation.py``) return "SKIRT" instead of
    the axis letter.
    """
    m = SKYMODEL_RE.match(fname)
    if not m:
        raise ValueError(
            "Filename does not match the expected SKIRT sky-model pattern "
            "'snapshot_<id>_<Region>_flux_map_<TEL>_axis_<axis>_SKIRT.fits': %r" % fname
        )
    return m.group("snapshot"), m.group("region"), m.group("axis")


def simulate_one(skymodel_path, out_dir, settings):
    """Run simobserve -> tclean -> impbcor -> exportfits for one SKIRT sky model.

    Parameters
    ----------
    skymodel_path : str
        Path to a SKIRT sky-model FITS file (``*_SKIRT.fits``).
    out_dir : str
        Directory where the final ``*_sim_observed_pbcor_SKIRT.fits`` is
        written (the same directory the thin pbcor images live in).
    settings : dict
        A copy of OBS_SETTINGS (or a user-modified version).
    """
    fname = os.path.basename(skymodel_path)
    snapshot, site, axis = parse_skirt_skymodel_filename(fname)
    telescope = "ALMA"

    # Read the model header to set the brightness scale and input cell size.
    # Same header keys as the thin pipeline -- do not change the SKIRT FITS
    # export format, this script (like casa_simulation.py) depends on it.
    hdr = imhead(skymodel_path, mode="list")
    inbright = hdr["datamax"]                       # peak brightness in the model
    imgmin = hdr["datamin"]                         # used as the clean threshold
    incell = hdr["cdelt1"] * 3600.0                 # model cell size in arcsec

    # Choose output image size based on how finely the model is sampled.
    if incell > settings["incell_threshold_arcsec"]:
        npix = settings["npix_coarse"]
    else:
        npix = settings["npix_fine"]
    imsize = [npix, npix]

    antennalist = settings["antennalist"]
    # Suffix the CASA project name too, so the working directory / measurement
    # set for a SKIRT run never collides with a thin run of the same
    # snapshot/axis/region.
    project_name = "%s_snapshot_%s_axis_%s_%s%s" % (telescope, snapshot, axis, site, SKIRT_SUFFIX)

    print("=" * 70)
    print("Simulating (SKIRT):", project_name)
    print("  model        :", skymodel_path)
    print("  inbright     :", inbright)
    print("  incell       : %.6f arcsec" % incell)
    print("  imsize       :", imsize)

    # 1) Simulate the observation.
    simobserve(
        project=project_name,
        skymodel=skymodel_path,
        inbright="%s" % inbright,
        indirection=settings["indirection"],
        incell="%.6farcsec" % incell,
        incenter=settings["incenter"],
        inwidth=settings["inwidth"],
        integration=settings["integration"],
        totaltime=settings["totaltime"],
        antennalist="%s.cfg" % antennalist,
        thermalnoise=settings["thermalnoise"],
    )

    # 2) Image / deconvolve the simulated visibilities.
    tclean(
        vis="%s/%s.%s.ms" % (project_name, project_name, antennalist),
        imagename="%s/%s_sim_observed" % (project_name, project_name),
        imsize=imsize,
        cell="%farcsec" % settings["cell_tclean_arcsec"],
        weighting=settings["weighting"],
        niter=settings["niter"],
        threshold="%sJy" % imgmin,
        deconvolver=settings["deconvolver"],
        interactive=False,
        stokes=settings["stokes"],
        restoringbeam="common",
        usemask="auto-multithresh",
    )

    # 3) Primary-beam correction.
    impbcor(
        imagename="%s/%s_sim_observed.image" % (project_name, project_name),
        pbimage="%s/%s_sim_observed.pb" % (project_name, project_name),
        outfile="%s/%s_sim_observed.pbcor" % (project_name, project_name),
        overwrite=True,
    )

    # 4) Export the pb-corrected image to FITS in the shared output folder,
    #    using the naming convention (SKIRT suffix immediately before the
    #    extension) rather than reusing `project_name`, so it lines up with
    #    what plotting_utils.py / analysis_skirt.py expect regardless of any
    #    future change to the internal CASA project-name scheme.
    out_fits = "%s/ALMA_snapshot_%s_axis_%s_%s_sim_observed_pbcor%s.fits" % (
        out_dir, snapshot, axis, site, SKIRT_SUFFIX)
    exportfits(
        imagename="%s/%s_sim_observed.pbcor" % (project_name, project_name),
        fitsimage=out_fits,
        overwrite=True,
    )
    print("  exported     :", out_fits)


def main():
    parser = argparse.ArgumentParser(
        description="Run CASA simobserve/tclean/impbcor on SKIRT sky-model FITS files."
    )
    parser.add_argument(
        "--skymodel-dir", default="skymodels",
        help="Directory containing sky-model FITS files (default: skymodels).",
    )
    parser.add_argument(
        "--skymodels", nargs="+", default=None,
        help="Explicit list of SKIRT sky-model FITS files (overrides --skymodel-dir).",
    )
    parser.add_argument(
        "--glob", default="*_SKIRT.fits",
        help="Glob used with --skymodel-dir to find SKIRT sky models "
             "(default: *_SKIRT.fits -- restricts to the SKIRT variant even "
             "when thin sky models live in the same folder).",
    )
    parser.add_argument(
        "--out-dir", default="pbcor_imgs",
        help="Directory for the exported pbcor FITS images (default: pbcor_imgs, "
             "same folder the thin images are written to).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip a model if its *_sim_observed_pbcor_SKIRT.fits already exists.",
    )
    # CASA passes its own flags too; parse only what we know and ignore the rest.
    args, _unknown = parser.parse_known_args()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    if args.skymodels:
        skymodels = args.skymodels
    else:
        skymodels = sorted(glob.glob(os.path.join(args.skymodel_dir, args.glob)))

    if not skymodels:
        raise SystemExit("No SKIRT sky-model FITS files found to process.")

    completed = set(os.listdir(args.out_dir))
    settings = dict(OBS_SETTINGS)

    print("Found %d SKIRT sky model(s) to process." % len(skymodels))
    for skymodel in skymodels:
        fname = os.path.basename(skymodel)
        snapshot, site, axis = parse_skirt_skymodel_filename(fname)
        out_fits_name = "ALMA_snapshot_%s_axis_%s_%s_sim_observed_pbcor%s.fits" % (
            snapshot, axis, site, SKIRT_SUFFIX)

        if args.skip_existing and out_fits_name in completed:
            print("Skipping (already done):", out_fits_name)
            continue

        simulate_one(skymodel, args.out_dir, settings)

    print("\nAll requested SKIRT simulations complete.")


if __name__ == "__main__":
    main()
