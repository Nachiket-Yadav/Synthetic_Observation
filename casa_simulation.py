"""
casa_simulation.py
==================

Run CASA's synthetic-observation pipeline on the sky-model FITS images
produced by ``skymodel_generation.py``.

For every sky model this script runs, in order:

  1. ``simobserve``  -- simulate an interferometer observation of the model.
  2. ``tclean``      -- image (deconvolve) the simulated visibilities.
  3. ``impbcor``     -- apply the primary-beam correction.
  4. ``exportfits``  -- export the primary-beam-corrected image to FITS.

The exported ``*_sim_observed_pbcor.fits`` files are what the analysis stage
(``analysis.py``) fits with ``imfit``.

IMPORTANT -- this script must be run inside CASA, not plain Python, because it
calls CASA tasks (``simobserve``, ``tclean``, ``imhead`` ...). Two ways to run
it:

    # 1) Non-interactively from a shell:
    casa --nogui --nologger -c casa_simulation.py --skymodel-dir skymodels --out-dir pbcor_imgs

    # 2) From inside an interactive CASA session:
    CASA <1>: import sys
    CASA <2>: sys.argv = ['casa_simulation.py', '--skymodel-dir', 'skymodels',
                          '--out-dir', 'pbcor_imgs']
    CASA <3>: execfile('casa_simulation.py')

The observing setup (pointing direction, frequency, integration time, array configuration,
etc.) is gathered in the ``OBS_SETTINGS`` dictionary so it can be edited in one
place.
"""

import os
import glob
import argparse


# ---------------------------------------------------------------------------
# Observation settings. Edit these to change the simulated observation.
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


def simulate_one(skymodel_path, out_dir, settings):
    """Run simobserve -> tclean -> impbcor -> exportfits for one sky model.

    Parameters
    ----------
    skymodel_path : str
        Path to a sky-model FITS file (output of skymodel_generation.py).
    out_dir : str
        Directory where the final *_sim_observed_pbcor.fits is written.
    settings : dict
        A copy of OBS_SETTINGS (or a user-modified version).
    """
    fname = os.path.basename(skymodel_path)

    # Parse metadata out of the filename:
    #   snapshot_<id>_<Region>_flux_map_<TEL>_axis_<axis>.fits
    parts = fname.split("_")
    snapshot = parts[1]
    axis = parts[-1].split(".")[0]
    site = "Orion" if "Orion" in fname else "Perseus"
    telescope = "ALMA"

    # Read the model header to set the brightness scale and input cell size.
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
    project_name = "%s_snapshot_%s_axis_%s_%s" % (telescope, snapshot, axis, site)

    print("=" * 70)
    print("Simulating:", project_name)
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

    # 4) Export the pb-corrected image to FITS in the shared output folder.
    out_fits = "%s/%s_sim_observed_pbcor.fits" % (out_dir, project_name)
    exportfits(
        imagename="%s/%s_sim_observed.pbcor" % (project_name, project_name),
        fitsimage=out_fits,
        overwrite=True,
    )
    print("  exported     :", out_fits)


def main():
    parser = argparse.ArgumentParser(
        description="Run CASA simobserve/tclean/impbcor on sky-model FITS files."
    )
    parser.add_argument(
        "--skymodel-dir", default="skymodels",
        help="Directory containing sky-model FITS files (default: skymodels).",
    )
    parser.add_argument(
        "--skymodels", nargs="+", default=None,
        help="Explicit list of sky-model FITS files (overrides --skymodel-dir).",
    )
    parser.add_argument(
        "--glob", default="*.fits",
        help="Glob used with --skymodel-dir to find sky models (default: *.fits).",
    )
    parser.add_argument(
        "--out-dir", default="pbcor_imgs",
        help="Directory for the exported pbcor FITS images (default: pbcor_imgs).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip a model if its *_sim_observed_pbcor.fits already exists.",
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
        raise SystemExit("No sky-model FITS files found to process.")

    completed = set(os.listdir(args.out_dir))
    settings = dict(OBS_SETTINGS)

    print("Found %d sky model(s) to process." % len(skymodels))
    for skymodel in skymodels:
        fname = os.path.basename(skymodel)
        parts = fname.split("_")
        snapshot, axis = parts[1], parts[-1].split(".")[0]
        site = "Orion" if "Orion" in fname else "Perseus"
        project_name = "ALMA_snapshot_%s_axis_%s_%s" % (snapshot, axis, site)

        if args.skip_existing and \
                ("%s_sim_observed_pbcor.fits" % project_name) in completed:
            print("Skipping (already done):", project_name)
            continue

        simulate_one(skymodel, args.out_dir, settings)

    print("\nAll requested simulations complete.")


if __name__ == "__main__":
    main()
