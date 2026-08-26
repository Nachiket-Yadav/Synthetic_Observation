#!/usr/bin/env python3
"""
plot_gallery.py
================

Paper-quality gallery figures: one row per snapshot, one column per
projection axis (x, y, z). Unlike the dark-themed QA figures elsewhere in
this repo (``plot_three_panel.py``, ``plot_axes_stack.py``), these are
white-background, black-text, serif-font figures at roughly AASTeX
body-text size, meant to be dropped straight into a manuscript -- see
``plot_gallery`` in ``plotting_utils.py`` for the rendering details.

One gallery = one (kind, field, colour mode) combination:

    kind  : 'skymodel' (stage-1 input) or 'pbcor' (stage-2 CASA observation)
    field : 'Orion' or 'Perseus'
    mode  : 'unscaled' (each panel normalised to its own data, own colorbar)
            or 'scaled' (every panel shares one colour norm taken from the
            brightest snapshot/axis in the gallery, one shared colorbar)

Each panel is shown at its own native frame footprint by default (no
cropping/zooming, nothing forced into a shared field-of-view "box") --
pass --shared-fov to instead crop every panel to one common physical FOV
sized from Rmaj, the old behaviour.

Usage
-----
Default run makes all 8 thin-variant galleries (2 kinds x 2 fields x
{scaled, unscaled}) for the snapshots currently under investigation:

    python plot_gallery.py

Restrict to specific snapshots/fields/kinds:

    python plot_gallery.py --snapshots 169 171 346 --fields Orion \\
        --kinds skymodel --out-dir gallery_figures

Only the scaled (or unscaled) version:

    python plot_gallery.py --modes scaled

SKIRT variant (once SKIRT skymodels/pbcor images exist -- reads/writes
``*_SKIRT`` files and defaults to ``fitting_results_skirt.json``):

    python plot_gallery.py --variant skirt

Run ``python plot_gallery.py --help`` for the full list of options.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")

from plotting_utils import plot_gallery

# Snapshots currently under investigation -- one gallery row each, in order.
DEFAULT_SNAPSHOTS = ["169", "171", "106", "307", "417", "319", "320", "323", "346"]


def main():
    parser = argparse.ArgumentParser(
        description="Paper-quality gallery figures (rows = snapshots, columns = x/y/z projection axes)."
    )
    parser.add_argument("--snapshots", nargs="+", default=DEFAULT_SNAPSHOTS,
                         help=f"Snapshot IDs, one row each, in order (default: {' '.join(DEFAULT_SNAPSHOTS)}).")
    parser.add_argument("--fields", nargs="+", default=["Orion", "Perseus"],
                         help="Regions, one gallery each (default: Orion Perseus).")
    parser.add_argument("--kinds", nargs="+", choices=["skymodel", "pbcor"], default=["skymodel", "pbcor"],
                         help="Image type(s) to make galleries for: 'skymodel' (stage-1 input) "
                              "and/or 'pbcor' (stage-2 CASA observation). Default: both.")
    parser.add_argument("--modes", nargs="+", choices=["scaled", "unscaled"], default=["scaled", "unscaled"],
                         help="Colour-scale variant(s) to write (default: both).")
    parser.add_argument("--axes", nargs="+", default=["x", "y", "z"],
                         help="Projection axes, one column each, in order (default: x y z).")
    parser.add_argument("--variant", choices=["thin", "skirt"], default="thin",
                         help="Which skymodel variant to plot: 'thin' (default) or 'skirt' "
                              "(reads/writes '*_SKIRT' files).")
    parser.add_argument("--skymodel-dir", default="skymodels",
                         help="Folder of stage-1 skymodel FITS files (default: skymodels).")
    parser.add_argument("--pbcor-dir", default="pbcor_imgs",
                         help="Folder of stage-2 pbcor FITS images (default: pbcor_imgs).")
    parser.add_argument("--results", default=None,
                         help="Fitting results JSON, used only to size each gallery's shared "
                              "field of view from Rmaj. Default: fitting_results.json for "
                              "--variant thin, fitting_results_skirt.json for --variant skirt. "
                              "If missing, the shared FOV falls back to the smallest native "
                              "frame footprint in that gallery.")
    parser.add_argument("--out-dir", default="gallery_figures",
                         help="Folder to write PNG figures to (default: gallery_figures).")
    parser.add_argument("--shared-fov", action="store_true",
                         help="Crop every panel to one shared physical field of view instead of "
                              "each panel's own native frame footprint (the default). The shared "
                              "FOV is sized from --zoom-factor x the largest fitted Rmaj in the "
                              "gallery (or --fixed-au explicitly).")
    parser.add_argument("--zoom-factor", type=float, default=3.0,
                         help="Shared field-of-view half-width, in units of the largest fitted "
                              "Rmaj anywhere in the gallery (default: 3). Only used with "
                              "--shared-fov; ignored when --fixed-au is given.")
    parser.add_argument("--fixed-au", type=float, default=None,
                         help="Explicit shared physical half-width (AU) for every gallery, "
                              "overriding the Rmaj-based sizing. Only used with --shared-fov.")
    parser.add_argument("--rmaj-exclude", nargs="+", default=None, metavar="SNAPSHOT:AXIS",
                         help="Leave these snapshot/axis fits out of the shared-FOV sizing, "
                              "e.g. --rmaj-exclude 417:z. Their own panel still renders, cropped "
                              "to whatever FOV the rest of the gallery ends up with. There is no "
                              "automatic outlier rejection here on purpose -- a large Rmaj can be "
                              "a genuinely extended structure rather than a bad fit (checked by "
                              "eye, not inferred from fit-quality stats, which don't reliably "
                              "tell the two apart in this pipeline). Applies to every gallery "
                              "in this run, so only pass axes you've confirmed for every field. "
                              "Only used with --shared-fov.")
    parser.add_argument("--cmap", default="jet", help="Colormap (default: jet).")
    parser.add_argument("--log-scale", action="store_true",
                         help="Log-scale the colour norm instead of linear (default: linear).")
    parser.add_argument("--fig-width", type=float, default=7.1,
                         help="Figure width in inches (default: 7.1, AASTeX two-column text width).")
    parser.add_argument("--row-height", type=float, default=2.3,
                         help="Per-row figure height in inches (default: 2.3).")
    parser.add_argument("--fontsize", type=float, default=10,
                         help="Base font size in points, matching AASTeX two-column body text "
                              "(default: 10).")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI (default: 300).")
    args = parser.parse_args()

    suffix = "" if args.variant == "thin" else "_SKIRT"
    results_path = args.results
    if results_path is None:
        results_path = "fitting_results.json" if args.variant == "thin" else "fitting_results_skirt.json"

    results = None
    if args.shared_fov:
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                results = json.load(f)
        else:
            print(f"[warn] {results_path} not found -- each gallery's shared field of view will "
                  f"fall back to its smallest native frame footprint instead of a Rmaj-based one.")

    image_dir_for_kind = {"skymodel": args.skymodel_dir, "pbcor": args.pbcor_dir}

    rmaj_exclude = None
    if args.rmaj_exclude:
        rmaj_exclude = set()
        for entry in args.rmaj_exclude:
            snapshot, _, axis = entry.partition(":")
            if not axis:
                raise SystemExit(f"--rmaj-exclude entries must look like SNAPSHOT:AXIS, got {entry!r}")
            rmaj_exclude.add((snapshot, axis))

    n_saved = 0
    for kind in args.kinds:
        for field in args.fields:
            for mode in args.modes:
                scaled = mode == "scaled"
                print(f"Plotting gallery: {kind} | {field} | {args.variant} | {mode}")
                out_fname = f"gallery_{kind}_{field}{suffix}_{mode}.png"
                saved = plot_gallery(
                    args.snapshots, field, image_dir_for_kind[kind],
                    kind=kind, results=results, axes=args.axes, suffix=suffix,
                    scaled=scaled, native_size=not args.shared_fov,
                    zoom_factor=args.zoom_factor, fixed_au=args.fixed_au,
                    rmaj_exclude=rmaj_exclude,
                    cmap=args.cmap, log_scale=args.log_scale,
                    fig_width_in=args.fig_width, row_height_in=args.row_height,
                    fontsize=args.fontsize, dpi=args.dpi,
                    savefig=os.path.join(args.out_dir, out_fname),
                )
                if saved:
                    n_saved += 1

    print(f"\nDone. Saved {n_saved} figure(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
