#!/usr/bin/env python3
"""
plot_three_panel.py
====================

Make zoomed three-panel QA figures for each fitted disk:

    [0] Skymodel (input)  |  [1] CASA pbcor observation  |  [2] imfit residual

The Gaussian ellipse fitted by ``analysis.py`` is overlaid on all three
panels, and the observation panel is annotated with the fitted size, flux,
inclination, S/N, and residual fraction. This is a visual QA step over the
outputs of stages 1-3, letting you eyeball how well the CASA pipeline
recovered each simulated disk.

Usage
-----
Loop over every snapshot/region/axis in fitting_results.json and write PNGs
to ./figures:

    python plot_three_panel.py --results fitting_results.json \\
        --skymodel-dir skymodels --pbcor-dir pbcor_imgs \\
        --residual-dir residual_imgs --out-dir figures

Restrict to specific snapshots/regions/axes:

    python plot_three_panel.py --results fitting_results.json \\
        --skymodel-dir skymodels --pbcor-dir pbcor_imgs \\
        --residual-dir residual_imgs --out-dir figures \\
        --snapshots 170 386 --fields Orion --axes y

Plot the SKIRT variant instead of the default thin one (reads
``*_SKIRT.fits`` skymodels/pbcor/residual files and, unless overridden,
``fitting_results_skirt.json``):

    python plot_three_panel.py --variant skirt --snapshots 170 --fields Orion --axes z

Run ``python plot_three_panel.py --help`` for the full list of options.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")

from plotting_utils import plot_three_panel


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Make zoomed three-panel QA figures (skymodel | observation | residual)."
    )
    parser.add_argument("--results", default=None,
                         help="Fitting results JSON written by analysis.py. Default: "
                              "fitting_results.json for --variant thin, "
                              "fitting_results_skirt.json for --variant skirt.")
    parser.add_argument("--skymodel-dir", default="skymodels",
                         help="Folder of stage-1 skymodel FITS files (default: skymodels).")
    parser.add_argument("--pbcor-dir", default="pbcor_imgs",
                         help="Folder of stage-2 pbcor FITS images (default: pbcor_imgs).")
    parser.add_argument("--residual-dir", default="residual_imgs",
                         help="Folder of stage-3 residual FITS images (default: residual_imgs).")
    parser.add_argument("--out-dir", default="figures",
                         help="Folder to write PNG figures to (default: figures).")
    parser.add_argument("--fields", nargs="+", default=["Orion", "Perseus"],
                         help="Regions to plot (default: Orion Perseus).")
    parser.add_argument("--snapshots", nargs="+", default=None,
                         help="Restrict to these snapshot IDs (default: all in --results).")
    parser.add_argument("--axes", nargs="+", default=["x", "y", "z"],
                         help="Projection axes to plot (default: x y z).")
    parser.add_argument("--zoom-factor", type=float, default=4,
                         help="Zoom half-width in units of Rmaj (default: 4). Ignored when "
                              "--fixed-au is given.")
    parser.add_argument("--fixed-au", type=float, default=None,
                         help="Fixed physical zoom half-width in AU, identical across the "
                              "three panels regardless of pixel scale (default: None, i.e. "
                              "keep the Rmaj-relative --zoom-factor zoom). Use this to compare "
                              "thin and SKIRT figures at the same physical scale -- see "
                              "plot_thin_vs_skirt.py for a figure that does this automatically.")
    parser.add_argument("--variant", choices=["thin", "skirt"], default="thin",
                         help="Which skymodel variant to plot: 'thin' (default, the original "
                              "optically-thin pipeline -- unchanged behaviour) or 'skirt' (SKIRT "
                              "Monte Carlo skymodels, reads/writes '*_SKIRT' files).")
    parser.add_argument("--dpi", type=int, default=130, help="Figure DPI (default: 130).")
    args = parser.parse_args()

    suffix = "" if args.variant == "thin" else "_SKIRT"
    results_path = args.results
    if results_path is None:
        results_path = "fitting_results.json" if args.variant == "thin" else "fitting_results_skirt.json"

    with open(results_path, "r") as f:
        master_dict = json.load(f)

    snapshots = args.snapshots if args.snapshots else list(master_dict.keys())

    n_saved = 0
    for snapshot in snapshots:
        fields = master_dict.get(snapshot, {})
        for field in args.fields:
            axes_dict = fields.get(field, {})
            for axis in args.axes:
                fit = axes_dict.get(axis)
                if fit is None:
                    continue
                pbcor_fname = f"ALMA_snapshot_{snapshot}_axis_{axis}_{field}_sim_observed_pbcor{suffix}.fits"
                pbcor_fpath = os.path.join(args.pbcor_dir, pbcor_fname)
                if not os.path.exists(pbcor_fpath):
                    print(f"[skip] pbcor not found: {pbcor_fname}")
                    continue
                print(f"Plotting: snapshot {snapshot} | {field} | axis {axis}")
                out_fname = f"snap{snapshot}_{field}_axis{axis}{suffix}_three_panel.png"
                saved = plot_three_panel(
                    pbcor_fpath, fit, args.skymodel_dir, args.residual_dir,
                    zoom_factor=args.zoom_factor, dpi=args.dpi,
                    savefig=os.path.join(args.out_dir, out_fname),
                    suffix=suffix, fixed_au=args.fixed_au,
                )
                if saved:
                    n_saved += 1

    print(f"\nDone. Saved {n_saved} figure(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
