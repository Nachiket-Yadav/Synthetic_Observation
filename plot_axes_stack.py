#!/usr/bin/env python3
"""
plot_axes_stack.py
===================

Make one three-panel QA figure per (snapshot, field), stacking that
snapshot's projection axes (x/y/z by default) as rows:

    [0] Skymodel (input)  |  [1] CASA pbcor observation  |  [2] imfit residual

Companion to ``plot_three_panel.py`` (one row per snapshot/field/axis) and
``plot_three_panel_stack`` in ``plotting_utils.py`` (rows = snapshots for a
fixed axis) -- this instead fixes the snapshot/field and stacks its axes.
Every row is drawn by the same ``render_disk_row`` helper used everywhere
else in this repo, so a given snapshot/axis renders identically regardless
of which driver produced the figure.

Usage
-----
Loop over every (snapshot, field) in fitting_results.json and write one PNG
per combination to ./plots_three_panel:

    python plot_axes_stack.py --results fitting_results.json \\
        --skymodel-dir skymodels --pbcor-dir pbcor_imgs \\
        --residual-dir residual_imgs --out-dir plots_three_panel

Restrict to specific snapshots/fields, and drop the fit-parameter info box:

    python plot_axes_stack.py --snapshots 170 386 --fields Orion \\
        --no-info-box --out-dir plots_three_panel/without_info_box

Run ``python plot_axes_stack.py --help`` for the full list of options.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")

from plotting_utils import plot_axes_stack


def main():
    parser = argparse.ArgumentParser(
        description="Make one three-panel QA figure per (snapshot, field), stacking axes as rows."
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
    parser.add_argument("--out-dir", default="plots_three_panel",
                         help="Folder to write PNG figures to (default: plots_three_panel).")
    parser.add_argument("--fields", nargs="+", default=["Orion", "Perseus"],
                         help="Regions to plot (default: Orion Perseus).")
    parser.add_argument("--snapshots", nargs="+", default=None,
                         help="Restrict to these snapshot IDs (default: all in --results).")
    parser.add_argument("--axes", nargs="+", default=["x", "y", "z"],
                         help="Projection axes to stack as rows, in order (default: x y z).")
    parser.add_argument("--zoom-factor", type=float, default=3,
                         help="Zoom half-width in units of Rmaj (default: 3). Ignored when "
                              "--fixed-au is given.")
    parser.add_argument("--fixed-au", type=float, default=None,
                         help="Fixed physical zoom half-width in AU, identical across all "
                              "panels/rows regardless of pixel scale (default: None, i.e. keep "
                              "the Rmaj-relative --zoom-factor zoom).")
    parser.add_argument("--variant", choices=["thin", "skirt"], default="thin",
                         help="Which skymodel variant to plot: 'thin' (default) or 'skirt' "
                              "(SKIRT Monte Carlo skymodels, reads/writes '*_SKIRT' files).")
    parser.add_argument("--no-info-box", action="store_true",
                         help="Hide the overlaid fit-parameter text box on the CASA "
                              "observation panel (default: shown).")
    parser.add_argument("--no-legend", action="store_true",
                         help="Hide the per-panel FWHM-fit legends (default: shown).")
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
            if field not in fields:
                continue
            print(f"Plotting: snapshot {snapshot} | {field}")
            info_tag = "" if not args.no_info_box else "_noinfo"
            out_fname = f"snap{snapshot}_{field}_axes_{''.join(args.axes)}_stack{suffix}{info_tag}.png"
            saved = plot_axes_stack(
                snapshot, field, master_dict,
                args.skymodel_dir, args.pbcor_dir, args.residual_dir,
                axes=args.axes, zoom_factor=args.zoom_factor, fixed_au=args.fixed_au,
                dpi=args.dpi, savefig=os.path.join(args.out_dir, out_fname),
                suffix=suffix, show_legend=not args.no_legend,
                show_info_box=not args.no_info_box,
            )
            if saved:
                n_saved += 1

    print(f"\nDone. Saved {n_saved} figure(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
