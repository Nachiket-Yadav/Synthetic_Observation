#!/usr/bin/env python3
"""
plot_thin_vs_skirt.py
======================

Two-row comparison figure for one snapshot/field/axis: the optically-thin
('thin') pipeline on top, the SKIRT Monte Carlo pipeline ('skirt') on the
bottom, each rendered as the usual three panels

    [0] Skymodel  |  [1] CASA pbcor observation  |  [2] imfit residual

Both rows are drawn by the same ``render_disk_row`` helper used by
``plot_three_panel.py`` (see ``plotting_utils.py``), so the only difference
between the rows is which files get loaded (thin vs. ``*_SKIRT`` files) and
which fit is annotated.

Why a fixed zoom (and shared colour scale)
-------------------------------------------
imfit measures a very different Rmaj for the same disk depending on which
skymodel it was observed from (e.g. Rmaj = 0.151" / 30 AU on the thin
image but 1.197" / 239 AU on the SKIRT image of snapshot 170). The
Rmaj-relative zoom used elsewhere in this repo (``--zoom-factor``) would
therefore put the two rows at wildly different physical scales -- not a
fair comparison. This script defaults to ``--fixed-au 300``: a physical
half-width shared by all six panels, chosen to sit inside both the thin
skymodel's extent and the pbcor image's extent for snapshot 170 / Orion /
z, so neither row clamps to its full image. Pass ``--no-fixed-au`` to fall
back to the per-row Rmaj-relative zoom instead (not recommended for this
comparison plot). The top row's skymodel/observation colour normalisation
is also reused for the bottom row so brightness is comparable at a glance;
the residual panels keep their own norm since the two runs' residual
amplitudes are not expected to match.

Usage
-----
    python plot_thin_vs_skirt.py --snapshot 170 --field Orion --axis z \\
        --thin-results fitting_results.json \\
        --skirt-results fitting_results_skirt.json \\
        --skymodel-dir skymodels --pbcor-dir pbcor_imgs \\
        --residual-dir residual_imgs --out-dir figures

Run ``python plot_thin_vs_skirt.py --help`` for the full list of options.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from plotting_utils import render_disk_row, style_ax


def plot_thin_vs_skirt(snapshot, field, axis, thin_results, skirt_results,
                        skymodel_dir, pbcor_dir, residual_dir,
                        zoom_factor=4, fixed_au=300.0, cmap="jet", dpi=130,
                        save=True, out_dir="figures", savefig=None, show=False):
    """Render the 2x3 thin-vs-SKIRT comparison figure for one disk.

    Parameters
    ----------
    snapshot, field, axis : str -- which disk to plot, e.g. '170', 'Orion', 'z'.
    thin_results, skirt_results : dict -- loaded fitting_results.json /
        fitting_results_skirt.json contents.
    skymodel_dir, pbcor_dir, residual_dir : str -- shared folders; the thin
        and SKIRT files are distinguished by the '_SKIRT' filename suffix,
        not by folder.
    zoom_factor : float -- fallback Rmaj-relative zoom, used only if
        ``fixed_au`` is None.
    fixed_au : float or None -- fixed physical zoom half-width in AU shared
        by both rows (see module docstring for why this matters here).
        None falls back to the independent Rmaj-relative zoom per row.
    save, out_dir, savefig, show : see ``plot_three_panel``.

    Returns
    -------
    str or None -- the path the figure was saved to, or None if a required
    input (fit entry or FITS file) was missing for either variant.
    """
    thin_fit = thin_results.get(snapshot, {}).get(field, {}).get(axis)
    skirt_fit = skirt_results.get(snapshot, {}).get(field, {}).get(axis)
    if thin_fit is None or skirt_fit is None:
        missing = "thin" if thin_fit is None else "skirt"
        print(f"[skip] snap {snapshot} | {field} | axis {axis}: no fit in {missing} results")
        return None

    thin_pbcor = os.path.join(
        pbcor_dir, f"ALMA_snapshot_{snapshot}_axis_{axis}_{field}_sim_observed_pbcor.fits")
    skirt_pbcor = os.path.join(
        pbcor_dir, f"ALMA_snapshot_{snapshot}_axis_{axis}_{field}_sim_observed_pbcor_SKIRT.fits")
    for p in (thin_pbcor, skirt_pbcor):
        if not os.path.exists(p):
            print(f"[skip] pbcor not found: {p}")
            return None

    fig, axes = plt.subplots(2, 3, figsize=(24, 12), gridspec_kw={"wspace": 0.35, "hspace": 0.45})
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes.flat:
        style_ax(ax)

    top = render_disk_row(
        fig, axes[0, 0], axes[0, 1], axes[0, 2], thin_pbcor, thin_fit,
        skymodel_dir, residual_dir, zoom_factor=zoom_factor, cmap=cmap,
        suffix="", fixed_au=fixed_au, title_prefix="THIN",
    )
    if top is None:
        plt.close(fig)
        return None

    bottom = render_disk_row(
        fig, axes[1, 0], axes[1, 1], axes[1, 2], skirt_pbcor, skirt_fit,
        skymodel_dir, residual_dir, zoom_factor=zoom_factor, cmap=cmap,
        suffix="_SKIRT", fixed_au=fixed_au, title_prefix="SKIRT",
        sky_norm=top["sky_norm"], obs_norm=top["obs_norm"],
    )
    if bottom is None:
        plt.close(fig)
        return None

    zoom_label = f"fixed zoom ±{fixed_au:.0f} AU" if fixed_au is not None else f"zoom ±{zoom_factor:.2f} x Rmaj (per row)"
    fig.suptitle(f"snap {snapshot}  |  {field}  |  axis {axis}  |  thin vs. SKIRT  ({zoom_label})",
                 color="white", fontsize=13, y=1.01)

    plt.tight_layout()
    saved_path = None
    if savefig:
        os.makedirs(os.path.dirname(savefig) or ".", exist_ok=True)
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        saved_path = savefig
    elif save and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_fname = f"snap{snapshot}_{field}_axis{axis}_thin_vs_skirt.png"
        saved_path = os.path.join(out_dir, out_fname)
        fig.savefig(saved_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)

    if saved_path:
        print(f"Saved: {saved_path}")
    return saved_path


def main():
    parser = argparse.ArgumentParser(
        description="Two-row thin-vs-SKIRT comparison figure for one snapshot/field/axis."
    )
    parser.add_argument("--snapshot", required=True, help="Snapshot ID, e.g. 170.")
    parser.add_argument("--field", default="Orion", help="Region (default: Orion).")
    parser.add_argument("--axis", default="z", help="Projection axis (default: z).")
    parser.add_argument("--thin-results", default="fitting_results.json",
                         help="Thin-variant fitting results JSON (default: fitting_results.json).")
    parser.add_argument("--skirt-results", default="fitting_results_skirt.json",
                         help="SKIRT-variant fitting results JSON (default: fitting_results_skirt.json).")
    parser.add_argument("--skymodel-dir", default="skymodels",
                         help="Folder of stage-1 skymodel FITS files (default: skymodels).")
    parser.add_argument("--pbcor-dir", default="pbcor_imgs",
                         help="Folder of stage-2 pbcor FITS images (default: pbcor_imgs).")
    parser.add_argument("--residual-dir", default="residual_imgs",
                         help="Folder of stage-3 residual FITS images (default: residual_imgs).")
    parser.add_argument("--out-dir", default="figures",
                         help="Folder to write the PNG figure to (default: figures).")
    parser.add_argument("--zoom-factor", type=float, default=4,
                         help="Fallback Rmaj-relative zoom per row, used only with "
                              "--no-fixed-au (default: 4).")
    parser.add_argument("--fixed-au", type=float, default=300.0,
                         help="Fixed physical zoom half-width in AU, shared by both rows "
                              "(default: 300).")
    parser.add_argument("--no-fixed-au", action="store_true",
                         help="Disable the fixed-AU zoom and fall back to each row's own "
                              "Rmaj-relative zoom (--zoom-factor). Not recommended here: thin "
                              "and SKIRT fit very different Rmaj for the same disk, so the two "
                              "rows would end up at different physical scales.")
    parser.add_argument("--dpi", type=int, default=130, help="Figure DPI (default: 130).")
    args = parser.parse_args()

    fixed_au = None if args.no_fixed_au else args.fixed_au

    with open(args.thin_results, "r") as f:
        thin_results = json.load(f)
    with open(args.skirt_results, "r") as f:
        skirt_results = json.load(f)

    plot_thin_vs_skirt(
        args.snapshot, args.field, args.axis, thin_results, skirt_results,
        args.skymodel_dir, args.pbcor_dir, args.residual_dir,
        zoom_factor=args.zoom_factor, fixed_au=fixed_au, dpi=args.dpi,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
