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

Run ``python plot_three_panel.py --help`` for the full list of options.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import Ellipse
from astropy.io import fits


DISTANCES_PC = {"Orion": 400, "Perseus": 300}


# ---------------------------------------------------------------------------
# FITS / display helpers
# ---------------------------------------------------------------------------
def get_pixel_scale_arcsec(header):
    """Return pixel scale in arcsec/pixel from a FITS header."""
    if "CDELT1" in header:
        cdelt = abs(header["CDELT1"])
        unit = header.get("CUNIT1", "deg").strip().lower()
        if unit in ("deg", "degree", "degrees", ""):
            return cdelt * 3600.0
        elif unit == "arcsec":
            return cdelt
        elif unit == "arcmin":
            return cdelt * 60.0
        else:
            return cdelt * 3600.0
    if "CD1_1" in header:
        return abs(header["CD1_1"]) * 3600.0
    raise ValueError("Cannot determine pixel scale.")


def make_norm(data, vmin_pct=0, vmax_pct=99.5, log_scale=False):
    """Clip to positive values, return a LogNorm or plain Normalize."""
    d = np.where(data > 0, data, np.nan)
    vmin = np.nanpercentile(d, vmin_pct)
    vmax = np.nanpercentile(d, vmax_pct)

    if log_scale:
        vmin = max(vmin, vmax * 1e-4)
        return LogNorm(vmin=vmin, vmax=vmax)
    return plt.Normalize(vmin=vmin, vmax=vmax)


def make_residual_norm(data):
    """Symmetric diverging norm centred on zero for residual images."""
    vmax = np.nanmax(np.abs(data))
    vmax = max(vmax, 1e-10)  # avoid zero-range norm
    return TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)


def zoom_bounds(cx, cy, Rmaj_pix, nx, ny, factor=4):
    """Return (x0, x1, y0, y1) pixel bounds for a zoomed cutout."""
    r = max(int(Rmaj_pix * factor), 30)
    return (
        max(int(cx) - r, 0), min(int(cx) + r, nx),
        max(int(cy) - r, 0), min(int(cy) + r, ny),
    )


def add_AU_ticks(ax, cx, cy, x0, x1, y0, y1, pix_scale, distance_pc, n_ticks=5):
    """Replace pixel tick labels with AU offsets from the image centre."""
    hw_x_AU = (x1 - x0) / 2 * pix_scale * distance_pc
    hw_y_AU = (y1 - y0) / 2 * pix_scale * distance_pc
    raw_step = min(hw_x_AU, hw_y_AU) / (n_ticks // 2)
    magnitude = 10 ** np.floor(np.log10(max(raw_step, 1e-10)))
    nice = magnitude * min([1, 2, 5, 10], key=lambda x: abs(x - raw_step / magnitude))
    step_AU = max(nice, 1.0)
    xtick_AU = np.arange(-hw_x_AU, hw_x_AU + step_AU, step_AU)
    ytick_AU = np.arange(-hw_y_AU, hw_y_AU + step_AU, step_AU)
    pix_per_AU = 1.0 / (pix_scale * distance_pc)
    ax.set_xticks(xtick_AU * pix_per_AU + cx)
    ax.set_xticklabels([f"{v:.0f}" for v in xtick_AU], color="white")
    ax.set_yticks(ytick_AU * pix_per_AU + cy)
    ax.set_yticklabels([f"{v:.0f}" for v in ytick_AU], color="white")


def draw_ellipse_on_ax(ax, cx, cy, Rmaj_pix, Rmin_pix, mpl_angle,
                        color="black", ls="--", lw=1.6, label="FWHM fit"):
    """Draw the FWHM ellipse and a fainter 2-sigma ellipse."""
    ax.add_patch(Ellipse(
        xy=(cx, cy), width=Rmaj_pix, height=Rmin_pix, angle=mpl_angle,
        edgecolor=color, facecolor="none",
        linewidth=lw, linestyle=ls, alpha=1.0, label=label,
    ))
    ax.add_patch(Ellipse(
        xy=(cx, cy), width=2 * Rmaj_pix, height=2 * Rmin_pix, angle=mpl_angle,
        edgecolor=color, facecolor="none",
        linewidth=lw * 0.6, linestyle=":", alpha=0.5,
    ))


def style_ax(ax):
    ax.set_facecolor("#0d0d0d")
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#555")


def add_colorbar(fig, im, ax, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, color="white", fontsize=7)
    cb.ax.yaxis.set_tick_params(color="white", labelsize=6)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")


# ---------------------------------------------------------------------------
# Three-panel plot
# ---------------------------------------------------------------------------
def plot_three_panel(pbcor_fpath, fit, skymodel_dir, residual_dir,
                      zoom_factor=4, cmap="jet", dpi=130,
                      save=True, out_dir="figures", savefig=None):
    """Render one row of three zoomed panels for a single fitted disk.

      [0] Skymodel (zoomed)
      [1] CASA pbcor observation (zoomed)
      [2] imfit residual image (zoomed, diverging colormap)

    The Gaussian ellipse from ``fit`` is overlaid on all three panels.

    Parameters
    ----------
    pbcor_fpath   : str  -- path to the pbcor FITS file
    fit           : dict -- entry from fitting_results[snapshot][field][axis]
    skymodel_dir  : str  -- folder containing the stage-1 skymodel FITS files
    residual_dir  : str  -- folder containing the stage-3 residual FITS files
    zoom_factor   : int  -- half-width of the zoom, in units of Rmaj
    save          : bool -- whether to save a PNG at all
    out_dir       : str  -- folder to save into when ``savefig`` is not given
    savefig       : str or None -- if given, save the PNG to this exact path
                    (overrides ``out_dir``/auto-generated naming); parent
                    directories are created as needed

    Returns
    -------
    str or None -- the path the figure was saved to, or None if not saved.
    """

    # -- Parse filename --------------------------------------------------
    fname = os.path.basename(pbcor_fpath)
    parts = fname.replace(".fits", "").split("_")
    snapshot = parts[2]
    axis = parts[4]
    field = parts[5]
    distance_pc = DISTANCES_PC.get(field, 400)

    # -- Load pbcor --------------------------------------------------------
    with fits.open(pbcor_fpath) as hdul:
        pb_hdr = hdul[0].header
        pb_data = hdul[0].data.squeeze()
    pb_ny, pb_nx = pb_data.shape
    pb_pix_as = get_pixel_scale_arcsec(pb_hdr)
    pb_pix_AU = pb_pix_as * distance_pc
    pb_cx, pb_cy = pb_nx / 2.0, pb_ny / 2.0

    # -- Load skymodel -------------------------------------------------------
    sky_fname = f"snapshot_{snapshot}_{field}_flux_map_ALMA_axis_{axis}.fits"
    sky_fpath = os.path.join(skymodel_dir, sky_fname)
    if not os.path.exists(sky_fpath):
        print(f"[warn] skymodel not found: {sky_fpath}")
        return None
    with fits.open(sky_fpath) as hdul:
        sky_hdr = hdul[0].header
        sky_data = hdul[0].data.squeeze()
    sky_ny, sky_nx = sky_data.shape
    sky_pix_as = get_pixel_scale_arcsec(sky_hdr)
    sky_cx, sky_cy = sky_nx / 2.0, sky_ny / 2.0

    # -- Load residual ---------------------------------------------------------
    res_fname = f"ALMA_snapshot_{snapshot}_axis_{axis}_{field}_sim_observed_pbcor_residual.fits"
    res_fpath = os.path.join(residual_dir, res_fname)
    if not os.path.exists(res_fpath):
        print(f"[warn] residual not found: {res_fpath}")
        return None
    with fits.open(res_fpath) as hdul:
        res_hdr = hdul[0].header
        res_data = hdul[0].data.squeeze()
    res_ny, res_nx = res_data.shape
    res_pix_as = get_pixel_scale_arcsec(res_hdr)
    res_cx, res_cy = res_nx / 2.0, res_ny / 2.0

    # -- Fit parameters ---------------------------------------------------------
    Rmaj_as = fit["Rmaj"]
    Rmin_as = fit["Rmin"]
    pa_deg = fit["pa"]
    inc = fit["inc"]
    r_AU_T = fit["radius_AU_Tobin"]
    flux = fit["flux"]
    snr = fit["snr"]
    prf = fit["peak_residual_fraction"]
    peak_res = fit["peak_residual"]

    fit_valid = all(
        v is not None and not (isinstance(v, float) and np.isnan(v))
        for v in [Rmaj_as, Rmin_as, pa_deg]
    )

    if fit_valid:
        Rmaj_AU = Rmaj_as * distance_pc
        Rmin_AU = Rmin_as * distance_pc
        pb_Rmaj_pix = Rmaj_as / pb_pix_as
        pb_Rmin_pix = Rmin_as / pb_pix_as
        sk_Rmaj_pix = Rmaj_as / sky_pix_as
        sk_Rmin_pix = Rmin_as / sky_pix_as
        res_Rmaj_pix = Rmaj_as / res_pix_as
        res_Rmin_pix = Rmin_as / res_pix_as
        mpl_angle = 90 + pa_deg
    else:
        print(f"[warn] NaN fit for snap {snapshot} | {field} | axis {axis} -- plotting images only")
        pb_Rmaj_pix = pb_nx * 0.1
        sk_Rmaj_pix = sky_nx * 0.1
        res_Rmaj_pix = res_nx * 0.1
        mpl_angle = 0

    # -- Figure ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), gridspec_kw={"wspace": 0.35})
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes:
        style_ax(ax)
    ax_sky, ax_obs, ax_res = axes

    # Panel 0: Skymodel zoomed
    sx0, sx1, sy0, sy1 = zoom_bounds(sky_cx, sky_cy, sk_Rmaj_pix, sky_nx, sky_ny, factor=zoom_factor)
    szd = sky_data[sy0:sy1, sx0:sx1]
    im0 = ax_sky.imshow(szd, origin="lower", cmap=cmap, norm=make_norm(szd), extent=[sx0, sx1, sy0, sy1])
    if fit_valid:
        draw_ellipse_on_ax(ax_sky, sky_cx, sky_cy, sk_Rmaj_pix, sk_Rmin_pix, mpl_angle)
    ax_sky.axhline(sky_cy, color="white", lw=0.4, alpha=0.35)
    ax_sky.axvline(sky_cx, color="white", lw=0.4, alpha=0.35)
    ax_sky.set_xlim(sx0, sx1)
    ax_sky.set_ylim(sy0, sy1)
    add_AU_ticks(ax_sky, sky_cx, sky_cy, sx0, sx1, sy0, sy1, sky_pix_as, distance_pc)
    ax_sky.set_xlabel(f"ΔRA (AU)  [d = {distance_pc} pc]", color="white")
    ax_sky.set_ylabel("ΔDec (AU)", color="white")
    ax_sky.set_title(f"Skymodel  -- zoomed (±{zoom_factor} x Rmaj)", color="white")
    ax_sky.legend(loc="upper right", facecolor="#1a1a1a", edgecolor="#555", labelcolor="white", fontsize=6)
    add_colorbar(fig, im0, ax_sky, sky_hdr.get("BUNIT", "Jy/pixel"))

    # Panel 1: CASA pbcor zoomed
    px0, px1, py0, py1 = zoom_bounds(pb_cx, pb_cy, pb_Rmaj_pix, pb_nx, pb_ny, factor=zoom_factor)
    pzd = pb_data[py0:py1, px0:px1]
    im1 = ax_obs.imshow(pzd, origin="lower", cmap=cmap, norm=make_norm(pzd), extent=[px0, px1, py0, py1])
    if fit_valid:
        draw_ellipse_on_ax(ax_obs, pb_cx, pb_cy, pb_Rmaj_pix, pb_Rmin_pix, mpl_angle)
    ax_obs.axhline(pb_cy, color="white", lw=0.4, alpha=0.35)
    ax_obs.axvline(pb_cx, color="white", lw=0.4, alpha=0.35)
    ax_obs.set_xlim(px0, px1)
    ax_obs.set_ylim(py0, py1)
    add_AU_ticks(ax_obs, pb_cx, pb_cy, px0, px1, py0, py1, pb_pix_as, distance_pc)
    ax_obs.set_xlabel(f"ΔRA (AU)  [d = {distance_pc} pc]", color="white")
    ax_obs.set_ylabel("ΔDec (AU)", color="white")
    ax_obs.set_title(f"CASA Observation  -- zoomed (±{zoom_factor} x Rmaj)", color="white")
    ax_obs.legend(loc="upper right", facecolor="#1a1a1a", edgecolor="#555", labelcolor="white")
    add_colorbar(fig, im1, ax_obs, pb_hdr.get("BUNIT", "Jy/beam"))

    if fit_valid:
        info = (
            f"snap {snapshot}  |  {field}  |  axis {axis}\n"
            f"Rmaj = {Rmaj_as:.3f}\"  ({Rmaj_AU / 2:.0f} AU)\n"
            f"Rmin = {Rmin_as:.3f}\"  ({Rmin_AU / 2:.0f} AU)\n"
            f"PA = {pa_deg:.1f}°    i = {inc:.1f}°\n"
            f"Flux = {flux:.3e} Jy\n"
            f"R_disk (Tobin) = {r_AU_T:.1f} AU\n"
            f"pix = {pb_pix_as:.4f}\" = {pb_pix_AU:.2f} AU\n"
            f"snr = {snr:.1f}\n"
            f"peak residual = {peak_res:.3e} Jy  ({prf:.1%} of peak)"
        )
        ax_obs.text(0.02, 0.98, info, transform=ax_obs.transAxes, va="top", ha="left",
                    color="white", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#111", edgecolor="cyan", alpha=0.88))
    else:
        ax_obs.text(0.02, 0.98, f"snap {snapshot}  |  {field}  |  axis {axis}\nFit failed -- no Gaussian parameters",
                    transform=ax_obs.transAxes, va="top", ha="left", color="orange", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#111", edgecolor="orange", alpha=0.88))

    # Panel 2: Residual zoomed
    rx0, rx1, ry0, ry1 = zoom_bounds(res_cx, res_cy, res_Rmaj_pix, res_nx, res_ny, factor=zoom_factor)
    rzd = res_data[ry0:ry1, rx0:rx1]
    im2 = ax_res.imshow(rzd, origin="lower", cmap="RdBu_r", norm=make_residual_norm(rzd), extent=[rx0, rx1, ry0, ry1])
    if fit_valid:
        draw_ellipse_on_ax(ax_res, res_cx, res_cy, res_Rmaj_pix, res_Rmin_pix, mpl_angle, color="black", ls="--")
    ax_res.axhline(res_cy, color="gray", lw=0.4, alpha=0.35)
    ax_res.axvline(res_cx, color="gray", lw=0.4, alpha=0.35)
    ax_res.set_xlim(rx0, rx1)
    ax_res.set_ylim(ry0, ry1)
    add_AU_ticks(ax_res, res_cx, res_cy, rx0, rx1, ry0, ry1, res_pix_as, distance_pc)
    ax_res.set_xlabel(f"ΔRA (AU)  [d = {distance_pc} pc]", color="white")
    ax_res.set_ylabel("ΔDec (AU)", color="white")
    ax_res.set_title(f"imfit Residual  -- zoomed (±{zoom_factor} x Rmaj)", color="white")
    ax_res.legend(loc="upper right", facecolor="#1a1a1a", edgecolor="#555", labelcolor="white", fontsize=6)
    add_colorbar(fig, im2, ax_res, "Jy/beam  (obs - model)")

    # -- Save --------------------------------------------------------------
    plt.tight_layout()
    saved_path = None
    if savefig:
        os.makedirs(os.path.dirname(savefig) or ".", exist_ok=True)
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        saved_path = savefig
    elif save and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_fname = f"snap{snapshot}_{field}_axis{axis}_three_panel.png"
        saved_path = os.path.join(out_dir, out_fname)
        fig.savefig(saved_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    if saved_path:
        print(f"Saved: {saved_path}")
    return saved_path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Make zoomed three-panel QA figures (skymodel | observation | residual)."
    )
    parser.add_argument("--results", default="fitting_results.json",
                         help="Fitting results JSON written by analysis.py (default: fitting_results.json).")
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
                         help="Zoom half-width in units of Rmaj (default: 4).")
    parser.add_argument("--dpi", type=int, default=130, help="Figure DPI (default: 130).")
    args = parser.parse_args()

    with open(args.results, "r") as f:
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
                pbcor_fname = f"ALMA_snapshot_{snapshot}_axis_{axis}_{field}_sim_observed_pbcor.fits"
                pbcor_fpath = os.path.join(args.pbcor_dir, pbcor_fname)
                if not os.path.exists(pbcor_fpath):
                    print(f"[skip] pbcor not found: {pbcor_fname}")
                    continue
                print(f"Plotting: snapshot {snapshot} | {field} | axis {axis}")
                out_fname = f"snap{snapshot}_{field}_axis{axis}_three_panel.png"
                saved = plot_three_panel(
                    pbcor_fpath, fit, args.skymodel_dir, args.residual_dir,
                    zoom_factor=args.zoom_factor, dpi=args.dpi,
                    savefig=os.path.join(args.out_dir, out_fname),
                )
                if saved:
                    n_saved += 1

    print(f"\nDone. Saved {n_saved} figure(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
