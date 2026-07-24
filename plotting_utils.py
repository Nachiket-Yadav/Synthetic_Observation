"""
plotting_utils.py
==================

Shared helpers for the three-panel QA figures (skymodel | CASA pbcor
observation | imfit residual), used by ``plot_three_panel.py`` (CLI batch
driver, thin variant), ``plot_thin_vs_skirt.py`` (thin-vs-SKIRT comparison
driver), and ``plot_three_panel.ipynb`` (interactive exploration).

Two skymodel variants
----------------------
The pipeline supports two independent skymodel sources for the same
snapshot: ``thin`` (the original optically-thin yt projection) and
``skirt`` (SKIRT Monte Carlo radiative transfer, produced outside this
repo). Every filename the SKIRT variant reads or writes is the thin
filename with ``_SKIRT`` appended to the stem, immediately before the
extension -- see ``suffix`` below. Threading a ``suffix`` parameter
through the filename construction (rather than duplicating this module)
is what lets both variants share one rendering code path.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import Ellipse
from astropy.io import fits


DISTANCES_PC = {"Orion": 400, "Perseus": 300}

# Filename-stem suffix for each skymodel variant. Appended immediately
# before the extension (or before a trailing "_residual"/"_model") so that
# positional parsing done elsewhere in the pipeline (e.g. filename.split("_"))
# is unaffected wherever the parsed field comes before the suffix.
VARIANT_SUFFIXES = {"thin": "", "skirt": "_SKIRT"}


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


def zoom_bounds(cx, cy, Rmaj_pix, nx, ny, factor=4,
                 fixed_au=None, pix_as=None, distance_pc=400.0):
    """Pixel bounds for a zoomed cutout.

    fixed_au : fixed PHYSICAL half-width in AU, identical across panels and
        rows -- required for like-for-like comparison (e.g. thin vs. SKIRT,
        which fit very different Rmaj for the same disk). Requires
        ``pix_as`` (this image's own arcsec/pixel scale) to also be given.
        Falls back to ``factor * Rmaj_pix`` with a 30-PIXEL floor, which is
        scale-dependent (e.g. 26 AU on a 0.86 AU/pix skymodel vs. 360 AU on
        a 12 AU/pix pbcor image of the same disk) -- fine within one image,
        not comparable across images of different pixel scale.
    """
    if fixed_au is not None and pix_as is not None:
        r = int(round(fixed_au / (pix_as * distance_pc)))
    else:
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
    # Symmetric tick construction: kx/ky ticks on either side of the centre,
    # never overshooting [-hw_AU, +hw_AU] (np.arange(-hw, hw+step, step) used
    # to overshoot by one step, producing asymmetric labels like -180..220).
    kx = int(np.floor(hw_x_AU / step_AU)); xtick_AU = np.arange(-kx, kx + 1) * step_AU
    ky = int(np.floor(hw_y_AU / step_AU)); ytick_AU = np.arange(-ky, ky + 1) * step_AU
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
# Shared row renderer -- one disk, three panels, into caller-supplied axes
# ---------------------------------------------------------------------------
def render_disk_row(fig, ax_sky, ax_obs, ax_res, pbcor_fpath, fit,
                     skymodel_dir, residual_dir, zoom_factor=4, cmap="jet",
                     suffix="", fixed_au=None,
                     sky_norm=None, obs_norm=None, res_norm=None,
                     title_prefix=None):
    """Render one row of [skymodel | pbcor | residual] panels into the given axes.

    This is the single rendering code path shared by ``plot_three_panel``
    (one row, thin *or* SKIRT depending on ``suffix``) and
    ``plot_thin_vs_skirt.py`` (two rows, one thin + one SKIRT, sharing axes
    limits/colour scale via ``fixed_au``/``sky_norm``/``obs_norm``). Keeping
    exactly one copy of this logic is what guarantees the two variants are
    rendered identically apart from the input files and the requested zoom.

    Parameters
    ----------
    fig                     : Figure -- needed for colorbars.
    ax_sky, ax_obs, ax_res  : Axes -- panels to draw into (already styled).
    pbcor_fpath             : str  -- path to the pbcor FITS file.
    fit                     : dict -- entry from fitting_results[snapshot][field][axis].
    skymodel_dir, residual_dir : str -- folders for the other two stages.
    suffix                  : str  -- "" for thin, "_SKIRT" for the SKIRT
                              variant; selects which skymodel/residual files
                              are loaded alongside ``pbcor_fpath``.
    fixed_au                : float or None -- fixed physical zoom half-width
                              in AU, identical across all three panels; see
                              ``zoom_bounds``. None keeps the original
                              Rmaj-relative zoom (``zoom_factor``).
    sky_norm, obs_norm, res_norm : Normalize or None -- reuse an existing
                              colour norm instead of computing one from this
                              row's data (for cross-row comparability).
    title_prefix            : str or None -- prefixed to panel titles and the
                              annotation box, e.g. "THIN" / "SKIRT".

    Returns
    -------
    dict or None -- metadata about what was drawn (snapshot/field/axis, the
    norms actually used, fit validity), or None if a required file was
    missing (nothing was drawn).
    """
    # -- Parse filename (variant-independent: snapshot/axis/field always sit
    #    at the same split() indices regardless of a trailing _SKIRT/suffix,
    #    since the suffix is appended after these fields in every filename). --
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
    sky_fname = f"snapshot_{snapshot}_{field}_flux_map_ALMA_axis_{axis}{suffix}.fits"
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
    res_fname = f"ALMA_snapshot_{snapshot}_axis_{axis}_{field}_sim_observed_pbcor{suffix}_residual.fits"
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
    Rmaj_as = fit.get("Rmaj")
    Rmin_as = fit.get("Rmin")
    pa_deg = fit.get("pa")
    inc = fit.get("inc")
    r_AU_T = fit.get("radius_AU_Tobin")
    flux = fit.get("flux")
    snr = fit.get("snr")
    prf = fit.get("peak_residual_fraction")
    peak_res = fit.get("peak_residual")

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

    zoom_label = f"±{fixed_au:.0f} AU" if fixed_au is not None else f"±{zoom_factor:.2f} x Rmaj"
    prefix = f"{title_prefix} " if title_prefix else ""

    # Panel 0: Skymodel zoomed
    sx0, sx1, sy0, sy1 = zoom_bounds(sky_cx, sky_cy, sk_Rmaj_pix, sky_nx, sky_ny,
                                      factor=zoom_factor, fixed_au=fixed_au,
                                      pix_as=sky_pix_as, distance_pc=distance_pc)
    szd = sky_data[sy0:sy1, sx0:sx1]
    used_sky_norm = sky_norm if sky_norm is not None else make_norm(szd)
    im0 = ax_sky.imshow(szd, origin="lower", cmap=cmap, norm=used_sky_norm, extent=[sx0, sx1, sy0, sy1])
    if fit_valid:
        draw_ellipse_on_ax(ax_sky, sky_cx, sky_cy, sk_Rmaj_pix, sk_Rmin_pix, mpl_angle)
    ax_sky.axhline(sky_cy, color="white", lw=0.4, alpha=0.35)
    ax_sky.axvline(sky_cx, color="white", lw=0.4, alpha=0.35)
    ax_sky.set_xlim(sx0, sx1)
    ax_sky.set_ylim(sy0, sy1)
    add_AU_ticks(ax_sky, sky_cx, sky_cy, sx0, sx1, sy0, sy1, sky_pix_as, distance_pc)
    ax_sky.set_xlabel(f"ΔRA (AU)  [d = {distance_pc} pc]", color="white")
    ax_sky.set_ylabel("ΔDec (AU)", color="white")
    ax_sky.set_title(f"{prefix}Skymodel  -- zoomed ({zoom_label})", color="white")
    ax_sky.legend(loc="upper right", facecolor="#1a1a1a", edgecolor="#555", labelcolor="white", fontsize=6)
    add_colorbar(fig, im0, ax_sky, sky_hdr.get("BUNIT", "Jy/pixel"))

    # Panel 1: CASA pbcor zoomed
    px0, px1, py0, py1 = zoom_bounds(pb_cx, pb_cy, pb_Rmaj_pix, pb_nx, pb_ny,
                                      factor=zoom_factor, fixed_au=fixed_au,
                                      pix_as=pb_pix_as, distance_pc=distance_pc)
    pzd = pb_data[py0:py1, px0:px1]
    used_obs_norm = obs_norm if obs_norm is not None else make_norm(pzd)
    im1 = ax_obs.imshow(pzd, origin="lower", cmap=cmap, norm=used_obs_norm, extent=[px0, px1, py0, py1])
    if fit_valid:
        draw_ellipse_on_ax(ax_obs, pb_cx, pb_cy, pb_Rmaj_pix, pb_Rmin_pix, mpl_angle)
    ax_obs.axhline(pb_cy, color="white", lw=0.4, alpha=0.35)
    ax_obs.axvline(pb_cx, color="white", lw=0.4, alpha=0.35)
    ax_obs.set_xlim(px0, px1)
    ax_obs.set_ylim(py0, py1)
    add_AU_ticks(ax_obs, pb_cx, pb_cy, px0, px1, py0, py1, pb_pix_as, distance_pc)
    ax_obs.set_xlabel(f"ΔRA (AU)  [d = {distance_pc} pc]", color="white")
    ax_obs.set_ylabel("ΔDec (AU)", color="white")
    ax_obs.set_title(f"{prefix}CASA Observation  -- zoomed ({zoom_label})", color="white")
    ax_obs.legend(loc="upper right", facecolor="#1a1a1a", edgecolor="#555", labelcolor="white")
    add_colorbar(fig, im1, ax_obs, pb_hdr.get("BUNIT", "Jy/beam"))

    header_line = f"{title_prefix}\n" if title_prefix else ""
    if fit_valid:
        info = (
            header_line +
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
        ax_obs.text(0.02, 0.98, header_line + f"snap {snapshot}  |  {field}  |  axis {axis}\nFit failed -- no Gaussian parameters",
                    transform=ax_obs.transAxes, va="top", ha="left", color="orange", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#111", edgecolor="orange", alpha=0.88))

    # Panel 2: Residual zoomed
    rx0, rx1, ry0, ry1 = zoom_bounds(res_cx, res_cy, res_Rmaj_pix, res_nx, res_ny,
                                      factor=zoom_factor, fixed_au=fixed_au,
                                      pix_as=res_pix_as, distance_pc=distance_pc)
    rzd = res_data[ry0:ry1, rx0:rx1]
    used_res_norm = res_norm if res_norm is not None else make_residual_norm(rzd)
    im2 = ax_res.imshow(rzd, origin="lower", cmap="RdBu_r", norm=used_res_norm, extent=[rx0, rx1, ry0, ry1])
    if fit_valid:
        draw_ellipse_on_ax(ax_res, res_cx, res_cy, res_Rmaj_pix, res_Rmin_pix, mpl_angle, color="black", ls="--")
    ax_res.axhline(res_cy, color="gray", lw=0.4, alpha=0.35)
    ax_res.axvline(res_cx, color="gray", lw=0.4, alpha=0.35)
    ax_res.set_xlim(rx0, rx1)
    ax_res.set_ylim(ry0, ry1)
    add_AU_ticks(ax_res, res_cx, res_cy, rx0, rx1, ry0, ry1, res_pix_as, distance_pc)
    ax_res.set_xlabel(f"ΔRA (AU)  [d = {distance_pc} pc]", color="white")
    ax_res.set_ylabel("ΔDec (AU)", color="white")
    ax_res.set_title(f"{prefix}imfit Residual  -- zoomed ({zoom_label})", color="white")
    ax_res.legend(loc="upper right", facecolor="#1a1a1a", edgecolor="#555", labelcolor="white", fontsize=6)
    add_colorbar(fig, im2, ax_res, "Jy/beam  (obs - model)")

    return {
        "snapshot": snapshot, "field": field, "axis": axis,
        "distance_pc": distance_pc, "fit_valid": fit_valid,
        "sky_norm": used_sky_norm, "obs_norm": used_obs_norm, "res_norm": used_res_norm,
        "sky_extent": (sx0, sx1, sy0, sy1), "pb_extent": (px0, px1, py0, py1),
        "res_extent": (rx0, rx1, ry0, ry1),
    }


# ---------------------------------------------------------------------------
# Three-panel plot
# ---------------------------------------------------------------------------
def plot_three_panel(pbcor_fpath, fit, skymodel_dir, residual_dir,
                      zoom_factor=4, cmap="jet", dpi=130,
                      save=True, out_dir="figures", savefig=None, show=False,
                      suffix="", fixed_au=None):
    """Render one row of three zoomed panels for a single fitted disk.

      [0] Skymodel (zoomed)
      [1] CASA pbcor observation (zoomed)
      [2] imfit residual image (zoomed, diverging colormap)

    The Gaussian ellipse from ``fit`` is overlaid on all three panels.
    All the actual loading/drawing happens in ``render_disk_row`` -- this
    function just sets up the figure/axes and handles saving.

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
    show          : bool -- whether to display the figure inline (e.g. in a
                    notebook) before closing it
    suffix        : str  -- "" for the thin pipeline (default, unchanged
                    behaviour), "_SKIRT" to load the SKIRT-variant skymodel
                    and residual files alongside ``pbcor_fpath``.
    fixed_au      : float or None -- fixed physical zoom half-width in AU
                    shared by all three panels (see ``zoom_bounds``).
                    Default None preserves the original Rmaj-relative zoom.

    Returns
    -------
    str or None -- the path the figure was saved to, or None if not saved.
    """

    fig, axes = plt.subplots(1, 3, figsize=(24, 6), gridspec_kw={"wspace": 0.35})
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes:
        style_ax(ax)
    ax_sky, ax_obs, ax_res = axes

    info = render_disk_row(fig, ax_sky, ax_obs, ax_res, pbcor_fpath, fit,
                            skymodel_dir, residual_dir, zoom_factor=zoom_factor,
                            cmap=cmap, suffix=suffix, fixed_au=fixed_au)
    if info is None:
        plt.close(fig)
        return None
    snapshot, field, axis = info["snapshot"], info["field"], info["axis"]

    # -- Save --------------------------------------------------------------
    plt.tight_layout()
    saved_path = None
    if savefig:
        os.makedirs(os.path.dirname(savefig) or ".", exist_ok=True)
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        saved_path = savefig
    elif save and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_fname = f"snap{snapshot}_{field}_axis{axis}{suffix}_three_panel.png"
        saved_path = os.path.join(out_dir, out_fname)
        fig.savefig(saved_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)

    if saved_path:
        print(f"Saved: {saved_path}")
    return saved_path


# ---------------------------------------------------------------------------
# Three-panel stack (multiple snapshots, one row each)
# ---------------------------------------------------------------------------
def plot_three_panel_stack(snapshots, field, axis, results, pbcor_dir, skymodel_dir, residual_dir,
                            zoom_factor=3, cmap="jet", vmin_pct=0, vmax_pct=99.5, log_scale=False,
                            dpi=130, save=True, out_dir="figures", savefig=None, show=False,
                            mass_dict=None, df=None, suffix="", fixed_au=None):
    """Stack multiple snapshots as rows of three zoomed panels each:

      [0] Skymodel (zoomed)
      [1] CASA pbcor observation (zoomed)
      [2] imfit residual image (zoomed)

    Parameters
    ----------
    snapshots    : list of str -- snapshot numbers, e.g. ['170','171',...,'175']
    field        : str  -- 'Orion' or 'Perseus'
    axis         : str  -- 'x', 'y', or 'z'
    results      : dict -- fitting_results.json contents (master_dict)
    pbcor_dir    : str  -- folder containing stage-2 pbcor FITS images
    skymodel_dir : str  -- folder containing the stage-1 skymodel FITS files
    residual_dir : str  -- folder containing the stage-3 residual FITS files
    zoom_factor  : int  -- half-width of zoom in units of Rmaj
    save         : bool -- whether to save a PNG at all
    out_dir      : str  -- folder to save into when ``savefig`` is not given
    savefig      : str or None -- if given, save the PNG to this exact path
                   (overrides ``out_dir``/auto-generated naming)
    show         : bool -- whether to display the figure inline before closing it
    mass_dict    : dict or None -- optional {'snapshot_<id>': [..., true_mass_1e8, ...]}
                   used to annotate the true dust mass (κ=1e-8 entry, index 2)
    df           : DataFrame or None -- optional table with columns
                   snapshot/field/axis/mass_fit_Msun, used to annotate the fitted mass
    suffix       : str  -- "" for thin (default, unchanged behaviour), "_SKIRT"
                   to stack the SKIRT variant instead.
    fixed_au     : float or None -- fixed physical zoom half-width in AU,
                   identical across all rows/panels. Default None preserves
                   the original Rmaj-relative zoom.

    Returns
    -------
    str or None -- the path the figure was saved to, or None if not saved.
    """

    n_rows = len(snapshots)
    fig, axes = plt.subplots(n_rows, 3, figsize=(24, 10 * n_rows),
                              gridspec_kw={"wspace": 0.35, "hspace": 0.4})
    fig.patch.set_facecolor("#0d0d0d")

    # Ensure axes is always 2D even if n_rows == 1
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, snapshot in enumerate(snapshots):

        ax_sky, ax_obs, ax_res = axes[row_idx]
        for ax in [ax_sky, ax_obs, ax_res]:
            style_ax(ax)

        distance_pc = DISTANCES_PC.get(field, 400)

        # -- Load pbcor ------------------------------------------------------
        pbcor_fname = f"ALMA_snapshot_{snapshot}_axis_{axis}_{field}_sim_observed_pbcor{suffix}.fits"
        pbcor_fpath = os.path.join(pbcor_dir, pbcor_fname)
        if not os.path.exists(pbcor_fpath):
            print(f"[skip] pbcor not found: {pbcor_fname}")
            for ax in [ax_sky, ax_obs, ax_res]:
                ax.text(0.5, 0.5, f"snap {snapshot}\nnot found",
                        transform=ax.transAxes, color="red",
                        ha="center", va="center")
            continue
        with fits.open(pbcor_fpath) as hdul:
            pb_hdr = hdul[0].header
            pb_data = hdul[0].data.squeeze()
        pb_ny, pb_nx = pb_data.shape
        pb_pix_as = get_pixel_scale_arcsec(pb_hdr)
        pb_cx, pb_cy = pb_nx / 2.0, pb_ny / 2.0

        # -- Load skymodel -----------------------------------------------------
        sky_fname = f"snapshot_{snapshot}_{field}_flux_map_ALMA_axis_{axis}{suffix}.fits"
        sky_fpath = os.path.join(skymodel_dir, sky_fname)
        if not os.path.exists(sky_fpath):
            print(f"[skip] skymodel not found: {sky_fname}")
            continue
        with fits.open(sky_fpath) as hdul:
            sky_hdr = hdul[0].header
            sky_data = hdul[0].data.squeeze()
        sky_ny, sky_nx = sky_data.shape
        sky_pix_as = get_pixel_scale_arcsec(sky_hdr)
        sky_cx, sky_cy = sky_nx / 2.0, sky_ny / 2.0

        # -- Load residual -------------------------------------------------------
        res_fname = f"ALMA_snapshot_{snapshot}_axis_{axis}_{field}_sim_observed_pbcor{suffix}_residual.fits"
        res_fpath = os.path.join(residual_dir, res_fname)
        if not os.path.exists(res_fpath):
            print(f"[skip] residual not found: {res_fname}")
            continue
        with fits.open(res_fpath) as hdul:
            res_hdr = hdul[0].header
            res_data = hdul[0].data.squeeze()
        res_ny, res_nx = res_data.shape
        res_pix_as = get_pixel_scale_arcsec(res_hdr)
        res_cx, res_cy = res_nx / 2.0, res_ny / 2.0

        # -- Fit parameters ---------------------------------------------------------
        fit = results.get(snapshot, {}).get(field, {}).get(axis, {})
        Rmaj_as = fit.get("Rmaj")
        Rmin_as = fit.get("Rmin")
        pa_deg = fit.get("pa")
        inc = fit.get("inc")
        r_AU_T = fit.get("radius_AU_Tobin")
        flux = fit.get("flux")
        snr = fit.get("snr")
        prf = fit.get("peak_residual_fraction")
        peak_res = fit.get("peak_residual")

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
            print(f"[warn] NaN fit: snap {snapshot} | {field} | axis {axis}")
            pb_Rmaj_pix = pb_nx * 0.1
            sk_Rmaj_pix = sky_nx * 0.1
            res_Rmaj_pix = res_nx * 0.1
            mpl_angle = 0

        # -- Mass lookup ---------------------------------------------------------
        fitted_mass = np.nan
        true_mass = np.nan

        if df is not None:
            row = df[(df["snapshot"] == snapshot) &
                     (df["field"] == field) &
                     (df["axis"] == axis)]
            if not row.empty:
                fitted_mass = row["mass_fit_Msun"].values[0]

        if mass_dict is not None:
            key = f"snapshot_{snapshot}"
            if key in mass_dict:
                true_mass = mass_dict[key][2]  # index 2 = 1e-8 opacity

        # -- Panel 0: Skymodel ----------------------------------------------------
        sx0, sx1, sy0, sy1 = zoom_bounds(sky_cx, sky_cy, sk_Rmaj_pix, sky_nx, sky_ny,
                                          factor=zoom_factor, fixed_au=fixed_au,
                                          pix_as=sky_pix_as, distance_pc=distance_pc)
        szd = sky_data[sy0:sy1, sx0:sx1]
        im0 = ax_sky.imshow(szd, origin="lower", cmap=cmap,
                             norm=make_norm(szd, vmin_pct=vmin_pct, vmax_pct=vmax_pct, log_scale=log_scale),
                             extent=[sx0, sx1, sy0, sy1])
        if fit_valid:
            draw_ellipse_on_ax(ax_sky, sky_cx, sky_cy, sk_Rmaj_pix, sk_Rmin_pix, mpl_angle)
        ax_sky.axhline(sky_cy, color="white", lw=0.4, alpha=0.35)
        ax_sky.axvline(sky_cx, color="white", lw=0.4, alpha=0.35)
        ax_sky.set_xlim(sx0, sx1)
        ax_sky.set_ylim(sy0, sy1)
        add_AU_ticks(ax_sky, sky_cx, sky_cy, sx0, sx1, sy0, sy1, sky_pix_as, distance_pc)
        ax_sky.set_xlabel(f"ΔRA (AU)  [d = {distance_pc} pc]", color="white")
        ax_sky.set_ylabel("ΔDec (AU)", color="white")
        ax_sky.set_title(f"snap {snapshot} — Skymodel", color="white")
        add_colorbar(fig, im0, ax_sky, sky_hdr.get("BUNIT", "Jy/pixel"))

        # -- Panel 1: CASA pbcor ---------------------------------------------------
        px0, px1, py0, py1 = zoom_bounds(pb_cx, pb_cy, pb_Rmaj_pix, pb_nx, pb_ny,
                                          factor=zoom_factor, fixed_au=fixed_au,
                                          pix_as=pb_pix_as, distance_pc=distance_pc)
        pzd = pb_data[py0:py1, px0:px1]
        im1 = ax_obs.imshow(pzd, origin="lower", cmap=cmap,
                             norm=make_norm(pzd, vmin_pct=vmin_pct, vmax_pct=vmax_pct, log_scale=log_scale),
                             extent=[px0, px1, py0, py1])
        if fit_valid:
            draw_ellipse_on_ax(ax_obs, pb_cx, pb_cy, pb_Rmaj_pix, pb_Rmin_pix, mpl_angle)
        ax_obs.axhline(pb_cy, color="white", lw=0.4, alpha=0.35)
        ax_obs.axvline(pb_cx, color="white", lw=0.4, alpha=0.35)
        ax_obs.set_xlim(px0, px1)
        ax_obs.set_ylim(py0, py1)
        add_AU_ticks(ax_obs, pb_cx, pb_cy, px0, px1, py0, py1, pb_pix_as, distance_pc)
        ax_obs.set_xlabel(f"ΔRA (AU)  [d = {distance_pc} pc]", color="white")
        ax_obs.set_ylabel("ΔDec (AU)", color="white")
        ax_obs.set_title(f"snap {snapshot} — CASA Observation", color="white")
        add_colorbar(fig, im1, ax_obs, pb_hdr.get("BUNIT", "Jy/beam"))

        # Annotation box
        if fit_valid:
            info = (
                f"Rmaj = {Rmaj_as:.3f}\"  ({Rmaj_AU / 2:.0f} AU)\n"
                f"Rmin = {Rmin_as:.3f}\"  ({Rmin_AU / 2:.0f} AU)\n"
                f"PA = {pa_deg:.1f}°    i = {inc:.1f}°\n"
                f"Flux = {flux:.3e} Jy\n"
                f"R_disk (Tobin) = {r_AU_T:.1f} AU\n"
                f"snr = {snr:.1f}\n"
                f"peak residual = {peak_res:.3e} Jy  ({prf:.1%} of peak)\n"
                f"─────────────────────────\n"
                f"Fitted mass  = {fitted_mass:.3e} M☉\n"
                f"True mass    = {true_mass:.3e} M☉  (κ=1e-8)\n"
                f"Ratio fit/true = {fitted_mass / true_mass:.2f}"
                if not np.isnan(fitted_mass) and not np.isnan(true_mass) and true_mass != 0
                else f"Fitted mass  = {fitted_mass:.3e} M☉\nTrue mass    = N/A"
            )
            color, edge = "white", "cyan"
        else:
            info = "Fit failed — no Gaussian parameters"
            color, edge = "orange", "orange"
        ax_obs.text(0.02, 0.98, info, transform=ax_obs.transAxes,
                    va="top", ha="left", color=color, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#111", edgecolor=edge, alpha=0.88))

        # -- Panel 2: Residual ----------------------------------------------------
        rx0, rx1, ry0, ry1 = zoom_bounds(res_cx, res_cy, res_Rmaj_pix, res_nx, res_ny,
                                          factor=zoom_factor, fixed_au=fixed_au,
                                          pix_as=res_pix_as, distance_pc=distance_pc)
        rzd = res_data[ry0:ry1, rx0:rx1]
        im2 = ax_res.imshow(rzd, origin="lower", cmap="RdBu_r", norm=make_residual_norm(rzd),
                             extent=[rx0, rx1, ry0, ry1])
        if fit_valid:
            draw_ellipse_on_ax(ax_res, res_cx, res_cy, res_Rmaj_pix, res_Rmin_pix, mpl_angle, color="black", ls="--")
        ax_res.axhline(res_cy, color="gray", lw=0.4, alpha=0.35)
        ax_res.axvline(res_cx, color="gray", lw=0.4, alpha=0.35)
        ax_res.set_xlim(rx0, rx1)
        ax_res.set_ylim(ry0, ry1)
        add_AU_ticks(ax_res, res_cx, res_cy, rx0, rx1, ry0, ry1, res_pix_as, distance_pc)
        ax_res.set_xlabel(f"ΔRA (AU)  [d = {distance_pc} pc]", color="white")
        ax_res.set_ylabel("ΔDec (AU)", color="white")
        ax_res.set_title(f"snap {snapshot} — imfit Residual", color="white")
        add_colorbar(fig, im2, ax_res, "Jy/beam  (obs − model)")

    # -- Save / show --------------------------------------------------------------
    plt.tight_layout()
    saved_path = None
    if savefig:
        os.makedirs(os.path.dirname(savefig) or ".", exist_ok=True)
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        saved_path = savefig
    elif save and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_fname = f"stack_{snapshots[0]}_{snapshots[-1]}_{field}_axis{axis}{suffix}.png"
        saved_path = os.path.join(out_dir, out_fname)
        fig.savefig(saved_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)

    if saved_path:
        print(f"Saved: {saved_path}")
    return saved_path
