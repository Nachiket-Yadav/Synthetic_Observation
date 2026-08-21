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
import warnings

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


def frame_half_width_au(nx, pix_as, distance_pc):
    """Physical half-width (AU) of a square image frame: half of nx across.

    Used on the skymodel, pbcor, and residual frames alike to find the
    tightest shared crop -- the smallest of the three native fields of view
    -- so a three-panel row can be zoomed to one shared physical scale with
    every panel filled edge-to-edge by real data, rather than the widest
    frame's footprint leaving blank margin on the narrower ones.
    """
    return (nx / 2.0) * pix_as * distance_pc


# Back-compat alias: this used to only ever be called on the skymodel frame.
skymodel_half_width_au = frame_half_width_au


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
                 fixed_au=None, pix_as=None, distance_pc=400.0, cap_au=None):
    """VIEW bounds (x0, x1, y0, y1) for a zoomed cutout, in this image's own
    pixel units -- deliberately NOT clamped to the image's actual
    [0, nx] x [0, ny] pixel range. Use ``clip_to_frame`` to get the separate,
    clamped bounds for slicing/plotting the real pixel data.

    Keeping the two separate is what lets every panel in a row be displayed
    at the same physical scale even when one panel's native field of view is
    physically smaller than another's (e.g. a small tclean pbcor frame vs. a
    much wider skymodel image -- the pbcor frame can be narrower than the
    skymodel's own footprint, not just wider as ``cap_au`` alone assumes).
    A panel whose frame doesn't reach the shared view just shows blank
    margin past its own edge rather than being more zoomed-in than the rest
    of the row. Callers that need pixel-array-safe bounds (e.g. for slicing)
    must clip these via ``clip_to_frame`` before indexing.

    fixed_au : fixed PHYSICAL half-width in AU, identical across panels and
        rows -- required for like-for-like comparison (e.g. thin vs. SKIRT,
        which fit very different Rmaj for the same disk). Requires
        ``pix_as`` (this image's own arcsec/pixel scale) to also be given.
        Falls back to ``factor * Rmaj_pix`` with a 30-PIXEL floor, which is
        scale-dependent (e.g. 26 AU on a 0.86 AU/pix skymodel vs. 360 AU on
        a 12 AU/pix pbcor image of the same disk) -- fine within one image,
        not comparable across images of different pixel scale.
    cap_au : optional PHYSICAL half-width CAP in AU applied ONLY to the
        factor*Rmaj path. The view can zoom in below the cap (zoom_factor
        still works) but is never wider than cap_au -- this is what keeps the
        pbcor/residual panels from showing tclean's empty border past the
        skymodel footprint. Ignored when fixed_au is given (fixed_au is
        already an explicit absolute half-width).
    """
    if fixed_au is not None and pix_as is not None:
        r = int(round(fixed_au / (pix_as * distance_pc)))
    else:
        # Small pixel guard only (was 30, which is ~360 AU on a 12 AU/pix pbcor
        # -- larger than the whole model footprint, so it pinned the pbcor view
        # and made zoom_factor a no-op while also desyncing panel scales). A
        # small floor keeps factor*Rmaj physically consistent across panels.
        r = max(int(Rmaj_pix * factor), 5)
        if cap_au is not None and pix_as is not None:
            r = min(r, int(round(cap_au / (pix_as * distance_pc))))
    return (int(cx) - r, int(cx) + r, int(cy) - r, int(cy) + r)


def clip_to_frame(bounds, nx, ny):
    """Clip VIEW bounds from ``zoom_bounds`` down to an image's actual
    [0, nx] x [0, ny] pixel range, for array slicing / imshow ``extent``.
    See ``zoom_bounds`` for why the two are kept separate."""
    x0, x1, y0, y1 = bounds
    return max(x0, 0), min(x1, nx), max(y0, 0), min(y1, ny)


def add_AU_ticks(ax, cx, cy, x0, x1, y0, y1, pix_scale, distance_pc, n_ticks=5,
                  label_color="white"):
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
    # The tiny 1e-6 nudge absorbs float error in hw_AU (e.g. a panel capped
    # to a shared physical footprint can land a hair under the true edge,
    # e.g. 599.9999999999759 instead of 600.0) so it doesn't silently drop
    # the outermost tick that a sibling panel -- capped to the same physical
    # value but landing a hair over it after its own pixel rounding -- keeps,
    # which would make otherwise identically-scaled panels look mismatched.
    kx = int(np.floor(hw_x_AU / step_AU + 1e-6)); xtick_AU = np.arange(-kx, kx + 1) * step_AU
    ky = int(np.floor(hw_y_AU / step_AU + 1e-6)); ytick_AU = np.arange(-ky, ky + 1) * step_AU
    pix_per_AU = 1.0 / (pix_scale * distance_pc)
    ax.set_xticks(xtick_AU * pix_per_AU + cx)
    ax.set_xticklabels([f"{v:.0f}" for v in xtick_AU], color=label_color)
    ax.set_yticks(ytick_AU * pix_per_AU + cy)
    ax.set_yticklabels([f"{v:.0f}" for v in ytick_AU], color=label_color)


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
                     suffix="", fixed_au=None, crop_to_skymodel=True,
                     sky_norm=None, obs_norm=None, res_norm=None,
                     title_prefix=None, show_legend=True, show_info_box=True):
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
    show_legend             : bool -- whether to draw the per-panel legends
                              (ellipse fit label). Default True.
    show_info_box           : bool -- whether to draw the overlaid text box
                              of fit parameters (Rmaj/Rmin/PA/flux/etc.) on
                              the CASA observation panel. Default True.

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

    # Cap the zoom at the TIGHTEST of the three frames' own native footprints
    # (skymodel, pbcor, residual can each have a different field of view --
    # e.g. a small tclean pbcor frame narrower than the skymodel crop) so
    # every panel in the row shares one physical scale and is filled
    # edge-to-edge by real data, rather than the widest frame's footprint
    # leaving blank margin on the narrower ones. Unlike setting fixed_au,
    # this is only an upper bound -- zoom_factor still zooms IN below it. An
    # explicit fixed_au overrides both.
    cap_au = None
    if fixed_au is None and crop_to_skymodel:
        cap_au = min(
            frame_half_width_au(sky_nx, sky_pix_as, distance_pc),
            frame_half_width_au(pb_nx, pb_pix_as, distance_pc),
            frame_half_width_au(res_nx, res_pix_as, distance_pc),
        )

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

    if fixed_au is not None:
        zoom_label = f"±{fixed_au:.0f} AU"
    elif cap_au is not None:
        zoom_label = f"±{zoom_factor:.2f} x Rmaj (cap ±{cap_au:.0f} AU)"
    else:
        zoom_label = f"±{zoom_factor:.2f} x Rmaj"
    prefix = f"{title_prefix} " if title_prefix else ""

    # Panel 0: Skymodel zoomed
    sx0, sx1, sy0, sy1 = zoom_bounds(sky_cx, sky_cy, sk_Rmaj_pix, sky_nx, sky_ny,
                                      factor=zoom_factor, fixed_au=fixed_au,
                                      pix_as=sky_pix_as, distance_pc=distance_pc,
                                      cap_au=cap_au)
    csx0, csx1, csy0, csy1 = clip_to_frame((sx0, sx1, sy0, sy1), sky_nx, sky_ny)
    szd = sky_data[csy0:csy1, csx0:csx1]
    used_sky_norm = sky_norm if sky_norm is not None else make_norm(szd)
    im0 = ax_sky.imshow(szd, origin="lower", cmap=cmap, norm=used_sky_norm, extent=[csx0, csx1, csy0, csy1])
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
    if show_legend:
        ax_sky.legend(loc="upper right", facecolor="#1a1a1a", edgecolor="#555", labelcolor="white", fontsize=6)
    add_colorbar(fig, im0, ax_sky, sky_hdr.get("BUNIT", "Jy/pixel"))

    # Panel 1: CASA pbcor zoomed
    px0, px1, py0, py1 = zoom_bounds(pb_cx, pb_cy, pb_Rmaj_pix, pb_nx, pb_ny,
                                      factor=zoom_factor, fixed_au=fixed_au,
                                      pix_as=pb_pix_as, distance_pc=distance_pc,
                                      cap_au=cap_au)
    cpx0, cpx1, cpy0, cpy1 = clip_to_frame((px0, px1, py0, py1), pb_nx, pb_ny)
    pzd = pb_data[cpy0:cpy1, cpx0:cpx1]
    used_obs_norm = obs_norm if obs_norm is not None else make_norm(pzd)
    im1 = ax_obs.imshow(pzd, origin="lower", cmap=cmap, norm=used_obs_norm, extent=[cpx0, cpx1, cpy0, cpy1])
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
    if show_legend:
        ax_obs.legend(loc="upper right", facecolor="#1a1a1a", edgecolor="#555", labelcolor="white")
    add_colorbar(fig, im1, ax_obs, pb_hdr.get("BUNIT", "Jy/beam"))

    header_line = f"{title_prefix}\n" if title_prefix else ""
    if not show_info_box:
        pass
    elif fit_valid:
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
                                      pix_as=res_pix_as, distance_pc=distance_pc,
                                      cap_au=cap_au)
    crx0, crx1, cry0, cry1 = clip_to_frame((rx0, rx1, ry0, ry1), res_nx, res_ny)
    rzd = res_data[cry0:cry1, crx0:crx1]
    used_res_norm = res_norm if res_norm is not None else make_residual_norm(rzd)
    im2 = ax_res.imshow(rzd, origin="lower", cmap="RdBu_r", norm=used_res_norm, extent=[crx0, crx1, cry0, cry1])
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
    if show_legend:
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
                      suffix="", fixed_au=None, crop_to_skymodel=True,
                      show_legend=True, show_info_box=True):
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
    show_legend   : bool -- whether to draw the per-panel legends. Default True.
    show_info_box : bool -- whether to draw the overlaid fit-parameter text
                    box on the CASA observation panel. Default True.

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
                            cmap=cmap, suffix=suffix, fixed_au=fixed_au,
                            crop_to_skymodel=crop_to_skymodel, show_legend=show_legend,
                            show_info_box=show_info_box)
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
                            mass_dict=None, df=None, suffix="", fixed_au=None,
                            crop_to_skymodel=True):
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

        # Cap the zoom at the TIGHTEST of the three frames' own native
        # footprints (see render_disk_row) so every panel shares one
        # physical scale and is filled edge-to-edge. Per-row local so an
        # explicit fixed_au still wins.
        cap_au = None
        if fixed_au is None and crop_to_skymodel:
            cap_au = min(
                frame_half_width_au(sky_nx, sky_pix_as, distance_pc),
                frame_half_width_au(pb_nx, pb_pix_as, distance_pc),
                frame_half_width_au(res_nx, res_pix_as, distance_pc),
            )

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
                                          factor=zoom_factor, fixed_au=fixed_au, cap_au=cap_au,
                                          pix_as=sky_pix_as, distance_pc=distance_pc)
        csx0, csx1, csy0, csy1 = clip_to_frame((sx0, sx1, sy0, sy1), sky_nx, sky_ny)
        szd = sky_data[csy0:csy1, csx0:csx1]
        im0 = ax_sky.imshow(szd, origin="lower", cmap=cmap,
                             norm=make_norm(szd, vmin_pct=vmin_pct, vmax_pct=vmax_pct, log_scale=log_scale),
                             extent=[csx0, csx1, csy0, csy1])
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
                                          factor=zoom_factor, fixed_au=fixed_au, cap_au=cap_au,
                                          pix_as=pb_pix_as, distance_pc=distance_pc)
        cpx0, cpx1, cpy0, cpy1 = clip_to_frame((px0, px1, py0, py1), pb_nx, pb_ny)
        pzd = pb_data[cpy0:cpy1, cpx0:cpx1]
        im1 = ax_obs.imshow(pzd, origin="lower", cmap=cmap,
                             norm=make_norm(pzd, vmin_pct=vmin_pct, vmax_pct=vmax_pct, log_scale=log_scale),
                             extent=[cpx0, cpx1, cpy0, cpy1])
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
                                          factor=zoom_factor, fixed_au=fixed_au, cap_au=cap_au,
                                          pix_as=res_pix_as, distance_pc=distance_pc)
        crx0, crx1, cry0, cry1 = clip_to_frame((rx0, rx1, ry0, ry1), res_nx, res_ny)
        rzd = res_data[cry0:cry1, crx0:crx1]
        im2 = ax_res.imshow(rzd, origin="lower", cmap="RdBu_r", norm=make_residual_norm(rzd),
                             extent=[crx0, crx1, cry0, cry1])
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


# ---------------------------------------------------------------------------
# Axes stack (one snapshot/field, one row per projection axis)
# ---------------------------------------------------------------------------
def plot_axes_stack(snapshot, field, results, skymodel_dir, pbcor_dir, residual_dir,
                     axes=("x", "y", "z"), zoom_factor=3, cmap="jet", dpi=130,
                     save=True, out_dir="figures", savefig=None, show=False,
                     suffix="", show_legend=True, show_info_box=True,
                     crop_to_skymodel=True, fixed_au=None):
    """Stack one snapshot/field's projection axes as rows of three zoomed
    panels each:

      [0] Skymodel (zoomed)
      [1] CASA pbcor observation (zoomed)
      [2] imfit residual (zoomed)

    Companion to ``plot_three_panel_stack`` (which stacks *snapshots* for a
    fixed axis) -- this stacks *axes* for a fixed snapshot/field. Every row
    is drawn by ``render_disk_row``, the same helper used by
    ``plot_three_panel`` and ``plot_thin_vs_skirt``, so scale/crop behaviour
    (including the shared physical zoom cap) is identical across all three
    driver scripts.

    Parameters
    ----------
    snapshot, field : str -- e.g. '346', 'Orion'.
    results         : dict -- fitting_results.json contents (master_dict).
    skymodel_dir, pbcor_dir, residual_dir : str -- stage 1/2/3 folders.
    axes            : sequence of str -- projection axes to stack, one row
                      each, in order (default: x, y, z).
    suffix          : str -- "" for thin (default), "_SKIRT" for the SKIRT
                      variant.
    Remaining parameters mirror ``plot_three_panel``.

    A row whose fit or FITS files are missing is rendered as a red "not
    found" placeholder rather than aborting the whole figure. Returns None
    (and saves nothing) only if *no* row could be drawn.

    Returns
    -------
    str or None -- the path the figure was saved to, or None if not saved.
    """
    n_rows = len(axes)
    fig, axes_grid = plt.subplots(n_rows, 3, figsize=(24, 6 * n_rows),
                                   gridspec_kw={"wspace": 0.35, "hspace": 0.4})
    fig.patch.set_facecolor("#0d0d0d")
    if n_rows == 1:
        axes_grid = axes_grid[np.newaxis, :]

    n_drawn = 0
    for row_idx, axis in enumerate(axes):
        ax_sky, ax_obs, ax_res = axes_grid[row_idx]
        for ax in (ax_sky, ax_obs, ax_res):
            style_ax(ax)

        fit = results.get(snapshot, {}).get(field, {}).get(axis)
        pbcor_fname = f"ALMA_snapshot_{snapshot}_axis_{axis}_{field}_sim_observed_pbcor{suffix}.fits"
        pbcor_fpath = os.path.join(pbcor_dir, pbcor_fname)

        if fit is None:
            print(f"[skip] snap {snapshot} | {field} | axis {axis}: no fit in results")
            msg = f"axis {axis}\nno fit"
        elif not os.path.exists(pbcor_fpath):
            print(f"[skip] pbcor not found: {pbcor_fname}")
            msg = f"axis {axis}\npbcor not found"
        else:
            info = render_disk_row(fig, ax_sky, ax_obs, ax_res, pbcor_fpath, fit,
                                    skymodel_dir, residual_dir, zoom_factor=zoom_factor,
                                    cmap=cmap, suffix=suffix, fixed_au=fixed_au,
                                    crop_to_skymodel=crop_to_skymodel,
                                    show_legend=show_legend, show_info_box=show_info_box)
            if info is not None:
                n_drawn += 1
                continue
            msg = f"axis {axis}\nskymodel/residual not found"

        for ax in (ax_sky, ax_obs, ax_res):
            ax.text(0.5, 0.5, msg, transform=ax.transAxes, color="red",
                    ha="center", va="center")

    if n_drawn == 0:
        plt.close(fig)
        print(f"[skip] snap {snapshot} | {field}: nothing drawn")
        return None

    variant = suffix.strip("_") or "thin"
    fig.suptitle(f"snap {snapshot}  |  {field}  |  {variant}", color="white", fontsize=13, y=1.005)

    plt.tight_layout()
    saved_path = None
    if savefig:
        os.makedirs(os.path.dirname(savefig) or ".", exist_ok=True)
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        saved_path = savefig
    elif save and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_fname = f"snap{snapshot}_{field}_axes_{''.join(axes)}_stack{suffix}.png"
        saved_path = os.path.join(out_dir, out_fname)
        fig.savefig(saved_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)

    if saved_path:
        print(f"Saved: {saved_path}")
    return saved_path


# ---------------------------------------------------------------------------
# Paper-quality gallery figures (rows = snapshots, columns = projection axes)
# ---------------------------------------------------------------------------
# Used by ``plot_gallery.py``. Unlike the dark-themed QA figures above, these
# are meant to be dropped straight into a manuscript: white background,
# black text, serif font at roughly AASTeX body-text size, no overlays, and
# every panel in a gallery cropped to one shared physical field of view so
# disk sizes across snapshots/axes are directly comparable by eye.

GALLERY_FILENAME_BUILDERS = {
    "skymodel": lambda snapshot, field, axis, suffix:
        f"snapshot_{snapshot}_{field}_flux_map_ALMA_axis_{axis}{suffix}.fits",
    "pbcor": lambda snapshot, field, axis, suffix:
        f"ALMA_snapshot_{snapshot}_axis_{axis}_{field}_sim_observed_pbcor{suffix}.fits",
}

GALLERY_DEFAULT_BUNIT = {"skymodel": "Jy/pixel", "pbcor": "Jy/beam"}


def _paper_rc(fontsize=10):
    """rcParams for one AASTeX-scale, white-background figure. Applied via
    ``plt.rc_context`` so it never leaks into other (dark-themed) figures."""
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": fontsize,
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": max(fontsize - 1, 6),
        "ytick.labelsize": max(fontsize - 1, 6),
        "legend.fontsize": max(fontsize - 1, 6),
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "text.color": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "savefig.facecolor": "white",
    }


def style_ax_paper(ax):
    """White-background, inward-tick styling for the gallery panels (AAS
    figure convention -- ticks pointing in, drawn on all four sides)."""
    ax.set_facecolor("white")
    ax.tick_params(colors="black", direction="in", top=True, right=True)
    for sp in ax.spines.values():
        sp.set_edgecolor("black")
        sp.set_linewidth(0.8)


def add_colorbar_paper(fig, im, ax, label, fontsize=8):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, fontsize=fontsize, color="black")
    cb.ax.tick_params(labelsize=max(fontsize - 1, 6), colors="black")
    cb.outline.set_edgecolor("black")
    return cb


def _load_gallery_frame(kind, image_dir, snapshot, field, axis, suffix):
    """Load one FITS frame for the gallery. Returns None if the file is
    missing (caller draws a placeholder instead of raising)."""
    build_name = GALLERY_FILENAME_BUILDERS[kind]
    fpath = os.path.join(image_dir, build_name(snapshot, field, axis, suffix))
    if not os.path.exists(fpath):
        print(f"[warn] {kind} not found: {fpath}")
        return None
    with fits.open(fpath) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data.squeeze()
    ny, nx = data.shape
    return {
        "data": data, "header": hdr, "nx": nx, "ny": ny,
        "cx": nx / 2.0, "cy": ny / 2.0,
        "pix_as": get_pixel_scale_arcsec(hdr),
        "bunit": hdr.get("BUNIT", GALLERY_DEFAULT_BUNIT.get(kind, "")),
        "fpath": fpath,
    }


def _gallery_fixed_au(snapshots, field, axes, results, zoom_factor, distance_pc, exclude=None):
    """Shared physical half-width (AU) for a whole gallery, sized to
    ``zoom_factor`` times the largest fitted Rmaj among every snapshot/axis
    in the gallery. Returns None if ``results`` has no usable Rmaj entries
    (caller falls back to the smallest native frame footprint instead).

    ``exclude`` : set of (snapshot, axis) or None -- entries to leave out of
    the FOV sizing (their own panels still render, just cropped to whatever
    FOV the rest of the gallery ends up with). There is deliberately no
    automatic "drop statistical outliers" option here: a large Rmaj is not
    necessarily a bad fit -- it can be a genuinely extended structure (e.g.
    an edge-on disk/envelope spanning most of the frame), and a fit-quality
    based heuristic doesn't reliably separate the two cases either (in this
    pipeline, a spurious large fit and a real extended one have shown up
    with equally good S/N and residual fraction). A single large-but-real
    axis fit for one snapshot can also look identical, numerically, to
    another snapshot's large fit that is corroborated across multiple axes
    and genuinely large. Decide per snapshot by looking at its own
    skymodel/pbcor image, then pass it here explicitly."""
    rmaj_au = []
    exclude = exclude or set()
    for snapshot in snapshots:
        for axis in axes:
            if (snapshot, axis) in exclude:
                continue
            fit = (results or {}).get(snapshot, {}).get(field, {}).get(axis, {}) or {}
            rmaj_as = fit.get("Rmaj")
            if rmaj_as is not None and not (isinstance(rmaj_as, float) and np.isnan(rmaj_as)):
                rmaj_au.append(rmaj_as * distance_pc)
    if not rmaj_au:
        return None
    return zoom_factor * max(rmaj_au)


def _add_gallery_col_labels(fig, axes_grid, col_labels, fontsize=11):
    """Column headers above the top row -- placed from the *drawn* axes
    positions so they land correctly regardless of how tight_layout ends up
    spacing the grid."""
    fig.canvas.draw()
    for j, label in enumerate(col_labels):
        pos = axes_grid[0, j].get_position()
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.012, label,
                  ha="center", va="bottom", fontsize=fontsize, color="black", weight="bold")


def add_row_label(ax, label, fontsize=11):
    """Row identifier as an in-panel annotation (top-left corner) rather
    than a rotated label in the left margin -- the margin approach has to
    dodge the y-axis tick labels, whose width depends on the shared FOV (2
    to 5 digit AU values), so a fixed offset that looks right on one gallery
    overlaps the ticks on another. An in-panel label with a background box
    is legible over any image content and needs no layout bookkeeping."""
    ax.text(0.05, 0.95, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=fontsize, color="black", weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="black", linewidth=0.6, alpha=0.85))


def plot_gallery(snapshots, field, image_dir, kind="skymodel", results=None,
                  axes=("x", "y", "z"), suffix="", scaled=False,
                  zoom_factor=3.0, fixed_au=None, rmaj_exclude=None,
                  cmap="jet", vmin_pct=0, vmax_pct=99.5, log_scale=False,
                  fig_width_in=7.1, row_height_in=2.3, fontsize=10,
                  dpi=300, save=True, out_dir="gallery_figures", savefig=None, show=False,
                  row_labels=None):
    """Paper-quality gallery: one row per snapshot, one column per
    projection axis, for a single field/kind/variant.

    Every panel in the gallery is cropped to the *same* physical field of
    view (in AU) -- sized from ``results`` as ``zoom_factor`` times the
    largest fitted Rmaj anywhere in the gallery, or, if no fit info is
    available, the smallest native frame footprint among the loaded frames
    -- so relative disk sizes are directly comparable by eye across rows and
    columns. Pass ``fixed_au`` to set it explicitly instead.

    Parameters
    ----------
    snapshots   : list of str -- snapshot IDs, one row each, in order.
    field       : str  -- 'Orion' or 'Perseus'.
    image_dir   : str  -- folder containing the FITS files for ``kind``
                  (the ``skymodels`` folder for kind='skymodel', the
                  ``pbcor_imgs`` folder for kind='pbcor').
    kind        : str  -- 'skymodel' (stage-1 sky model) or 'pbcor' (stage-2
                  CASA primary-beam-corrected observation).
    results     : dict or None -- fitting_results.json contents (used only
                  to size the shared field of view via Rmaj; not required).
    suffix      : str  -- "" for thin (default), "_SKIRT" for the SKIRT variant.
    scaled      : bool -- False (default): every panel is normalised to its
                  own data (one colorbar per panel). True: every panel shares
                  one colour norm taken from the brightest snapshot/axis in
                  the gallery, with a single colorbar for the whole figure.
    zoom_factor : float -- shared FOV half-width in units of the largest
                  fitted Rmaj in the gallery (default: 3). Ignored when
                  ``fixed_au`` is given.
    fixed_au    : float or None -- explicit shared physical half-width (AU),
                  overriding the Rmaj-based sizing.
    rmaj_exclude : set of (snapshot, axis) or None -- entries to leave out
                  of the FOV sizing (their panels still render, cropped to
                  whatever FOV the rest of the gallery ends up with). See
                  ``_gallery_fixed_au`` for why this is manual rather than
                  an automatic outlier filter.
    fig_width_in, row_height_in : float -- figure width (AASTeX two-column
                  text width, 7.1 in, by default) and per-row height, inches.
    fontsize    : float -- base font size in points (AASTeX two-column body
                  text is 10 pt; this is also the default here).
    row_labels  : list of str or None -- left-margin row labels, default
                  "snap <id>" for each entry in ``snapshots``.

    Returns
    -------
    str or None -- path the figure was saved to, or None if nothing in the
    gallery could be loaded.
    """
    distance_pc = DISTANCES_PC.get(field, 400)
    axes = list(axes)

    # -- Load every frame once (no cropping yet -- need the shared FOV first) --
    frames = {}
    for snapshot in snapshots:
        for axis in axes:
            frames[(snapshot, axis)] = _load_gallery_frame(kind, image_dir, snapshot, field, axis, suffix)
    loaded = [fr for fr in frames.values() if fr is not None]
    if not loaded:
        print(f"[skip] gallery {kind} | {field}{suffix}: no frames found in {image_dir}")
        return None

    # -- Resolve one shared physical field of view for the whole gallery ------
    if fixed_au is None:
        fixed_au = _gallery_fixed_au(snapshots, field, axes, results, zoom_factor, distance_pc,
                                      exclude=rmaj_exclude)
    if fixed_au is None:
        fixed_au = min(frame_half_width_au(fr["nx"], fr["pix_as"], distance_pc) for fr in loaded)
        print(f"[info] no Rmaj fits available -- shared FOV falls back to the smallest "
              f"native frame footprint (±{fixed_au:.0f} AU)")

    # -- Crop every frame to that shared FOV -----------------------------------
    for fr in loaded:
        x0, x1, y0, y1 = zoom_bounds(fr["cx"], fr["cy"], 0, fr["nx"], fr["ny"],
                                      fixed_au=fixed_au, pix_as=fr["pix_as"], distance_pc=distance_pc)
        cx0, cx1, cy0, cy1 = clip_to_frame((x0, x1, y0, y1), fr["nx"], fr["ny"])
        fr["crop"] = fr["data"][cy0:cy1, cx0:cx1]
        fr["extent"] = (cx0, cx1, cy0, cy1)
        fr["view"] = (x0, x1, y0, y1)

    # -- Resolve the shared colour scale for scaled=True -----------------------
    shared_norm = None
    if scaled:
        def _peak(fr):
            pos = fr["crop"][fr["crop"] > 0]
            return np.percentile(pos, vmax_pct) if pos.size else -np.inf
        brightest = max(loaded, key=_peak)
        shared_norm = make_norm(brightest["crop"], vmin_pct=vmin_pct, vmax_pct=vmax_pct, log_scale=log_scale)

    n_rows, n_cols = len(snapshots), len(axes)
    row_labels = row_labels or [f"snap {s}" for s in snapshots]
    col_labels = [f"axis {a}" for a in axes]

    with plt.rc_context(_paper_rc(fontsize)):
        fig, axes_grid = plt.subplots(
            n_rows, n_cols, squeeze=False,
            figsize=(fig_width_in, row_height_in * n_rows),
            gridspec_kw={"wspace": 0.08 if scaled else 0.4, "hspace": 0.3},
        )
        fig.patch.set_facecolor("white")

        last_im, bunit = None, GALLERY_DEFAULT_BUNIT.get(kind, "")
        for i, snapshot in enumerate(snapshots):
            for j, axis in enumerate(axes):
                ax = axes_grid[i, j]
                style_ax_paper(ax)
                fr = frames[(snapshot, axis)]
                if fr is None:
                    ax.text(0.5, 0.5, "not found", transform=ax.transAxes, color="red",
                            ha="center", va="center", fontsize=max(fontsize - 1, 6))
                    ax.set_xticks([]); ax.set_yticks([])
                    if j == 0:
                        add_row_label(ax, row_labels[i], fontsize=fontsize + 1)
                    continue

                norm = shared_norm if scaled else make_norm(
                    fr["crop"], vmin_pct=vmin_pct, vmax_pct=vmax_pct, log_scale=log_scale)
                cx0, cx1, cy0, cy1 = fr["extent"]
                im = ax.imshow(fr["crop"], origin="lower", cmap=cmap, norm=norm,
                                extent=[cx0, cx1, cy0, cy1])
                last_im, bunit = im, fr["bunit"]
                if j == 0:
                    add_row_label(ax, row_labels[i], fontsize=fontsize + 1)

                x0, x1, y0, y1 = fr["view"]
                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)
                add_AU_ticks(ax, fr["cx"], fr["cy"], x0, x1, y0, y1, fr["pix_as"], distance_pc,
                             label_color="black")

                if i == n_rows - 1:
                    ax.set_xlabel("ΔRA (AU)")
                else:
                    ax.set_xticklabels([])
                if j == 0:
                    ax.set_ylabel("ΔDec (AU)")
                else:
                    ax.set_yticklabels([])
                if not scaled:
                    add_colorbar_paper(fig, im, ax, fr["bunit"], fontsize=max(fontsize - 2, 6))

        right = 0.90 if scaled else 0.98
        with warnings.catch_warnings():
            # fig.colorbar() axes (added per-panel above, for the unscaled case)
            # aren't tracked by tight_layout the way regular Axes are -- benign,
            # tight_layout still spaces everything else correctly around them.
            warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
            fig.tight_layout(rect=[0.0, 0.0, right, 0.96])
        _add_gallery_col_labels(fig, axes_grid, col_labels, fontsize=fontsize + 1)

        if scaled and last_im is not None:
            cbar_ax = fig.add_axes([right + 0.015, 0.12, 0.018, 0.76])
            cb = fig.colorbar(last_im, cax=cbar_ax)
            cb.set_label(bunit, fontsize=fontsize)
            cb.ax.tick_params(labelsize=max(fontsize - 1, 6), colors="black")
            cb.outline.set_edgecolor("black")

        saved_path = None
        if savefig:
            os.makedirs(os.path.dirname(savefig) or ".", exist_ok=True)
            fig.savefig(savefig, dpi=dpi, bbox_inches="tight", facecolor="white")
            saved_path = savefig
        elif save and out_dir:
            os.makedirs(out_dir, exist_ok=True)
            mode = "scaled" if scaled else "unscaled"
            out_fname = f"gallery_{kind}_{field}{suffix}_{mode}.png"
            saved_path = os.path.join(out_dir, out_fname)
            fig.savefig(saved_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        if show:
            plt.show()
        plt.close(fig)

    if saved_path:
        print(f"Saved: {saved_path}")
    return saved_path
