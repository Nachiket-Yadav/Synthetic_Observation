"""
plot_surface_density.py
========================
Paper figure: disk surface density vs. disk radius, colour/marker-coded by
disk-size category, with shaded "optically thick" / "optically thin"
reference bands.

Reads directly from ``disks_synthetic_obs_subset.xlsx`` (sheet
``synthetic_obs_disks``) -- nothing about a specific disk is hardcoded here,
so re-running after the spreadsheet is updated (new disks, corrected
masses/radii, etc.) reproduces the figure with the new values automatically.

See ``figures_paper/SURFACE_DENSITY_PLOT.md`` for the full spec, the
rationale behind each choice, the short-name lookup table, and the caption
text -- that file is what we iterate against between sessions instead of
re-deriving these choices from scratch each time. Update BOTH files together
when the design changes.
"""
from __future__ import annotations

import os

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plotting_utils import _paper_rc, style_ax_paper

# ---------------------------------------------------------------------------
# Config -- the knobs to turn when iterating (see SURFACE_DENSITY_PLOT.md)
# ---------------------------------------------------------------------------
XLSX_PATH = "disks_synthetic_obs_subset.xlsx"
SHEET_NAME = "synthetic_obs_disks"
OUT_DIR = "figures_paper"
OUT_STEM = "disk_surface_density_vs_radius"

FIG_WIDTH_IN = 7.1     # AASTeX two-column full text width
FIG_HEIGHT_IN = 5.2
FONT_SIZE_PT = 15      # 1.5x this repo's normal 10pt AASTeX figure convention
                       # (plotting_utils._paper_rc / plot_gallery default) --
                       # bumped per reviewer feedback that labels read too
                       # small next to the paper's body text.
DPI = 300

X_LIM = (10, 600)

# Optically-thick / optically-thin reference bands (M_sun / AU^2). Per Marc's
# note: 1e-4 to 1e-5 = optically thick, 1e-5 to 1e-6 = optically thin.
THICK_HI, THICK_LO = 1e-4, 1e-5
THIN_HI, THIN_LO = 1e-5, 1e-6
BAND_THICK_COLOR = "#e8734a"
BAND_THIN_COLOR = "#4a86c8"

# Colour per "Disk size" category (column in the spreadsheet) -- this is the
# magnitude/grouping channel. Kept as a *second*, separate legend from the
# per-disk shapes below, so identity (shape) and size category (colour)
# don't collide into one over-loaded encoding.
SIZE_STYLE = {
    "small":      dict(color="#2f6fb0"),
    "medium":     dict(color="#e5a51c"),
    "large":      dict(color="#1f9e77"),
    "very large": dict(color="#d35400"),
}
SIZE_ORDER = list(SIZE_STYLE)  # small, medium, large, very large
MARKER_SIZE = 90

# Legend sizing -- deliberately smaller than the base figure font (which is
# already 1.5x the paper's normal AASTeX text size) so the legend box reads
# as a compact reference, not another headline element on the plot.
LEGEND_FONT_SIZE = FONT_SIZE_PT * 0.6
LEGEND_MARKER_SIZE = 6

# Per-disk marker shape -- this is the identity channel (see
# SURFACE_DENSITY_PLOT.md's naming-scheme table). Every disk gets its own
# shape so it's identifiable straight off the plot via the "Disk" legend
# (labelled with the short name, never the snapshot number), on top of the
# colour already grouping it by size category. Keyed by snapshot # from the
# spreadsheet so a re-run stays in sync if the sheet is edited.
DISK_META = {
    169: dict(short="OAHcr_S1a", marker="*"),
    171: dict(short="OAHcr_S1b", marker="h"),
    106: dict(short="ID_B1",     marker="o"),
    307: dict(short="OA_S1",     marker="s"),
    319: dict(short="OAH_S1",    marker="^"),
    346: dict(short="OAH_B2",    marker="v"),
    417: dict(short="OAH_B3",    marker="D"),
    320: dict(short="OAH_B1a",   marker="P"),
    323: dict(short="OAH_B1b",   marker="X"),
}


def load_disks(xlsx_path=XLSX_PATH, sheet_name=SHEET_NAME):
    """Read the disk table, dropping the trailing notes rows (identified by
    a non-numeric Snapshot #)."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df = df[pd.to_numeric(df["Snapshot #"], errors="coerce").notna()].copy()
    df["Snapshot #"] = df["Snapshot #"].astype(int)
    return df


def make_plot(df, out_stem=None):
    """Build the figure and save it as both .png and .pdf.

    out_stem : optional path (without extension) overriding
        ``OUT_DIR/OUT_STEM`` -- ``.png``/``.pdf`` are appended to it.
    Returns the list of saved paths.
    """
    with plt.rc_context(_paper_rc(FONT_SIZE_PT)):
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
        fig.patch.set_facecolor("white")
        style_ax_paper(ax)

        # -- optically-thick / optically-thin shaded bands ------------------
        ax.axhspan(THICK_LO, THICK_HI, color=BAND_THICK_COLOR, alpha=0.12, zorder=0)
        ax.axhspan(THIN_LO, THIN_HI, color=BAND_THIN_COLOR, alpha=0.12, zorder=0)
        for y in (THICK_HI, THICK_LO, THIN_LO):
            ax.axhline(y, color="black", ls="--", lw=1.1, zorder=1)

        # Vertical geometric-mean placement (not just below the band's top
        # edge) keeps this clear of markers that sit high in the thick band
        # (e.g. the two small/compact disks near 5-6e-5).
        blended = ax.get_yaxis_transform()  # x = axes fraction, y = data
        ax.text(0.03, (THICK_HI * THICK_LO) ** 0.5, "optically thick", color="#a8461c",
                 style="italic", ha="left", va="center", transform=blended)
        ax.text(0.03, (THIN_HI * THIN_LO) ** 0.5, "optically thin", color="#2f5f8f",
                 style="italic", ha="left", va="center", transform=blended)

        # -- one point per disk: shape = identity, colour = size category --
        # No per-point text labels (short disk names go in the two legends
        # below, and the caption/table -- never as annotations on the plot
        # itself -- see SURFACE_DENSITY_PLOT.md's naming-scheme table).
        for _, row in df.iterrows():
            snap = int(row["Snapshot #"])
            meta = DISK_META.get(snap)
            if meta is None:
                continue  # spreadsheet row not in the documented naming table yet
            color = SIZE_STYLE[row["Disk size"]]["color"]
            ax.scatter(row["Disk radius (AU)"], row["Surface density (M_sun/AU^2)"],
                       s=MARKER_SIZE, marker=meta["marker"], facecolor=color,
                       edgecolor="black", linewidth=0.8, zorder=3)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*X_LIM)
        ax.set_xlabel("Disk radius (AU)")
        ax.set_ylabel(r"Surface density (M$_{\odot}$ / AU$^2$)")

        # Shape -> short disk name (identity channel), grouped by size
        # category then chronologically, matching SURFACE_DENSITY_PLOT.md.
        # No separate "Disk size" colour legend -- one compact legend only.
        ordered_snaps = sorted(DISK_META, key=lambda s: (
            SIZE_ORDER.index(df.loc[df["Snapshot #"] == s, "Disk size"].iloc[0]), s))
        disk_handles = [
            Line2D([0], [0], marker=DISK_META[s]["marker"], linestyle="None",
                    markerfacecolor=SIZE_STYLE[df.loc[df["Snapshot #"] == s, "Disk size"].iloc[0]]["color"],
                    markeredgecolor="black", markersize=LEGEND_MARKER_SIZE, label=DISK_META[s]["short"])
            for s in ordered_snaps
        ]
        ax.legend(handles=disk_handles, title="Disk", loc="upper right", ncol=2,
                   frameon=True, facecolor="white", edgecolor="black",
                   fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE,
                   handletextpad=0.3, columnspacing=0.8, labelspacing=0.3,
                   borderpad=0.4, handlelength=1.2)

        plt.tight_layout()
        stem = out_stem if out_stem else os.path.join(OUT_DIR, OUT_STEM)
        os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
        saved = []
        for ext in ("png", "pdf"):
            path = f"{stem}.{ext}"
            fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
            saved.append(path)
        plt.close(fig)
    return saved


if __name__ == "__main__":
    disks = load_disks()
    paths = make_plot(disks)
    for p in paths:
        print(f"Saved: {p}")
