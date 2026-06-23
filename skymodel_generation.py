#!/usr/bin/env python3
"""
skymodel_generation.py
======================

Generate synthetic interferometer "sky model" images from GIZMO/GADGET-style
HDF5 snapshots of protostellar disk simulations.

What this script does
---------------------
For each input snapshot it:

  1. Loads the snapshot with ``yt`` and defines derived dust fields
     (dust temperature, dust mass/density, Planck function B_nu, and the
     dust emissivity at the observing frequency).
  2. Makes line-of-sight projections of the dust emissivity along the x, y
     and z axes, centred on the densest gas cell.
  3. Converts the projected emissivity into a flux-density map (Jy/pixel) as
     it would appear if the disk were placed at the distance of a chosen
     star-forming region (by default Orion at 400 pc and Perseus at 300 pc).
  4. Writes one FITS file per (snapshot, region, axis) combination. These
     FITS files are the "sky models" that are later fed to CASA's
     ``simobserve`` (see ``casa_simulation.py``).

Each output filename follows the pattern expected by the downstream tools:

    <snapshot_name>_<Region>_flux_map_<TELESCOPE>_axis_<axis>.fits
    e.g. snapshot_170_Orion_flux_map_ALMA_axis_x.fits

Usage
-----
Process every snapshot in a directory and write FITS files to ./skymodels:

    python skymodel_generation.py --input-dir /path/to/snapshots --output-dir skymodels

Process a specific list of snapshots:

    python skymodel_generation.py --snapshots snap_170.hdf5 snap_171.hdf5 -o skymodels

Use the VLA configuration instead of ALMA:

    python skymodel_generation.py -i ./data -o ./skymodels --telescope VLA

Run ``python skymodel_generation.py --help`` for the full list of options.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import h5py
import yt
from astropy.io import fits


# ---------------------------------------------------------------------------
# Physical constants (CGS units, wrapped as yt quantities where needed)
# ---------------------------------------------------------------------------
SOLAR_MASS_G = 1.989e33                      # grams in one solar mass
PC_TO_CM = 3.086e18                          # centimetres in one parsec
JY_PER_CGS_FLUX = 1e23                       # 1 Jy = 1e-23 erg/s/cm^2/Hz

H_PLANCK = yt.YTQuantity(6.62607015e-27, "erg*s")   # Planck constant
C_LIGHT = yt.YTQuantity(2.99792458e10, "cm/s")      # speed of light
K_BOLTZ = yt.YTQuantity(1.380649e-16, "erg/K")      # Boltzmann constant


# ---------------------------------------------------------------------------
# Telescope presets: dust opacity and observing frequency.
#
# kappa_nu and nu values follow Williams et al. (2019). Add new instruments
# here and they immediately become valid arguments to --telescope.
# ---------------------------------------------------------------------------
TELESCOPE_PRESETS: Dict[str, Dict[str, float]] = {
    "ALMA": {"kappa_nu": 1.84, "nu_hz": 338e9},   # 3.38 cm^2/g style value at 338 GHz
    "VLA":  {"kappa_nu": 0.33, "nu_hz": 33e9},    # 0.33 cm^2/g at 33 GHz
}


@dataclass
class Config:
    """User-tunable parameters for sky-model generation.

    Attributes
    ----------
    telescope:
        Key into ``TELESCOPE_PRESETS`` ("ALMA" or "VLA"). Sets the dust
        opacity and the observing frequency.
    resolution:
        Pixel dimensions of the projected image, (nx, ny).
    zoom_base:
        Base zoom factor applied to the yt projection. The effective zoom is
        ``zoom_base * box_size`` when the box is larger than 0.1 pc, otherwise
        just ``zoom_base`` (matches the original notebook behaviour).
    number_density_threshold:
        Gas number-density cut (cm^-3) used when summing the "true" disk dust
        mass for a sanity-check printout. Not used for the FITS output itself.
    dust_to_gas:
        Dust-to-gas mass ratio used to derive dust mass/density from gas.
    region_distances_pc:
        Mapping of star-forming region name -> distance in parsecs. One FITS
        file is produced per region.
    axes:
        Projection axes to loop over.
    """

    telescope: str = "ALMA"
    resolution: tuple = (800, 800)
    zoom_base: float = 30.0
    number_density_threshold: float = 1e8
    dust_to_gas: float = 0.01
    region_distances_pc: Dict[str, float] = field(
        default_factory=lambda: {"Orion": 400.0, "Perseus": 300.0}
    )
    axes: tuple = ("x", "y", "z")

    @property
    def kappa_nu(self) -> "yt.YTQuantity":
        return yt.YTQuantity(TELESCOPE_PRESETS[self.telescope]["kappa_nu"], "cm**2/g")

    @property
    def nu(self) -> "yt.YTQuantity":
        return yt.YTQuantity(TELESCOPE_PRESETS[self.telescope]["nu_hz"], "Hz")


# ---------------------------------------------------------------------------
# Derived-field definitions
# ---------------------------------------------------------------------------
def _register_dust_fields(ds, cfg: Config) -> None:
    """Add the derived dust fields needed for the emissivity projection.

    These mirror the field definitions used throughout the project. They are
    defined as closures so that they can capture ``cfg`` (for the dust-to-gas
    ratio, opacity and frequency) without relying on module-level globals.

    Parameters
    ----------
    ds:
        A loaded yt dataset.
    cfg:
        The active :class:`Config`.
    """
    kappa_nu = cfg.kappa_nu
    nu = cfg.nu
    dust_to_gas = cfg.dust_to_gas

    def _dust_temp(field, data):
        # Dust temperature is stored dimensionless; attach Kelvin units.
        return data[("PartType0", "Dust_Temperature")] * yt.YTQuantity(1.0, "K")

    def _dust_mass(field, data):
        return dust_to_gas * data[("PartType0", "mass")]

    def _dust_density(field, data):
        return dust_to_gas * data[("PartType0", "density")]

    def _planck_nu(field, data):
        # Planck function B_nu(T) in erg/s/cm^2/Hz/sr.
        T = data[("PartType0", "Dust_Temperature_K")]
        x = (H_PLANCK * nu) / (K_BOLTZ * T)               # dimensionless
        bnu = (2 * H_PLANCK * nu**3 / C_LIGHT**2) / (np.exp(x) - 1)
        return bnu / yt.YTQuantity(1.0, "sr")

    def _dust_emissivity_nu(field, data):
        # j_nu = kappa_nu * rho_dust * B_nu   ->   erg/s/cm^3/Hz/sr
        rho_d = data[("PartType0", "dust_density")]
        bnu = data[("PartType0", "b_nu")]
        return kappa_nu * rho_d * bnu

    def _number_density(field, data):
        # Gas number density assuming molecular gas (H2 + He), mu = 1.3.
        m_H = yt.YTQuantity(1.6736e-24, "g")
        mu = 1.3
        return data[("PartType0", "density")] / (mu * m_H)

    ds.add_field(("PartType0", "number_density"), function=_number_density,
                 units="cm**-3", sampling_type="particle", force_override=True)
    ds.add_field(("PartType0", "Dust_Temperature_K"), function=_dust_temp,
                 units="K", sampling_type="particle", force_override=True)
    ds.add_field(("PartType0", "dust_mass"), function=_dust_mass,
                 units="g", sampling_type="particle", force_override=True)
    ds.add_field(("PartType0", "dust_density"), function=_dust_density,
                 units="g/cm**3", sampling_type="particle", force_override=True)
    ds.add_field(("PartType0", "b_nu"), function=_planck_nu,
                 units="erg/s/cm**2/Hz/sr", sampling_type="local", force_override=True)
    ds.add_field(("PartType0", "dust_emissivity_nu"), function=_dust_emissivity_nu,
                 units="erg/s/cm**3/Hz/sr", sampling_type="local", force_override=True)


# ---------------------------------------------------------------------------
# Snapshot loading
# ---------------------------------------------------------------------------
def _load_snapshot(path: str):
    """Open a snapshot and return ``(ds, ad, box_size_pc, snapshot_name)``.

    The unit base is read from the snapshot header so that the loader works
    regardless of the code-unit conventions in a given run. ``UnitB`` is set
    to a fixed 1e4 because some snapshots do not store a magnetic-field unit.
    """
    snapshot_name = os.path.splitext(os.path.basename(path))[0]

    with h5py.File(path, "r") as f:
        header = f["Header"].attrs
        unit_base = {
            "UnitLength_in_cm": header["UnitLength_In_CGS"],
            "UnitMass_in_g": header["UnitMass_In_CGS"],
            "UnitVelocity_In_CGS": header["UnitVelocity_In_CGS"],
            "UnitB": 1e4,
        }
        # Box size converted from code units to parsecs.
        box_size_pc = header["BoxSize"] * unit_base["UnitLength_in_cm"] / PC_TO_CM

    ds = yt.load(path, unit_base=unit_base)
    ad = ds.all_data()
    return ds, ad, box_size_pc, snapshot_name


def _snapshot_id(snapshot_name: str) -> str:
    """Return the bare snapshot ID used as a key in the results JSON files.

    The sky-model filename stem is e.g. ``snapshot_170``; downstream
    (``analysis.py``) keys ``fitting_results.json`` by the trailing numeric ID
    ``170``. We key the luminosity file the same way so the two line up.
    """
    return snapshot_name.split("_")[-1]


def _total_luminosity_lsun(ad) -> float:
    """Total protostellar (sink) luminosity in solar luminosities.

    Sums ``StarLuminosity_Solar`` over the PartType5 sink particles. Returns
    0.0 if the snapshot has no sinks / no luminosity field (e.g. a snapshot
    taken before any protostar has formed).
    """
    field = ("PartType5", "StarLuminosity_Solar")
    if field not in ad.ds.field_list:
        return 0.0
    return float(np.sum(ad[field]).d)


# ---------------------------------------------------------------------------
# FITS writing
# ---------------------------------------------------------------------------
def _write_fits(image_jy: np.ndarray, pixel_ang_deg: float, pixel_arcsec: float,
                cfg: Config, out_path: str) -> None:
    """Write a flux-density map to a FITS file with a minimal WCS header.

    Parameters
    ----------
    image_jy:
        2-D array of flux density in Jy/pixel.
    pixel_ang_deg:
        Angular pixel size in degrees (used for CDELT1/CDELT2).
    pixel_arcsec:
        Angular pixel size in arcseconds (stored in the custom PARC keyword
        for convenience downstream).
    cfg:
        Active configuration (for the image resolution).
    out_path:
        Destination FITS path.
    """
    nx, ny = cfg.resolution
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = nx
    header["NAXIS2"] = ny
    header["CRPIX1"] = nx / 2 + 0.5
    header["CRPIX2"] = ny / 2 + 0.5
    header["CDELT1"] = pixel_ang_deg            # deg/pixel
    header["CDELT2"] = pixel_ang_deg            # deg/pixel
    header["CTYPE1"] = "LINEAR"
    header["CTYPE2"] = "LINEAR"
    header["BUNIT"] = "Jy/pixel"
    header["MAXSNU"] = float(np.max(image_jy))  # peak flux, handy for inbright
    header["PARC"] = pixel_arcsec               # pixel size in arcsec

    fits.PrimaryHDU(data=image_jy, header=header).writeto(out_path, overwrite=True)


# ---------------------------------------------------------------------------
# Core per-snapshot processing
# ---------------------------------------------------------------------------
def process_snapshot(path: str, cfg: Config, out_dir: str):
    """Generate all sky-model FITS files for a single snapshot.

    Returns
    -------
    snapshot_id : str
        Bare snapshot ID (e.g. ``"170"``), matching the keys used in the
        downstream results JSON.
    total_luminosity_lsun : float
        Summed protostellar luminosity in solar luminosities.
    written : list of str
        Paths of the FITS files written for this snapshot.
    """
    ds, ad, box_size_pc, snapshot_name = _load_snapshot(path)
    _register_dust_fields(ds, cfg)

    snapshot_id = _snapshot_id(snapshot_name)

    # Total protostellar luminosity (used downstream to set the dust
    # temperature in the dust-mass estimate).
    total_luminosity_lsun = _total_luminosity_lsun(ad)
    print(f"[{snapshot_name}] total protostellar luminosity "
          f"= {total_luminosity_lsun:.3f} Lsun")

    # Quick sanity-check: integrated disk dust mass above the density cut.
    msk = np.where(ad[("PartType0", "number_density")] > cfg.number_density_threshold)
    disk_dust_mass = ad[("PartType0", "dust_mass")][msk]
    disk_dust_mass_msun = float(np.sum(disk_dust_mass) / SOLAR_MASS_G)
    print(f"[{snapshot_name}] disk dust mass (n > {cfg.number_density_threshold:.0e}) "
          f"= {disk_dust_mass_msun:.3e} Msun")

    emissivity_field = ("PartType0", "dust_emissivity_nu")
    written: List[str] = []

    for axis in cfg.axes:
        # Project the dust emissivity along this axis (no weighting -> integral
        # along the line of sight).
        prj = yt.ProjectionPlot(ds, axis, emissivity_field, weight_field=None)
        prj.set_cmap(emissivity_field, "plasma")
        prj.set_log(emissivity_field, True)

        # Centre on the densest gas cell. The two in-plane coordinates depend
        # on which axis we are projecting along.
        _, cen = ds.find_max(("PartType0", "density"))
        if axis == "x":
            prj.set_center((cen[1], cen[2]))
        elif axis == "y":
            prj.set_center((cen[2], cen[0]))
        elif axis == "z":
            prj.set_center((cen[0], cen[1]))

        # Zoom: scale with box size for large boxes, fixed otherwise.
        zoom = cfg.zoom_base * box_size_pc if box_size_pc > 0.1 else cfg.zoom_base
        prj.zoom(zoom)

        # Fixed-resolution buffer -> numpy array of projected emissivity.
        frb = prj.frb
        prj_width_cm = prj.width[0].to("cm").d
        pixel_length_cm = prj_width_cm / cfg.resolution[0]
        pixel_area_cm2 = pixel_length_cm**2

        projected_emissivity = frb[emissivity_field].d  # erg/s/cm^2/Hz/sr * cm

        # Convert emissivity -> flux density (Jy/pixel) for each region by
        # dividing by distance^2 and multiplying by the pixel solid-angle area.
        for region, dist_pc in cfg.region_distances_pc.items():
            dist_cm = dist_pc * PC_TO_CM

            s_nu_cgs = projected_emissivity * (pixel_area_cm2 / dist_cm**2)  # erg/s/cm^2/Hz
            s_nu_jy = s_nu_cgs * JY_PER_CGS_FLUX                             # Jy/pixel

            # Angular pixel size (small-angle approximation).
            pixel_ang_rad = pixel_length_cm / dist_cm
            pixel_ang_deg = np.degrees(pixel_ang_rad)
            pixel_arcsec = pixel_ang_deg * 3600.0

            out_name = (f"{snapshot_name}_{region}_flux_map_"
                        f"{cfg.telescope}_axis_{axis}.fits")
            out_path = os.path.join(out_dir, out_name)
            _write_fits(s_nu_jy, pixel_ang_deg, pixel_arcsec, cfg, out_path)
            written.append(out_path)
            print(f"  saved {out_name} | pixel size = {pixel_arcsec:.4f} arcsec")

    return snapshot_id, total_luminosity_lsun, written


# ---------------------------------------------------------------------------
# Input gathering and CLI
# ---------------------------------------------------------------------------
def _gather_inputs(args) -> List[str]:
    """Resolve the list of snapshot paths from the CLI arguments."""
    if args.snapshots:
        return list(args.snapshots)
    pattern = os.path.join(args.input_dir, args.glob)
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No snapshots matched {pattern!r}")
    return paths


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate synthetic sky-model FITS images from simulation snapshots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("-i", "--input-dir",
                     help="Directory containing snapshot .hdf5 files.")
    src.add_argument("--snapshots", nargs="+",
                     help="Explicit list of snapshot files to process.")
    p.add_argument("--glob", default="*.hdf5",
                   help="Glob pattern used with --input-dir to find snapshots.")
    p.add_argument("-o", "--output-dir", default="skymodels",
                   help="Directory to write sky-model FITS files into.")
    p.add_argument("--luminosity-file", default="luminosities.json",
                   help="JSON file to write per-snapshot total luminosities "
                        "into (read by analysis.py to set the dust temperature).")
    p.add_argument("--telescope", default="ALMA", choices=sorted(TELESCOPE_PRESETS),
                   help="Instrument preset (sets dust opacity and frequency).")
    p.add_argument("--resolution", type=int, default=800,
                   help="Square projection resolution in pixels.")
    return p


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    cfg = Config(
        telescope=args.telescope,
        resolution=(args.resolution, args.resolution),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    paths = _gather_inputs(args)
    print(f"Processing {len(paths)} snapshot(s) with telescope={cfg.telescope}, "
          f"resolution={cfg.resolution}")

    total = 0
    luminosities = {}
    for path in paths:
        snapshot_id, lum_lsun, written = process_snapshot(path, cfg, args.output_dir)
        luminosities[snapshot_id] = lum_lsun
        total += len(written)

    # Write the per-snapshot luminosities for analysis.py to read in.
    # Keys are bare snapshot IDs (e.g. "170") to match fitting_results.json.
    with open(args.luminosity_file, "w") as f:
        json.dump(luminosities, f, indent=2)

    print(f"\nDone. Wrote {total} FITS file(s) to {args.output_dir!r}.")
    print(f"Wrote luminosities for {len(luminosities)} snapshot(s) "
          f"to {args.luminosity_file!r}.")


if __name__ == "__main__":
    main()
