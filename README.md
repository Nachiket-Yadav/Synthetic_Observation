# Synthetic ALMA/VLA Disk Observation Pipeline

A pipeline for turning hydrodynamic simulation snapshots of protostellar disks
into **synthetic interferometer observations**, fitting those observations, and
recovering disk dust masses — so that "observed" disk properties can be compared
directly against the known simulation ground truth.

## Example output

Three-panel QA figures produced by `plot_three_panel.py` (Stage 4): input
skymodel, CASA synthetic observation, and imfit residual, for two Orion
snapshots at very different disk sizes/inclinations.

**Snapshot 170 — Orion — axis y**
![Snapshot 170, Orion, axis y](figures/snap170_Orion_axisy_three_panel.png)

**Snapshot 386 — Orion — axis y**
![Snapshot 386, Orion, axis y](figures/snap386_Orion_axisy_three_panel.png)

---

The workflow has four stages:

```
 snapshots (.hdf5)
        │   skymodel_generation.py
        ▼
 sky models (.fits) + luminosities.json   ── visualization.ipynb (optional checks)
        │   casa_simulation.py   (runs inside CASA)
        ▼
 observed images (.fits)
        │   analysis.py          (runs inside CASA, reads luminosities.json)
        ▼
 fitting_results.json   (fitted sizes/fluxes + predicted dust masses)
        │   plot_three_panel.py
        ▼
 figures/   (skymodel | observation | residual QA figures, per snapshot/region/axis)
```

---

## Repository contents

| File | What it does | Where it runs |
|------|--------------|---------------|
| `skymodel_generation.py` | Reads simulation snapshots, builds dust-emissivity projections, and writes sky-model FITS images. | Plain Python (needs `yt`) |
| `visualization.ipynb` | Interactive sanity checks: inspect a snapshot, view emissivity/flux maps, compare a sky model against a CASA image. **Not required to run the pipeline.** | Jupyter |
| `casa_simulation.py` | Runs `simobserve` → `tclean` → `impbcor` → `exportfits` to produce synthetic observed images. | Inside CASA |
| `analysis.py` | Fits each observed image with `imfit` and converts the fitted flux into a dust mass. Combines the old `imfit_new.py` and `dust_prediction_code.py` into one pass over a folder. | Inside CASA |
| `plot_three_panel.py` | Renders zoomed skymodel / observation / residual QA figures for each fitted disk in `fitting_results.json`. | Plain Python |
| `plot_three_panel.ipynb` | Interactive version of the same three-panel plot, plus scratch analysis cells. **Not required to run the pipeline.** | Jupyter |

---

## Stage 1 — Generate sky models

```bash
python skymodel_generation.py --input-dir /path/to/snapshots --output-dir skymodels
```

For each snapshot this writes one FITS file per (region, projection axis), e.g.
`snapshot_170_Orion_flux_map_ALMA_axis_x.fits`. The output filename pattern is
relied on by the later stages, so don't rename the files.

It also writes `luminosities.json`, a small file mapping each snapshot ID to its
total protostellar luminosity (summed `StarLuminosity_Solar` over the sink
particles). Stage 3 reads this to set the dust temperature used in the dust-mass
estimate. Change its location with `--luminosity-file`.

Common options:

```bash
# Process a whole directory
python skymodel_generation.py -i ./data -o skymodels

# Process specific snapshots instead of a whole directory
python skymodel_generation.py --snapshots snap_170.hdf5 snap_171.hdf5 -o skymodels

# only snapshots 170-179
python skymodel_generation.py -i ./data --glob "snapshot_17*.hdf5" -o skymodels

# Use the VLA preset (different opacity/frequency)
python skymodel_generation.py -i ./data -o ./skymodels --telescope VLA
```

The physics and geometry (dust opacity `kappa_nu`, frequency `nu`,
dust-to-gas ratio, projection resolution, region distances) live in the
`Config` dataclass and `TELESCOPE_PRESETS` dictionary at the top of the script.
Run `python skymodel_generation.py --help` for all options.

### Optional: sanity-check a snapshot first

Open `visualization.ipynb`, set `SNAPSHOT_PATH`, and run the cells to look at
the disk mass, the dust-emissivity projection, and the per-region flux maps
before committing to a batch run.

---

## Stage 2 — Synthetic observation (CASA)

Run on a folder of sky models:

```bash
casa --nogui --nologger -c casa_simulation.py \
    --skymodel-dir skymodels --out-dir pbcor_imgs
```

or, from inside an interactive CASA session:

```python
import sys
sys.argv = ['casa_simulation.py', '--skymodel-dir', 'skymodels',
            '--out-dir', 'pbcor_imgs']
execfile('casa_simulation.py')
main()
```

This produces a primary-beam-corrected image
`ALMA_snapshot_<id>_axis_<axis>_<Region>_sim_observed_pbcor.fits` for each sky
model in `pbcor_imgs/`. Pass `--skip-existing` to resume an interrupted run.

The observing setup (pointing, frequency, integration time, array
configuration, noise model, imaging parameters) is collected in the
`OBS_SETTINGS` dictionary near the top of the script.

---

## Stage 3 — Fit & predict dust masses (CASA)

```bash
casa --nogui --nologger -c analysis.py \
    --image-dir pbcor_imgs --results fitting_results.json \
    --luminosity-file luminosities.json
```
or, from inside an interactive CASA session:

```python
import sys
sys.argv = ['analysis.py', '--image-dir', 'pbcor_imgs',
            '--results', 'fitting_results.json',
            '--luminosity-file', 'luminosities.json']
execfile('analysis.py')
main()  
```

For each observed image this:

1. fits a 2-D Gaussian with `imfit` (size, flux, inclination, position angle,
   fit-quality metrics);
2. exports the residual and model images to `residual_imgs/` and `model_imgs/`;
3. converts the fitted flux into a dust mass assuming optically thin,
   isothermal dust.

Results are written to `fitting_results.json` with the structure:

```
{ snapshot: { region: { axis: { Rmaj, flux, inc, radius_AU_Tobin,
                                snr, dust_mass_Msun, ... } } } }
```

**Dust temperature.** The dust mass depends on an assumed dust temperature.
Stage 3 reads `luminosities.json` (written by stage 1) and scales the
temperature as `T = 43 K · L^0.25` per snapshot. If the luminosity file is
missing, or a snapshot's luminosity is zero (no protostar has formed yet), it
falls back to `--dust-temp` (default 43 K). 

---

## Stage 4 — Three-panel QA figures

```bash
python plot_three_panel.py \
    --results fitting_results.json --skymodel-dir skymodels \
    --pbcor-dir pbcor_imgs --residual-dir residual_imgs --out-dir figures
```

For every snapshot/region/axis present in `fitting_results.json` this writes
a `snap<snapshot>_<Region>_axis<axis>_three_panel.png` to `--out-dir`, each
showing the skymodel, the CASA observation, and the imfit residual side by
side with the fitted Gaussian ellipse overlaid and the fit statistics
annotated. Restrict the run with `--snapshots`, `--fields`, and `--axes`,
e.g. `--snapshots 170 386 --fields Orion --axes y` to reproduce the figures
shown at the top of this README.

The core `plot_three_panel(pbcor_fpath, fit, skymodel_dir, residual_dir, ...)`
function takes a `savefig` argument — an explicit output path that overrides
the `out_dir`/auto-generated naming — so it can also be called directly
(e.g. from `plot_three_panel.ipynb` or another script) to save a single
figure wherever you like.

---

## Output files

| File / folder | Produced by | Contents |
|---------------|-------------|----------|
| `skymodels/` | Stage 1 | Sky-model FITS images |
| `luminosities.json` | Stage 1 | Per-snapshot total protostellar luminosity (Lsun) |
| `pbcor_imgs/` | Stage 2 | Synthetic pb-corrected observations |
| `residual_imgs/`, `model_imgs/` | Stage 3 | imfit residual & model images |
| `fitting_results.json` | Stage 3 | Fitted properties + predicted dust masses |
| `figures/` | Stage 4 | Three-panel skymodel / observation / residual QA PNGs |

---

## Notes 

- Snapshot **unit conventions are read from each file's header**, so the sky-model
  stage adapts to different runs without code changes.
- The four stages communicate purely through **files and filename conventions**.
  If you change a filename pattern in one stage, update the parsing in the next.
- The scripts check-point as they go (writing JSON / FITS after each item), so a
  long run can be interrupted and resumed.

