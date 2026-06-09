# 🌑 Rover-Accessible Science Priority Mapping — Lunar South Pole

> **Research Internship @ NIT Trichy** | Aligned with ISRO & NASA VIPER Mission Planning  
> *A 13-step geospatial ML pipeline to identify which scientifically valuable lunar targets a rover can safely and realistically reach.*

---

## 🚀 Project Overview

The lunar south pole is one of the most scientifically compelling — and operationally challenging — destinations for rover missions. This pipeline answers a precise question:

**Which scientifically interesting targets near the lunar south pole can a rover safely and realistically reach?**

Using 6 real sensor datasets from NASA and ISRO/Chandrayaan-1, the pipeline segments 26,237 terrain-bounded regions, scores each for science value and rover accessibility, and produces a ranked, Pareto-optimal target list validated against real traversability and shadow/thermal constraints.

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Terrain regions segmented | 26,237 |
| QUALIFIED safe candidates | 2,605 (9.9%) |
| UNCERTAIN (saved, not scored) | 1,847 |
| REMOVED (unsafe/insufficient data) | 21,785 |
| Feasible rover targets | ~180–220 |
| Sensor datasets used | 6 |
| Mission strategies benchmarked | 3 |
| Sensitivity tests conducted | 5 |
| Pipeline steps | 13 |

---

## 🛰️ Datasets

| Dataset | Instrument | Resolution | Role |
|---|---|---|---|
| `lola_downsampled_25m.tif` | NASA LOLA | 25 m/px | Slope, roughness, segmentation |
| `lroc_5m_float32_clean.tif` | NASA LROC NAC | 5 m/px | Terrain context |
| `lroc_shadow_mask_final.tif` | NASA LROC | 5 m/px | Shadow/illumination risk |
| `minirf_s1_pju_mosaic_circle.tif` | Chandrayaan-1 Mini-RF | 59.225 m | SAR science indicator |
| `dfsar_cpr_mosaic_circle.tif` | LRO DFSAR | 25 m/px | Circular polarisation ratio |
| `m3_mosaic_group2_july13_14_allbands_280m.tif` | Chandrayaan-1 M3 | 280 m, 85 bands | Spectral science context |
| `gravil_anom_l900_native_10km_circular.tif` | GRAIL | 10 km | Regional gravity context |
| `gravil_boug_l900_native_10km_circular.tif` | GRAIL | 10 km | Regional gravity context |

All datasets share **Polar Stereographic Moon** projection (lat_origin=−90°, meridian=0°, spheroid=1,737,400 m).

---

## 🔬 Pipeline Architecture

```
INPUT: 6 Sensor Rasters (LOLA, LROC, Mini-RF, DFSAR, M3, GRAIL)
        │
        ▼
STEP 1  File description table + alignment verification
        │
        ▼
STEP 2  Marker-Controlled Watershed Segmentation (scikit-image)
        SEED_WINDOW = 375m | SEED_MAX_SLOPE = 20° | ROUGHNESS_KERN = 175m
        → 26,237 terrain-bounded candidate regions
        │
        ▼
STEP 3  9-Criterion Qualification Filter
        QUALIFIED (2,605) | UNCERTAIN (1,847) | REMOVED (21,785)
        ── UNCERTAIN saved separately, never scored downstream ──
        │
        ▼
STEP 4  M3 Spectral Band-Depth Indices (QUALIFIED only)
        BD1000 | BD2000 | OH_SPECTRAL_INDICATOR | ICE_SPECTRAL_INDICATOR
        │
        ▼
STEP 5  Shadow Risk Map
STEP 6  Radar Feature Extraction (DFSAR CPR + Mini-RF log(1+x))
STEP 7  Science Context Summary
        │
        ▼
STEP 8  Science Score
        ICE×0.35 + OH×0.25 + BD1000×0.20 + BD2000×0.20  (normalised 0→1)
        │
        ▼
STEP 9  Rover Accessibility Constraints
        slope_mean ≤ 15° | shadow_fraction ≤ 0.50
        roughness_mean ≤ 50m | dist_from_centre ≤ 40km
        │
        ▼
STEP 10 Feasible Target Regions (~180–220 validated candidates)
STEP 11 Ranked Target List (Top 20 by Science Score)
        │
        ▼
STEP 12 Pareto Front Selection
        Science Score vs Slope — 3 strategies benchmarked
        (science-only | safety-only | weighted hybrid)
        │
        ▼
STEP 13 Sensitivity Analysis (5 tests)
        Slope ±3° | Shadow ±0.20 | Weight perturbations | Sensor removal

OUTPUT: Ranked, terrain-validated, Pareto-optimal science target list
```

---

## 🧪 Science Scoring

M3 spectral band-depth indices are computed from Chandrayaan-1 M3 L2 thermally-corrected reflectance using published band pairs:

| Index | Band Pair | Reference |
|---|---|---|
| **BD1000** | B9 (750nm) ref, B22 (1010nm) abs | Lucey et al. 2000, JGR |
| **BD2000** | B56 (1818nm) ref, B61 (2018nm) abs | Mustard et al. 2011, JGR |
| **OH_SPECTRAL_INDICATOR** | B74 (2537nm) ref, B81 (2817nm) abs | Pieters et al. 2009, Science |
| **ICE_SPECTRAL_INDICATOR** | B84 (2936nm) ref, B85 (2976nm) abs | Li & Milliken 2017, Sci. Advances |

> ⚠️ **Limitation:** These are spectral context indicators only. They identify regions where absorption features occur at wavelengths associated with OH and H₂O ice in published literature. They do **not** constitute verified ice or water detection. Quantitative volatile abundance requires additional photometric correction and validation (Clark et al. 2011, JGR Planets).

---

## 🤖 Rover Accessibility (VIPER-class)

| Constraint | Threshold | Physical Basis |
|---|---|---|
| `slope_mean_deg` | ≤ 15° | JPL rocker-bogie wheel-slip limit on lunar regolith (Kobayashi et al. 2010) |
| `shadow_fraction` | ≤ 0.50 | Minimum illumination for sustained solar rover power |
| `roughness_mean_m` | ≤ 50m | ~16° average undulation at 175m kernel scale |
| `dist_from_centre_km` | ≤ 40km | Exceeds VIPER planned traverse (~26km); geometric proxy |

---

## 📐 Qualification Criteria (Step 3)

Nine hard checks applied to all 26,237 regions:

```
1. area_km2             ≥ 0.0784 km²    (minimum 280m-scale region)
2. slope_mean_deg       ≤ 20°           (HARD REMOVE — crater wall)
3. slope_max_deg        ≤ 30°           (HARD REMOVE — cliff pixel present)
4. lola_valid_pct       ≥ 0.50          (data sufficiency)
5. slope_std_deg        ≤ 5°            → UNCERTAIN flag if exceeded
6. roughness_std_m      ≤ 10m           → UNCERTAIN flag if exceeded
7. shadow_std           ≤ 0.35          → UNCERTAIN flag if exceeded
8. fill_ratio           ≥ 0.25          (ROI edge partial coverage)
9. bbox_aspect          ≤ 10            (sliver shape rejection)
```

---

## 🛠️ Tech Stack

```
Python 3.x        NumPy · Pandas · Matplotlib
Rasterio          Geospatial raster I/O and CRS handling
GDAL              Raster projection and transform arithmetic
scikit-image      Marker-Controlled Watershed segmentation
SciPy             Generic filter for roughness computation
Jupyter Notebook  Full pipeline execution environment
```

---

## 📁 Repository Structure

```
📦 lunar-south-pole-pipeline
 ┣ 📜 LUNAR_PIPELINE_FINAL_v3.md     ← Full pipeline (Steps 1–13)
 ┣ 📁 raster_layers/                 ← Input rasters (not tracked — see Datasets)
 ┣ 📁 outputs/
 ┃ ┣ candidate_regions.csv
 ┃ ┣ accepted_candidate_regions.csv
 ┃ ┣ uncertain_candidate_regions.csv
 ┃ ┣ science_score.csv
 ┃ ┣ feasible_target_regions.csv
 ┃ ┣ pareto_target_comparison.csv
 ┃ ┣ constraint_sensitivity_results.csv
 ┃ ┣ science_priority_among_feasible_targets.csv
 ┃ ┗ *.png                           ← All visualisation outputs
 ┗ 📜 README.md
```

---

## 📤 Outputs

| File | Description |
|---|---|
| `candidate_regions.csv` | All 26,237 segmented regions with full feature set |
| `accepted_candidate_regions.csv` | 2,605 QUALIFIED regions only |
| `uncertain_candidate_regions.csv` | 1,847 UNCERTAIN regions (not scored) |
| `science_score.csv` | Weighted science score per QUALIFIED region |
| `feasible_target_regions.csv` | ~180–220 rover-accessible candidates |
| `pareto_target_comparison.csv` | Pareto-optimal targets + 3-strategy comparison |
| `constraint_sensitivity_results.csv` | 5-test sensitivity results with top-10 overlap |
| `science_priority_among_feasible_targets.csv` | Final ranked target list |
| `ranked_targets_map.png` | Top 20 targets annotated on science score map |
| `target_strategy_comparison_plot.png` | Science-only vs safety-only vs Pareto comparison |
| `sensitivity_summary.png` | Feasible count and ranking stability plots |

---

## 📚 Key References

- Meyer, F. (1992). Topographic distance and watershed lines. *Signal Processing*, 38(1).
- Drăguţ, L. & Blaschke, T. (2006). Automated classification of landform elements. *Geomorphology*, 81(3–4).
- Lucey, P.G. et al. (2000). Lunar iron and titanium abundance algorithms. *JGR Planets*.
- Mustard, J.F. et al. (2011). Compositional diversity and geological insights. *JGR Planets*.
- Pieters, C.M. et al. (2009). Character and spatial distribution of OH/H₂O on the surface of the Moon. *Science*, 326(5952).
- Li, S. & Milliken, R.E. (2017). Water on the surface of the Moon. *Science Advances*, 3(9).
- Clark, R.N. et al. (2011). Detection and mapping of hydroxyls and water on the Moon. *JGR Planets*.
- Kobayashi, T. et al. (2010). Measurement of lunar surface slope. *Journal of Terramechanics*, 47(5–6).
- Ulaby, F.T. et al. (1982). *Microwave Remote Sensing: Active and Passive*. Vol. II.

---

## 🏛️ Affiliation

**Research Internship** — National Institute of Technology Tiruchirappalli (NIT Trichy)  
Aligned with ISRO Chandrayaan mission planning and NASA VIPER rover traversability criteria.

> 📄 *Research paper in preparation: Multi-criteria Rover Target Ranking for Lunar South Pole Exploration (2026)*

---

## 👤 Author

**Vignesh B S**  
B.E. Computer Science Engineering, Sri Krishna College of Technology, Coimbatore  
[GitHub](https://github.com/itsvickyhere) · [LinkedIn](https://www.linkedin.com/in/vignesh-b-s-50981b32b/)
