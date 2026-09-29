# Data Sources, Ingestion Inventory, and Sensor Footprint Specifications

This document records the provenance, empirical availability, access protocols, and sensor resolutions for all datasets used in the **SIH26071 Heavy Rainfall Early Warning and Inundation Prediction System**.

---

## 1. Empirical Probe Results for Cyclone Michaung (2023-12-03)

The empirical probe script (`scripts/probe_sources.py`) tested live availability for the Cyclone Michaung focus date (`2023-12-03 00:00 UTC`). The real probe results are recorded in [`outputs/provenance/source_probe.json`](file:///d:/rainfall-ews/sih26071-rainfall-ews/outputs/provenance/source_probe.json):

| Dataset | Provider / Source | Live Probe Status | Measured File Size / Latency | Key Findings & Operational Policy |
| :--- | :--- | :--- | :--- | :--- |
| **NOAA GFS (v16)** | AWS Open Data (`herbie`) | **Available (HTTP 200)** | Index: 40.5 KB (7.2s query) | Fully accessible on AWS S3 back to **2021-03-22** (earliest GFS v16 cycle). Byte-range subsetting used to avoid full-file downloads. |
| **ECMWF Open Data** | ECMWF Open Data API (`ifs 0p25`) | **Unavailable (HTTP 404)** | 0 bytes (8.1s) | **Confirmed empirically**: The rolling open-data bucket only maintains the current rolling operational window (~4-5 days). Historical Dec 2023 is not retained on the open bucket. Fallback: NOAA GFS is the primary historical NWP; manual GRIB folder supported at `data/raw/nwp/ecmwf/`. |
| **Copernicus DEM GLO-30** | AWS Open Data S3 | **Available (HTTP 200)** | 11.79 MB (1.83s) | Direct COG download for tile `Copernicus_DSM_COG_10_N13_00_E080_00_DEM.tif` covering Chennai ($13^\circ\text{N}, 80^\circ\text{E}$). |
| **ESA WorldCover 10m** | AWS Open Data S3 | **Available (HTTP 200)** | 113.55 MB (1.16s) | Direct GeoTIFF download for tile `ESA_WorldCover_10m_2021_v200_N12E078_Map.tif`. |
| **ISRO MOSDAC INSAT-3D/3DR** | ISRO MOSDAC Portal | **Portal Online (HTTP 200)** | N/A | Portal is accessible, but interactive downloads require session captcha. Pipeline uses official download routes or parses `.h5` files placed in `data/raw/insat/`. |
| **Sentinel-1 SAR** | ASF DAAC Search API | **Queried (2 scenes found)** | Search completed in 2.1s | 2 GRD scenes found over Chennai for the window: both acquired on **2023-11-30** (onset baseline). Sentinel-1 repeat cycle is 12 days, so no overpass occurred during the peak flood (Dec 4-5). Documented published flood extent maps (Copernicus EMS EMSR706, ISRO Bhuvan, UNOSAT) used for peak flood extent. |
| **GPM IMERG (V07B)** | NASA GES DISC | **Requires Earthdata Auth** | ~20 MB per granule | Requires active `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` in `.env` for HTTPS redirection through URS. |
| **ERA5 Reanalysis** | ECMWF CDS API | **Available (1940-present)** | ~20 MB per day | Requires `CDSAPI_KEY` in `.env`. Historical coverage verified for Dec 2023. |

---

## 2. Spatial Resolution and Sensor Footprint Honesty

- **Pilot Rainfall Grid**: $0.1^\circ$ ($\approx 11\text{ km}$ at Chennai latitude).
  - This matches the native physical resolution of NASA GPM IMERG.
  - **No Sub-Grid False Skill**: Resampling $0.1^\circ$ IMERG or $0.25^\circ$ GFS onto finer grids (e.g. $0.02^\circ$) is an interpolation/upsampling step. It does not introduce true convective microscale information. Any high-resolution rainfall product generated is strictly labelled **"downscaled"**, and verified skill numbers are reported at the native sensor footprint.
- **Inundation Grid**: $30\text{ m}$ (derived from Copernicus GLO-30 in `EPSG:32644` UTM Zone 44N).
  - A coarser $90\text{ m}$ multi-resolution hydro grid is used for regional watershed inflow routing, with $30\text{ m}$ reserved for urban floodplains.

---

## 3. Observation Truth Conventions (Satellite vs. Gauge)

To ensure scientific honesty and transparency, all verification scores must declare their reference truth:

1. **Satellite-Referenced Truth (`imerg_final`)**:
   - GPM IMERG Final Run V07B (gauge-calibrated multi-satellite product with ~3.5 month latency).
   - Provides spatially continuous 2D rainfall fields across both land and the Bay of Bengal ocean fetch.
   - Evaluated using Fractions Skill Score (FSS @ 10, 20, 40, 80 km), Critical Success Index (CSI), and Brier Skill Score.
2. **Gauge-Referenced Truth (`imd_aws_arg`, `imd_gridded_daily`)**:
   - In-situ point gauge observations from IMD Automated Weather Stations (AWS) and Automated Rain Gauges (ARG) in the Chennai Metropolitan Area (e.g., Meenambakkam, Nungambakkam, Chembarambakkam).
   - IMD $0.25^\circ$ daily gridded rainfall dataset ($08:30$ to $08:30$ IST).
   - Point-to-pixel comparisons use nearest-neighbor or bilinear interpolation with explicit notation of representativeness error.

---

## 4. Soil and Runoff Modeling Specifications

For hydrodynamic runoff generation using the USDA NRCS / SCS Runoff Curve Number method:
- **Soil Hydrologic Dataset**:
  - **Primary Source**: ISRIC SoilGrids 250m v2.0 (sand, silt, clay fractions and depth to bedrock).
  - **Hydrologic Soil Group (HSG)**: Mapped to USDA HSG classes (A, B, C, D) using standard USDA texture triangle boundaries:
    - Group A: Sand, loamy sand, sandy loam (high infiltration $>0.30\text{ in/hr}$)
    - Group B: Silt loam, loam (moderate infiltration $0.15 - 0.30\text{ in/hr}$)
    - Group C: Sandy clay loam (slow infiltration $0.05 - 0.15\text{ in/hr}$)
    - Group D: Clay, silty clay, sandy clay, clay loam (very slow infiltration $<0.05\text{ in/hr}$)
  - **Secondary / Fallback Source**: FAO Harmonized World Soil Database (HWSD v2.0).
- **Curve Number (CN) Lookup Table**:
  - Derived from **USDA NRCS National Engineering Handbook, Part 630 (Hydrology), Chapter 9** and **TR-55 (Urban Hydrology for Small Watersheds)**.
  - Cross-references ESA WorldCover 10m classes (Built-up, Tree cover, Grassland, Cropland, Bare/sparse, Water bodies) against Hydrologic Soil Groups A, B, C, D under Antecedent Moisture Condition II (AMC-II), with standard adjustments for AMC-I (dry) and AMC-III (wet).

---

## 5. Digital Elevation Models & Surface Corrections

- **Copernicus DEM GLO-30**:
  - Open Copernicus license, 30 m spatial resolution.
  - Note: GLO-30 is a Digital Surface Model (DSM), which includes tree canopies and building rooftops.
- **FABDEM (Forest And Buildings removed DEM)**:
  - Global 30 m bare-earth DTM generated by removing building and tree height biases from Copernicus DEM.
  - **License Warning**: FABDEM is distributed under **CC-BY-NC-SA 4.0** (Non-Commercial). It is evaluated for research comparison; the open-source pipeline default remains Copernicus GLO-30 with OSM waterway channel burning.
- **Hydro-Enforcement**:
  - OpenStreetMap river and canal vectors (Adyar, Cooum, Buckingham Canal, Kosasthalaiyar) are burned into the DEM by depressing channel elevations by $1.5\text{–}2.5\text{ m}$ to prevent artificial damming at road bridges.

---

## 6. Sentinel-1 SAR & Published Inundation Datasets

- **Acquisitions Identified over Chennai**:
  1. `S1A_IW_GRDH_1SDV_20231130T003226_20231130T003251_051439_06353D_9270` (2023-11-30 00:32 UTC)
  2. `S1A_IW_GRDH_1SDV_20231130T003201_20231130T003226_051439_06353D_343D` (2023-11-30 00:32 UTC)
- **Honest Observational Limitation**:
  - The Sentinel-1A repeat orbit cycle over Chennai is 12 days. The direct pass on 2023-11-30 captured the pre-storm baseline. Peak flooding occurred on Dec 4–5, during which no Sentinel-1 overpass occurred over Chennai.
- **Published Flood Extent Ground Truth**:
  - **Copernicus Emergency Management Service (EMS)**: Rapid Mapping Activation `EMSR706` (Cyclone Michaung in Tamil Nadu).
  - **ISRO Bhuvan Disaster Management Support**: Flood Inundation Map of Chennai Dec 2023 (derived from RISAT-1A / EOS-04 SAR).
  - **UNOSAT Rapid Inundation Analysis**: Chennai Metropolitan Area Dec 2023.
