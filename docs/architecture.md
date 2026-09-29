# System Architecture and Component Specifications

The **SIH26071 Early Warning and Inundation Prediction System (REWS)** is structured as a modular, physics-informed hydrometeorological pipeline designed for operational forecasting across lead times from minutes to 72 hours.

---

## 1. High-Level System Architecture

```mermaid
graph TD
    subgraph Ingestion ["1. Multi-Sensor Ingestion Layer"]
        A1["NASA GPM IMERG (V07B)"]
        A2["NOAA GFS v16 (AWS S3)"]
        A3["ISRO MOSDAC INSAT-3D/3DR"]
        A4["IMD Radar (Chennai DWR)"]
        A5["Copernicus DEM GLO-30 & OSM"]
    end

    subgraph Preprocessing ["2. Preprocessing & Harmonization"]
        B1["Conservative Area-Weighted Regridding (0.1 deg)"]
        B2["Meteorological QC & Hot-Pixel Filtering"]
        B3["Feature Engineering (Lags, Rolling Acc, IMD Day Progress)"]
        B4[("Unified Zarr Datacube")]
    end

    subgraph Modeling ["3. Multi-Horizon Forecasting Engine"]
        C1["Stage 3: Nowcasting (0 to 3 h)<br><b>PySteps (S-PROG & STEPS)</b>"]
        C2["Stage 4: NWP Post-Processing (3 to 72 h)<br><b>EQM + LightGBM Quantile Regression</b>"]
        C3["Stage 5: Blended Fusion<br><b>Smooth Sigmoidal Transition (1 to 3 h)</b>"]
    end

    subgraph Hydro ["4. Hydrologic & Inundation Layer"]
        D1["USDA NRCS / SCS Runoff Model"]
        D2["2D Local-Inertial Shallow Water Solver"]
        D3["Physics-Emulated U-Net Flood Surrogate"]
    end

    subgraph Downstream ["5. Alerts & Impact Verification"]
        E1["Ward-Level Critical Infrastructure Exposure"]
        E2["CAP 1.2 XML / JSON Alerts (IMD/NDMA Aligned)"]
        E3["Dual-Layer Ground Truth Verification"]
    end

    Ingestion --> Preprocessing
    B4 --> C1
    B4 --> C2
    C1 --> C3
    C2 --> C3
    C3 --> Hydro
    Hydro --> Downstream
```

---

## 2. Stage 3 Nowcaster Scoping & Scientific Rationale

### Approved Design Decision
Stage 3 is scoped exclusively to **PySteps** (deterministic S-PROG optical flow extrapolation and probabilistic STEPS / LINDA ensemble perturbation).

### Deep Learning Nowcasting (ConvLSTM / U-Net) Documented as Future Work
Deep spatio-temporal neural networks (such as ConvLSTM, PredRNN, or U-Net nowcasters) were evaluated and explicitly deferred as future work for the following technical reasons:
1. **Time Scope and Storm Sample Regimes**: Training high-capacity recurrent spatio-temporal architectures without severe spatial blurring or mode collapse requires multi-year continuous sequences across hundreds of independent convective events, requiring training and tuning cycles outside the timeline of this hackathon submission.
2. **Effective Spatial Resolution vs. Convective Cell Scales**: At the 0.1° (~10 km) grid resolution of satellite precipitation products (IMERG), fine convective storm initiation and cellular decay cannot be resolved by neural recurrence; semi-Lagrangian optical flow advection with spectral autoregressive cascades (PySteps S-PROG / STEPS) represents the established operational meteorological benchmark for this scale.
3. **Marginal Value and Operational Robustness**: PySteps requires zero offline parameter fitting, runs inference in <2 seconds on standard CPU infrastructure, avoids the unphysical blur of $L_1/L_2$ regression losses, and generates calibrated probabilistic ensembles directly from recent radar/satellite motion fields.
