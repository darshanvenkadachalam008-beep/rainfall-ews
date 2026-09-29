# Evaluation Methodology, Data Splits, and Verification Standards

This document specifies the evaluation protocol, model benchmark baselines, split boundaries, and honesty criteria for the **SIH26071 Early Warning and Inundation Prediction System**.

---

## 1. Multi-Year Corpus and Event-Based Splitting

An operational early-warning model cannot be trained or evaluated on a single disaster event without catastrophic overfitting and inflated skill claims. We employ an **event-based leave-one-event-out splitting strategy**:

### Split Partitions

| Split | Description | Temporal Coverage | Role |
| :--- | :--- | :--- | :--- |
| **Train Split** | Historical Northeast Monsoon (NEM) seasons where GFS v16 AWS archive is available. | **2021-10-01 to 2021-12-31** (excluding Nov event)<br>**2022-10-01 to 2022-12-31** | Model weight optimization and feature learning. |
| **Validation Split** | Pre-cyclone 2023 season. | **2023-10-01 to 2023-11-20 UTC** | Hyperparameter tuning, early stopping, probability calibration curve fitting (Platt scaling & isotonic regression). |
| **Held-out Test Event 1** | **Cyclone Michaung** (Primary Hindcast). | **2023-11-30 to 2023-12-08 UTC** (Focus: Dec 03–06) | Primary disaster demonstration; zero training exposure. |
| **Held-out Test Event 2** | **Chennai Severe Inundation Event** (Secondary Hindcast). | **2021-11-05 to 2021-11-13 UTC** | Multi-event generalization verification. |

### Purge Gap Enforcement (Amendment D)
To eliminate temporal autocorrelation leakage between training, validation, and testing:
- A strict **temporal purge gap of $\ge 75\text{ hours}$** ($72\text{ h}$ NWP forecast lead time $+ 3\text{ h}$ nowcast input sequence) is enforced before and after every evaluation boundary.
- Any timestamp falling inside the purge window is tagged as `"purge_gap"` and excluded from both training and metric computation.
- Programmatic verification (`SplitDefinition.verify_no_leakage()`) guarantees $0\%$ sample overlap and enforces the separation buffer.

---

## 2. Mandatory Baselines and Skill Metrics

Every ML model must be compared against established physical and statistical baselines on the **exact same held-out test events**:

1. **Nowcasting (0 to 3 h)**:
   - *Operational Model & Physical Baseline*: **PySteps** optical-flow deterministic extrapolation (S-PROG) and probabilistic ensemble (STEPS / LINDA).
   - *Deep Learning Status*: Deep recurrent nowcasters (ConvLSTM / PredRNN) deferred as future work due to the required time scope for multi-season video model convergence, the effective resolution limits of 0.1° satellite grids for convective cell initiation, and the strong operational robustness of PySteps without offline training.
2. **Short-Range NWP (3 to 72 h)**:
   - *Baseline*: Raw **NOAA GFS v16** and Climatological Persistence.
   - *ML Benchmark*: Empirical Quantile Mapping (EQM) + LightGBM quantile regression.

### Reported Verification Metrics (from `configs/thresholds.yaml`)
Skill is evaluated across lead times ($+1\text{h}, +2\text{h}, +3\text{h}, +6\text{h}, +12\text{h}, +24\text{h}, +48\text{h}, +72\text{h}$) and IMD thresholds ($2.5, 15.6, 64.5, 115.6, 204.5\text{ mm}$):
- **Categorical Scores**:
  - Probability of Detection: $\text{POD} = \frac{\text{Hits}}{\text{Hits} + \text{Misses}}$
  - False Alarm Ratio: $\text{FAR} = \frac{\text{False Alarms}}{\text{Hits} + \text{False Alarms}}$
  - Critical Success Index: $\text{CSI} = \frac{\text{Hits}}{\text{Hits} + \text{Misses} + \text{False Alarms}}$
  - Frequency Bias: $\text{Bias} = \frac{\text{Hits} + \text{False Alarms}}{\text{Hits} + \text{Misses}}$
  - Heidke Skill Score: $\text{HSS}$
  - Fractions Skill Score: $\text{FSS}$ computed over spatial neighborhood radii $[10, 20, 40, 80]\text{ km}$.
- **Probabilistic Scores**:
  - Brier Score & Brier Skill Score ($\text{BSS}$) relative to climatology.
  - Continuous Ranked Probability Score ($\text{CRPS}$).
  - Reliability Diagrams with $10$ calibration bins.
  - Receiver Operating Characteristic Area Under Curve ($\text{ROC-AUC}$).
- **Confidence Intervals**:
  - All reported metrics include **$95\%$ confidence intervals** calculated via $1,000$ block-bootstrap resamples (block size = $24\text{ hours}$).

---

## 3. Inundation Dual-Layer Honesty Protocol

The U-Net flood surrogate is trained to emulate a 2D local-inertial hydrodynamic solver.
Therefore, high accuracy of the surrogate against the simulator **does not prove skill against reality**.

We strictly separate and report both layers:
1. **Surrogate Emulation Skill (Surrogate vs. Physics Simulator)**:
   - Evaluates how faithfully the U-Net reproduces the numerical shallow-water solver.
   - Metrics: Nash-Sutcliffe Efficiency ($\text{NSE}$), Root Mean Square Error ($\text{RMSE}$ in meters), and Intersection over Union ($\text{IoU}$).
2. **Physical Forecast Skill (Simulator / Surrogate vs. Observed Flood Truth)**:
   - Evaluates simulated flood extents against empirical observation (Sentinel-1 SAR surface water masks and emergency damage mappings).
   - Metrics: $\text{CSI}$, $\text{POD}$, $\text{FAR}$, and $\text{IoU}$ against the observed water boundary.
   - If no cloud-free/SAR acquisition is available for a given timestamp, the system explicitly reports **"No observational truth available for this timestep"** rather than fabricating ground truth.

---

## 4. Frozen Stage 3 Nowcasting Benchmark Findings & Honest Limitations

Rigorous evaluation of PySteps S-PROG and STEPS on held-out events (Cyclone Michaung Dec 2023 and Chennai Nov 2021 Inundation) using operational IMERG Early inputs against IMERG Final truth yields the following empirical findings:

1. **Deterministic CSI Reality (S-PROG vs. Persistence)**:
   - Using pooled contingency tables, climatology CSI is strictly $0.0000$ across all operational thresholds ($2.5, 10, 25, 50\text{ mm/h}$).
   - S-PROG optical flow does **not** demonstrate statistically significant superiority over Persistence in deterministic CSI at $0-1\text{ h}$ ($0.533$ vs $0.568$, $95\%$ CI of difference $[-0.053, -0.013]$), and degrades further at $2-3\text{ h}$.
2. **Probabilistic Skill & Climatology Reference Reality**:
   - While STEPS ensemble achieves lower Brier Score than Persistence, Brier score against persistence (a binary $0/1$ forecast) is a weak reference.
   - When referenced against the training-derived climatological base rate ($\text{BSS}$ vs climatology):
     - At moderate rain ($2.5\text{ mm/h}$), STEPS achieves positive skill ($\text{BSS} \approx +0.45$).
     - At heavier thresholds ($10.0\text{ mm/h}$), probabilistic skill degrades to approximately zero at $2-3\text{ h}$ lead time ($\text{BSS} \approx +0.0002$).
     - At severe thresholds ($25.0\text{ mm/h}$), STEPS BSS vs climatology is **negative** ($\text{BSS} = -0.10$ to $-0.16$). Thus, **no probabilistic skill over climatology is demonstrated for heavy convective rain**.
3. **Reliability and Overconfidence**:
   - Reliability diagrams reveal systemic **overconfidence in the upper forecast bins** (e.g. forecast probabilities in the $0.75-0.97$ range verify against lower observed frequencies of $0.68-0.91$).
   - Raw ensemble probabilities must not be issued directly as public hazard alerts without post-hoc isotonic or Platt probability calibration.

