# 3-Minute Hackathon Demonstration Script (SIH26071)

**Target Audience**: Smart India Hackathon Evaluators, IMD/NDMA Operational Panel  
**Time Limit**: Strictly 3 Minutes (180 Seconds)  
**Live URL**: `http://localhost:8000` (Local Single-Page Operational Dashboard)

---

## Minute 1: Problem Statement, Operational Honesty & Architecture (0:00 – 1:00)

### Visual: Slide 1 or Dashboard Header
> *"Respected evaluators, urban flooding in coastal mega-cities like Chennai during extreme Northeast Monsoon cyclones—such as Cyclone Michaung in 2023—presents a dual challenge: short-term convective rainfall moves too fast for traditional NWP, while 3-to-48 hour NWP models suffer from significant location and intensity biases."*

### Key Points to Deliver:
1. **The Mission**: We built **REWS**, an end-to-end, physics-informed early warning and inundation screening system evaluated on real held-out disasters—**Cyclone Michaung (Dec 2023)** and the **Chennai Nov 2021 Floods**—with strict 75-hour temporal purge gaps to prevent data leakage.
2. **Honesty Up Front**:
   - Rather than claiming 99% accuracy with overfitted black-box deep learning, we benchmark established operational baselines: **PySteps optical flow** for nowcasting (0–3h) and **NOAA GFS v16 with Empirical Quantile Mapping (EQM)** for NWP (3–48h).
   - Our nowcaster uses operational **IMERG Early** satellite feeds for input and evaluates against **IMERG Final** ground truth.
   - For inundation, we implement **screening-level HAND (Height Above Nearest Drainage) and SCS-CN runoff**, and explicitly label it as a screening tool, not a full 2D hydrodynamic simulation.

---

## Minute 2: Live Operational Dashboard & Disaster Timeline Demo (1:00 – 2:00)

### Visual: Dashboard Map & Alert Scrubber (`http://localhost:8000`)
*(Action: Time slider loads by default on Dec 03 18:00 UTC at the peak Michaung landfall step, or click 'Play' to scrub)*

> [!NOTE]
> **Venue Internet / Offline Behavior**:
> - The Leaflet JavaScript and CSS assets are bundled locally in `dashboard/leaflet/`, ensuring instant offline initialization with zero CDN timeouts.
> - Base map tiles are fetched live from OpenStreetMap. If presenting in a venue with restricted internet, the base tile layer displays a dark slate background (`#0f172a`), while all dynamic rainfall circles, locality flood markers, and alert panels function 100% offline.

### Script:
> *"Here is our operational dashboard, running live on real exported data from Cyclone Michaung.*
>
> *On initial load, notice the timeline is set to December 3rd at 18:00 UTC during the eyewall landfall peak, and the alert banner is active at **RED ALERT**.*
> *Our system doesn't just output raw numbers: it applies **Isotonic Probability Calibration** (fitted in-sample to demonstrate calibration capability).*
> *Notice the calibrated exceedance probabilities: $P(\ge 50\text{ mm/h}) = 0.63$, $P(\ge 25\text{ mm/h}) = 0.88$, and peak rate of $39.1\text{ mm/h}$. This triggers automated public advisories aligned with IMD color codes.*
>
> *Looking at the map, the dynamic blue-to-red layer represents the real-time satellite rainband as Michaung stalls off Chennai's coast.*
> *The amber boundary indicates our screening-level HAND sub-domain across Chennai city.*
> *Clicking on these locality pins reveals our topographic flood susceptibility predictions against official ground truth from **Sphere India Situation Reports** and local media.*
>
> *For natural low-lying depressions like **Velachery** and **Pallikaranai Marsh**, our HAND model predicts 0.45m inundation, showing good qualitative agreement with observed 1.5–2m floods.*
> *Crucially, we report our failures honestly: in **T. Nagar**, observed flooding was severe, but HAND predicted moderate depth because HAND is purely topographic and cannot simulate urban stormwater drain siltation and culvert blockages."*

---

## Minute 3: Scientific Verification, Real Metrics & Dissemination (2:00 – 3:00)

### Visual: Click 'Nowcast' and 'GFS EQM' Tabs, then 'CAP Alert' Tab

### Script:
> *"Let's examine our quantitative verification results.*
>
> *In **Stage 3 Nowcasting**, using pooled contingency tables over 94 cycles:
> - PySteps S-PROG achieves CSI of 0.533 at 0–1h for light rain, but we document that it does not beat persistence (CSI 0.568), proving that naive claims of optical flow superiority are physically unsubstantiated.
> - Probabilistically, STEPS achieves a Brier Skill Score of $+0.45$ over climatology at 2.5 mm/h, but degrades to zero at 10 mm/h and becomes negative at 25 mm/h, proving that satellite nowcasting alone has limited heavy-convective skill.*
>
> *In **Stage 4 NWP**, applying Empirical Quantile Mapping (EQM) to NOAA GFS v16 yields a **statistically significant CSI gain of $+0.022$** across both Michaung and the Nov 2021 disaster with 95% bootstrap confidence intervals excluding zero.*
>
> *Finally, in the **CAP Alert Tab**, our system generates fully validated **OASIS CAP 1.2 XML** emergency alerts ready for NDMA's Sachet portal, along with machine-drafted Tamil and Hindi templates explicitly flagged for native-speaker operational review.*
>
> *In summary, SIH26071 delivers a working, rigorously validated, and scientifically honest early-warning pipeline for Chennai. Thank you."*

---

## Quick Reference Q&A for Jurors

- **Q: Why didn't you train a ConvLSTM or PredRNN nowcaster?**  
  *A: A 24-hour hackathon cannot converge a multi-season video model without severe blurring, and 0.1° satellite grids cannot resolve convective cell initiation. PySteps provides an operationally robust physical benchmark.*
- **Q: Why not use a 2D hydrodynamic solver (like LISFLOOD or SWMM)?**  
  *A: True 2D hydrodynamic simulation requires sub-meter LiDAR DEMs, underground stormwater pipe networks, and tidal surge boundaries. A screening-level HAND/SCS-CN model provides fast, regional flood susceptibility without pretending to be a physics solver.*
- **Q: What happened to IMD gauge calibration?**  
  *A: We attempted automated daily gridded ingestion via `imdlib`, but IMD's servers timed out without institutional credentials. We honestly report satellite IMERG as our ground reference.*
