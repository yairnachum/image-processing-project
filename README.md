# Image Processing / Vision Course Project

Evaluating the **robustness** of image processing and vision algorithms under image distortions, and the effect of two recovery strategies: classical image enhancement (pre-processing) and model fine-tuning.

## Team

_To be filled in (names + emails)._

## Project structure

For each chosen task on the chosen dataset, we will produce four measurements:

1. **Baseline** — performance on clean images (vs. ground truth where available; otherwise used as pseudo-GT for later stages).
2. **Distorted** — performance on images degraded by 3 distortions, swept across intensities and reported per SNR.
3. **Enhanced (restored)** — performance after applying classical enhancement methods on the distorted images.
4. **Fine-tuned** — performance of a DL model fine-tuned on distorted images.

All performance reported **per class** and **per SNR**.

## Decisions

These tables are placeholders, to be filled in during weeks 2–3.

### Dataset

| Choice | Link | Why |
|--------|------|-----|
| _TBD_  |      |     |

### Tasks (3, at least one DL + low-level / high-level mix)

| # | Task | Model / Algorithm | Metric |
|---|------|-------------------|--------|
| 1 | _TBD_ |                   |        |
| 2 | _TBD_ |                   |        |
| 3 | _TBD_ |                   |        |

### Distortions (3) and enhancements (per distortion)

| # | Distortion | Enhancement |
|---|------------|-------------|
| 1 | _TBD_      | _TBD_       |
| 2 | _TBD_      | _TBD_       |
| 3 | _TBD_      | _TBD_       |

## Results

To be added per stage:

- **Baseline** — per-class metric tables, sample visualizations.
- **Distorted** — degradation tables, SNR sweep curves, before/after grids.
- **Enhanced** — comparison tables vs. distorted, side-by-side grids.
- **Fine-tuned** — comparison tables vs. distorted baseline.

## Repository layout (planned)

```
.
├── README.md     # this file = the project report
├── data/         # (gitignored) raw / distorted / restored
├── notebooks/    # EDA + experiments
├── src/          # reusable code (distortions, restoration, eval)
├── outputs/      # tables, figures, sample grids
└── runs/         # (gitignored) model checkpoints / training runs
```

## Weekly plan 

| Wk | Task | Artifact |
|----|------|----------|
| 1  | Form team, open Git, register | Opened GitHub repo, entry in course project table |
| 2  | Research & select dataset, distortions, tasks | Decisions tables in README |
| 3  | Research & select methods and enhancements | Decisions tables in README |
| 4  | Download data, visualize images and annotations | EDA code, sample image grid in README |
| 5  | Run methods/models on clean data | Folder with outcomes/labels |
| 6  | Measure performance vs GT | Results tables, per-class viz |
| 7  | Apply distortions, save data | Distortion code, before/after visuals |
| 8  | Run models on distorted, measure degradation | Perf tables, comparison visuals |
| 9  | Apply enhancements, measure | Side-by-side grids, perf comparison |
| 10 | Fine-tune model(s) | FT code, checkpoint/weights |
| 11 | Measure fine-tuned performance | Results table, visualization |
| 12 | Polish README | Rich, detailed README |
| 13 | Prepare PPT, review repo | Slides (PPT + PDF), final repo |
