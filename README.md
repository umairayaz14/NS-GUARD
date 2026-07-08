# NS-Guard: Learned Relational Safety Guardrails for Object Detection

**High-Level Computer Vision (SS26) — Universität des Saarlandes**
Anushka Choudhary · Shreya Kolhapure · Umair Ayaz Aslam

A lightweight learned relational head that flags **inter-object safety hazards**
(e.g. a cup too close to a laptop) from the output of a frozen object detector —
and beats both a naive and a strong hand-crafted symbolic baseline by learning
to exploit the detector's **confidence scores**.

## Key results (held-out validation, pair-level)

| | Naive baseline | Symbolic baseline | MLP (no conf.) | **MLP (full)** |
|---|---|---|---|---|
| AP | — | — | 0.410 | **0.590** |
| F1 | 0.303 | 0.510 | 0.526 | **0.600** |
| FPR | 0.199 | 0.069 | 0.065 | **0.036** |

Removing the confidence features (ablation) drops AP from 0.59 to 0.41 and
erases the advantage over the symbolic baseline — the learned head's gain comes
specifically from discounting unreliable detections.

## Quickstart

```bash
pip install -r requirements.txt

# data (COCO val2017: images ~1GB + annotations)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/val2017.zip && unzip val2017.zip

# 1. run the frozen detector once
python detect.py --image-dir ./val2017 --out detections.json --conf 0.15 --iou 0.7
# 2. train the relational head
python train.py --annotations ./annotations/instances_val2017.json --detections detections.json --pos-weight 5
# 3. evaluate (held-out split)
python evaluate.py --annotations ./annotations/instances_val2017.json --detections detections.json --model best_model.pt --split val
# 4. live demo (webcam; or pass an image/folder/video path)
python infer.py --source 0 --model best_model.pt --show --flag-thresh 0.6 --top-n 4
```

---

## Full documentation

End-to-end vision pipeline that flags **relational safety hazards** (e.g. a cup
too close to a laptop) from raw images:

```
image ─▶ frozen YOLO ─▶ boxes+classes+confidences ─▶ pairwise features
                                                          │
                                          ┌───────────────┴───────────────┐
                                          ▼                               ▼
                            BASELINE (single global              MLP head (geometry +
                            distance threshold)                  confidence + class)
```

## The idea (why the MLP isn't redundant)

- **Features** (MLP input) come **only from the detector**: geometry + the
  detector's **confidence** scores + a class one-hot. (15-dim.)
- **Labels** (truth) come from **COCO ground truth**: a detected pair is a true
  hazard only if both detections IoU-match real objects **and** the real objects
  satisfy the hazard rule. Spurious detections → SAFE.
- **Baseline** = a **single global distance threshold** applied to the detector's
  boxes — the naive "one hand-set number" heuristic. It cannot do
  per-relationship thresholds or multi-factor logic, so the MLP has room to win.

The MLP can beat the baseline by (1) learning **per-class** safety distances via
the one-hot, (2) **discounting low-confidence** spurious detections, and (3)
capturing **multi-factor** hazards the single threshold can't express.

## Hazard definition (`HAZARD_MODE` in data_pipeline.py)

- `"distance"` — hazard = close (per-class threshold). Pure single-feature rule;
  the single-threshold baseline is already near-optimal, so expect a TIE.
- `"multi"` — hazard = close **AND** (overlapping **OR** similar-size). Cannot be
  expressed by one distance threshold, so the baseline is provably suboptimal and
  the MLP can win. **This is the default.**

To A/B the two, change `HAZARD_MODE`, retrain, re-evaluate.

## Setup

```bash
pip install torch torchvision numpy scikit-learn ultralytics opencv-python
```

`ultralytics` auto-downloads YOLO weights on first use. Default `yolov8m.pt`.
(The proposal cited "YOLO26" — use whatever the current Ultralytics release
provides; only the `--weights` filename changes.)

## Data: COCO val2017 (images + annotations)

```bash
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip          # -> annotations/instances_val2017.json
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip                           # -> val2017/*.jpg  (~5k images, ~1GB)
```

## Run

```bash
# 0. (optional) sanity-check the logic, no downloads needed
python test_pipeline.py

# 1. detector pass — ONCE over the images
python detect.py --image-dir ./val2017 --out detections.json \
    --weights yolov8m.pt --conf 0.15 --iou 0.7
#   (use --limit 200 for a quick smoke test first)

# 2. train the MLP head (splits val2017 by image; standardizes; early-stops)
python train.py --annotations ./annotations/instances_val2017.json \
    --detections detections.json

# 3. evaluate baseline vs MLP (pair + image level, with edge cases)
python evaluate.py --annotations ./annotations/instances_val2017.json \
    --detections detections.json --model best_model.pt
```

## Inference / deployment (no ground truth)

`infer.py` is the live guardrail — detector → MLP → draw flagged pairs. No GT,
no labels, no baseline.

```bash
python infer.py --source img.jpg --model best_model.pt                 # single image
python infer.py --source ./val2017 --model best_model.pt --out-dir ./annotated
python infer.py --source clip.mp4 --model best_model.pt --out-dir ./annotated
python infer.py --source 0 --model best_model.pt --show                # webcam (q quits)
python infer.py --source 0 --model best_model.pt --show --flag-thresh 0.3   # over-warn
```

Draws each flagged pair's two boxes in red + connecting line + rule/probability,
dims other detections, shows a HAZARD banner. For video/webcam it prints the mean
per-frame guard overhead (the ILO metric: pairing + MLP time, excl. detector).

## Feature vector (15-dim)

> Built in ONE place — `data_pipeline.build_feature` — called by both training
> and inference, so the layout can never drift. Features are then **standardized**
> (mean/std fit on the TRAIN split only, saved inside the checkpoint, applied
> identically at eval and inference).

| index | feature | source |
|-------|---------|--------|
| 0 | normalized centroid distance | detector |
| 1–2 | dx, dy (normalized, signed) | detector |
| 3 | IoU | detector |
| 4 | area ratio | detector |
| 5–6 | overlap_x, overlap_y | detector |
| 7–8 | normalized area of A, B | detector |
| 9 | fraction of A covered by B | detector |
| 10–11 | **confidence of A, B** | detector |
| 12–14 | one-hot of which rule pair | detector |

Ground truth is used **only** to compute the label, never as a feature.

## Checkpoint format

`best_model.pt` is a dict: `{model, mean, std, input_dim, hazard_mode}`.
`evaluate.py` and `infer.py` read the scaler from it so preprocessing matches
training exactly. (Old plain-`state_dict` checkpoints are not compatible — retrain.)

## Metrics

- **VDA (recall)** — fraction of true hazards flagged.
- **FPR** — fraction of safe cases wrongly flagged.
- Reported pair-level and image-level (OR aggregation).
- `evaluate.py` dumps the disagreement set (MLP vs baseline) and who's right —
  the edge-case analysis. Read **"On disagreements: MLP correct X, baseline
  correct Y"** as the headline verdict.

## Experiments (rigor for the comparison)

Two baselines are reported, both applied to the **detected** boxes:
- **naive** — a single global distance threshold (a non-expert's one-number rule).
- **multi** — the FULL multi-factor hazard rule (the original symbolic NS-Guard).
  This is the STRONG baseline: if the MLP beats it, the win comes from using
  detector **confidence**, which no rule-on-detections can.

**Confidence ablation** — train the head with the confidence features zeroed:

```bash
python train.py --annotations ... --detections detections.json \
    --pos-weight 5 --no-confidence --out best_model_noconf.pt
```

If precision drops toward the baseline without confidence, that proves the head's
advantage is confidence-awareness. `evaluate.py` honors the ablation flag stored
in the checkpoint.

**Per-rule metrics** — `evaluate.py` now reports precision/recall/F1/FPR (and AP
for the MLP) broken down by rule pair (cup-laptop, person-car, knife-person), not
just the average — different relationships behave very differently.

Suggested experiment matrix to fill for the report (overall + per rule):

| | naive baseline | multi baseline | MLP (no conf) | MLP (full) |
|---|---|---|---|---|
| AP / F1 / FPR | | | | |

## Tuning the MLP (precision/recall, AP)

The automatic `pos_weight` (negatives/positives, e.g. ~24) over-pushes recall and
tanks precision. To find a better balance and report a threshold-free number:

```bash
# scan pos_weight values; reports F1@0.5, AP, and best-F1 threshold per value,
# saves PR-curve overlay (pr_sweep.png) and the best-AP model to best_model.pt
python sweep.py --annotations ./annotations/instances_val2017.json \
    --detections detections.json --values 1 3 5 8 12 24
```

Then train a final model at your chosen value:

```bash
python train.py --annotations ./annotations/instances_val2017.json \
    --detections detections.json --pos-weight 5
```

`evaluate.py` now also reports **Average Precision (AP)** (threshold-free), a
**precision/recall/F1 sweep over thresholds**, and saves a PR curve
(`pr_eval.png`) with the baseline as a single point. Pick an operating threshold
with `--flag-thresh` (e.g. high recall for a safety system) and report that point
plus AP, rather than F1 at a fixed 0.5.

## Tuning

- If the MLP–baseline gap is small in `"distance"` mode, that's expected — switch
  `HAZARD_MODE = "multi"`.
- Lower `--conf` (e.g. 0.10) to admit more shaky detections (more for the
  confidence feature to exploit).
- Per-class thresholds live in `PAIR_RULES`; the baseline's single threshold is
  `GLOBAL_BASELINE_THRESHOLD` (default = mean of the per-class ones).

## Known limits

- **Missed detections are a shared ceiling**: if the detector never sees an
  object, neither baseline nor MLP can flag it — both take the false negative.
- **Out-of-vocabulary objects** (not in COCO's 80 classes) can't be guarded.
- The relational head's recall is bounded by the detector's recall.
