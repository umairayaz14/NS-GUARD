"""
data_pipeline.py
Builds train/eval data for the relational safety classifier.

KEY DESIGN
----------
* FEATURES (MLP input) come ONLY from the DETECTOR's boxes
      geometry(10) + confidence(2) + class one-hot(N_PAIR_TYPES)  -> 15-dim default.
* LABELS (true hazard) come from the COCO GROUND-TRUTH boxes, via IoU matching.
      A spurious detection (no GT match) is labelled SAFE.
* BASELINE is a SINGLE GLOBAL distance threshold on the DETECTOR's boxes
      i.e. the naive "trust the detector + one hand-set threshold" heuristic.
      It deliberately CANNOT do per-relationship thresholds or multi-factor
      logic, so there is a real gap for the learned MLP to close.

Two hazard definitions (HAZARD_MODE):
  "distance" : close (per-class threshold).                      [single feature]
  "multi"    : close AND (overlapping OR similar-size).          [needs interaction]
The "multi" definition cannot be expressed by a single distance threshold, so
the single-threshold baseline is provably suboptimal there.
"""

import json
import numpy as np
from collections import defaultdict

try:
    from torch.utils.data import Dataset
except ImportError:          # lets this module import without torch (for tests)
    Dataset = object


# ── Hazard rules ───────────────────────────────────────────────────────────
# (classA, classB, per-class normalized threshold (fraction of image diagonal), rule_type)
PAIR_RULES = [
    ("cup",    "laptop",  0.15,  "proximity"),
    ("person", "car",     0.10,  "proximity"),
    ("knife",  "person",  0.15,  "proximity"),
]

HAZARD_MODE = "multi"          # "distance" or "multi"  (the LABEL definition)

# The baseline is a single global threshold (mean of the per-class ones by default).
# A non-expert would hand-set one number; the MLP must learn the per-class /
# multi-factor structure that this single number can't capture.
GLOBAL_BASELINE_THRESHOLD = float(np.mean([r[2] for r in PAIR_RULES]))  # ~0.133

# "multi" sub-conditions
OVERLAP_IOU         = 0.0      # any bounding-box overlap counts as "overlapping"
SIMILAR_AREA_RATIO  = 0.5      # min/max area > this counts as "similar size"

IOU_MATCH_THRESH = 0.5         # detection is a real object if IoU>this with same-class GT
N_PAIR_TYPES = len(PAIR_RULES)
FEATURE_DIM  = 10 + 2 + N_PAIR_TYPES
CONF_IDX     = [10, 11]         # confidence feature columns (for the ablation)


def zero_confidence(X):
    """Return a copy of X with the confidence columns zeroed (ablation)."""
    X = X.copy()
    X[:, CONF_IDX] = 0.0
    return X


# ── COCO / detections loading ──────────────────────────────────────────────
def load_coco_gt(annotation_path):
    with open(annotation_path) as f:
        coco = json.load(f)
    name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
    img_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)
    img_dims = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}
    return name_to_id, img_to_anns, img_dims


def load_detections(detections_path):
    with open(detections_path) as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        try:
            out[int(k)] = v
        except ValueError:
            out[k] = v
    return out


# ── Geometry ───────────────────────────────────────────────────────────────
def bbox_xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def iou_xyxy(boxA, boxB):
    x1a, y1a, x2a, y2a = boxA
    x1b, y1b, x2b, y2b = boxB
    ix1, iy1 = max(x1a, x1b), max(y1a, y1b)
    ix2, iy2 = min(x2a, x2b), min(y2a, y2b)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    areaA = max(0.0, x2a - x1a) * max(0.0, y2a - y1a)
    areaB = max(0.0, x2b - x1b) * max(0.0, y2b - y1b)
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def compute_geometric_features(boxA, boxB, img_w, img_h):
    x1a, y1a, x2a, y2a = boxA
    x1b, y1b, x2b, y2b = boxB
    cxa, cya = (x1a + x2a) / 2, (y1a + y2a) / 2
    cxb, cyb = (x1b + x2b) / 2, (y1b + y2b) / 2

    diag = np.sqrt(img_w ** 2 + img_h ** 2)
    centroid_dist = np.sqrt((cxa - cxb) ** 2 + (cya - cyb) ** 2) / diag
    dx = (cxb - cxa) / img_w
    dy = (cyb - cya) / img_h

    ix1, iy1 = max(x1a, x1b), max(y1a, y1b)
    ix2, iy2 = min(x2a, x2b), min(y2a, y2b)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    areaA = (x2a - x1a) * (y2a - y1a)
    areaB = (x2b - x1b) * (y2b - y1b)
    union = areaA + areaB - inter
    iou = inter / union if union > 0 else 0.0
    area_ratio = min(areaA, areaB) / (max(areaA, areaB) + 1e-6)
    overlap_x = max(0, min(x2a, x2b) - max(x1a, x1b)) / img_w
    overlap_y = max(0, min(y2a, y2b) - max(y1a, y1b)) / img_h
    norm_areaA = areaA / (img_w * img_h)
    norm_areaB = areaB / (img_w * img_h)

    return np.array([
        centroid_dist, dx, dy, iou, area_ratio,
        overlap_x, overlap_y, norm_areaA, norm_areaB,
        min(inter / (areaA + 1e-6), 1.0),
    ], dtype=np.float32)


def build_feature(boxA, boxB, confA, confB, pair_idx, img_w, img_h):
    """SINGLE SOURCE OF TRUTH for the feature layout — used by train AND infer."""
    geom = compute_geometric_features(boxA, boxB, img_w, img_h)
    conf = np.array([confA, confB], dtype=np.float32)
    onehot = np.zeros(N_PAIR_TYPES, dtype=np.float32)
    onehot[pair_idx] = 1.0
    return np.concatenate([geom, conf, onehot])


# ── Labels ──────────────────────────────────────────────────────────────────
def _is_close(boxA, boxB, threshold, img_w, img_h):
    x1a, y1a, x2a, y2a = boxA
    x1b, y1b, x2b, y2b = boxB
    cxa, cya = (x1a + x2a) / 2, (y1a + y2a) / 2
    cxb, cyb = (x1b + x2b) / 2, (y1b + y2b) / 2
    diag = np.sqrt(img_w ** 2 + img_h ** 2)
    dist = np.sqrt((cxa - cxb) ** 2 + (cya - cyb) ** 2) / diag
    return dist < threshold


def hazard_label(boxA, boxB, threshold, img_w, img_h, mode=None):
    """TRUE hazard label (computed on GT boxes)."""
    mode = mode or HAZARD_MODE
    if not _is_close(boxA, boxB, threshold, img_w, img_h):
        return 0
    if mode == "distance":
        return 1
    if mode == "multi":
        overlapping = iou_xyxy(boxA, boxB) > OVERLAP_IOU
        areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
        similar = (min(areaA, areaB) / (max(areaA, areaB) + 1e-6)) > SIMILAR_AREA_RATIO
        return int(overlapping or similar)
    raise ValueError(f"unknown HAZARD_MODE {mode}")


def baseline_predict(boxA, boxB, img_w, img_h):
    """Naive baseline: single GLOBAL distance threshold on detected boxes."""
    return int(_is_close(boxA, boxB, GLOBAL_BASELINE_THRESHOLD, img_w, img_h))


# ── GT matching ──────────────────────────────────────────────────────────────
def match_to_gt(det_box, gt_boxes):
    best_iou, best_box = 0.0, None
    for gt in gt_boxes:
        i = iou_xyxy(det_box, gt)
        if i > best_iou:
            best_iou, best_box = i, gt
    return best_box if best_iou >= IOU_MATCH_THRESH else None


# ── Standardization (fit on TRAIN only; apply everywhere) ────────────────────
def fit_scaler(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_scaler(X, mean, std):
    return ((X - mean) / std).astype(np.float32)


# ── Main extraction ──────────────────────────────────────────────────────────
def extract_pairs(annotation_path, detections_path):
    name_to_id, img_to_anns, img_dims = load_coco_gt(annotation_path)
    detections = load_detections(detections_path)

    features, labels, image_ids = [], [], []
    base_naive, base_multi = [], []

    for pair_idx, (classA, classB, threshold, _rt) in enumerate(PAIR_RULES):
        if classA not in name_to_id or classB not in name_to_id:
            print(f"Warning: {classA} or {classB} not in COCO categories — skipping")
            continue
        idA, idB = name_to_id[classA], name_to_id[classB]

        for img_id, dets in detections.items():
            detA = [d for d in dets if d["class_name"] == classA]
            detB = [d for d in dets if d["class_name"] == classB]
            if not detA or not detB:
                continue
            img_w, img_h = img_dims.get(img_id, (640, 480))
            anns = img_to_anns.get(img_id, [])
            gtA = [bbox_xywh_to_xyxy(a["bbox"]) for a in anns if a["category_id"] == idA]
            gtB = [bbox_xywh_to_xyxy(a["bbox"]) for a in anns if a["category_id"] == idB]

            for da in detA:
                for db in detB:
                    boxA, boxB = da["bbox"], db["bbox"]
                    feat = build_feature(boxA, boxB, da["conf"], db["conf"],
                                         pair_idx, img_w, img_h)

                    # Two baselines, both on DETECTED boxes:
                    #  naive = single global distance threshold
                    #  multi = the FULL hazard rule (the original symbolic NS-Guard)
                    bn = baseline_predict(boxA, boxB, img_w, img_h)
                    bm = hazard_label(boxA, boxB, threshold, img_w, img_h)

                    mA, mB = match_to_gt(boxA, gtA), match_to_gt(boxB, gtB)
                    if mA is None or mB is None:
                        label = 0
                    else:
                        label = hazard_label(mA, mB, threshold, img_w, img_h)

                    features.append(feat)
                    labels.append(label)
                    image_ids.append(img_id)
                    base_naive.append(bn)
                    base_multi.append(bm)

    if not features:
        raise RuntimeError("No pairs built — check detections/annotations alignment.")

    features   = np.stack(features)
    labels     = np.array(labels, dtype=np.float32)
    image_ids  = np.array(image_ids)
    baselines  = {"naive": np.array(base_naive, dtype=np.float32),
                  "multi": np.array(base_multi, dtype=np.float32)}

    print(f"HAZARD_MODE = {HAZARD_MODE} | naive baseline thr = {GLOBAL_BASELINE_THRESHOLD:.3f}")
    print(f"Total detected pairs: {len(labels)}")
    print(f"True violations:      {labels.sum():.0f} ({100*labels.mean():.1f}%)")
    print(f"Naive baseline fires: {baselines['naive'].sum():.0f} "
          f"({100*baselines['naive'].mean():.1f}%)")
    print(f"Multi baseline fires: {baselines['multi'].sum():.0f} "
          f"({100*baselines['multi'].mean():.1f}%)")
    return features, labels, image_ids, baselines


class PairDataset(Dataset):
    def __init__(self, features, labels):
        import torch
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels,   dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
