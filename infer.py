"""
infer.py
Deployment-time inference for NS-Guard. NO ground truth is used anywhere here —
this is the live guardrail:

    image / video / webcam frame
        -> frozen detector  (boxes + class + confidence)
        -> form pairs by rule classes (detected cup x detected laptop, ...)
        -> build the SAME 15-dim feature vector as training (data_pipeline.build_feature)
        -> MLP -> hazard probability
        -> draw the flagged pairs (red boxes + link + label)

There is no label and no baseline comparison here, because at run time there is
no oracle — only the detector's (possibly imperfect) output to guard.

Usage:
    # single image
    python infer.py --source path/to/img.jpg --model best_model.pt

    # folder of images (annotated copies written to --out-dir)
    python infer.py --source path/to/folder --model best_model.pt --out-dir ./annotated

    # video file
    python infer.py --source path/to/clip.mp4 --model best_model.pt --out-dir ./annotated

    # live webcam (camera index 0); press 'q' to quit
    python infer.py --source 0 --model best_model.pt --show
"""

import os
import time
import argparse
from glob import glob

import numpy as np
import cv2
import torch

from data_pipeline import PAIR_RULES, build_feature, apply_scaler
from model import SafetyMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ── detector ────────────────────────────────────────────────────────────────
def run_detector(detector, frame, conf, iou):
    """Run YOLO on one BGR frame, return list of dicts {class_name, bbox, conf}."""
    results = detector.predict(frame, conf=conf, iou=iou, verbose=False)
    r = results[0]
    names = detector.names
    dets = []
    for box in r.boxes:
        cls_id = int(box.cls.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        dets.append({
            "class_name": names[cls_id],
            "bbox":       [x1, y1, x2, y2],
            "conf":       float(box.conf.item()),
        })
    return dets


# ── pairing + MLP ─────────────────────────────────────────────────────────
def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ar = (a[2]-a[0])*(a[3]-a[1]); br = (b[2]-b[0])*(b[3]-b[1])
    u = ar + br - inter
    return inter / u if u > 0 else 0.0


def _dedup(flagged, iou_thr=0.6):
    """Greedy: keep highest-prob pairs; drop later pairs that overlap a kept one
    (same rule + both boxes overlapping). Removes near-duplicate cards."""
    flagged = sorted(flagged, key=lambda f: -f[3])     # high prob first
    kept = []
    for boxA, boxB, rule, p in flagged:
        dup = False
        for kA, kB, krule, _ in kept:
            if krule == rule and _iou(boxA, kA) > iou_thr and _iou(boxB, kB) > iou_thr:
                dup = True; break
        if not dup:
            kept.append((boxA, boxB, rule, p))
    return kept


def predict_pairs(dets, mlp, mean, std, img_w, img_h, flag_thresh,
                  no_conf=False, top_n=None, dedup=True):
    """
    Form all rule-pairs from detections, run the MLP, return flagged pairs
    (boxA, boxB, rule_name, prob), de-duplicated and optionally capped to top_n.
    """
    feats, meta = [], []
    for pair_idx, (classA, classB, _thr, _rt) in enumerate(PAIR_RULES):
        detA = [d for d in dets if d["class_name"] == classA]
        detB = [d for d in dets if d["class_name"] == classB]
        for da in detA:
            for db in detB:
                f = build_feature(da["bbox"], db["bbox"],
                                  da["conf"], db["conf"],
                                  pair_idx, img_w, img_h)
                feats.append(f)
                meta.append((da["bbox"], db["bbox"], f"{classA}-{classB}"))

    if not feats:
        return []

    Xraw = np.stack(feats)
    if no_conf:
        from data_pipeline import zero_confidence
        Xraw = zero_confidence(Xraw)
    X = apply_scaler(Xraw, mean, std)                     # same preprocessing as training
    X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        probs = torch.sigmoid(mlp(X)).cpu().numpy()

    flagged = [(bA, bB, rn, float(p)) for (bA, bB, rn), p in zip(meta, probs)
               if p > flag_thresh]
    if dedup:
        flagged = _dedup(flagged)
    flagged.sort(key=lambda f: -f[3])
    if top_n:
        flagged = flagged[:top_n]
    return flagged


# ── drawing (Info-card style) ────────────────────────────────────────────────
# Colors are BGR (OpenCV).
_C_ACCENT  = (90, 90, 240)      # hazard red
_C_GLOW    = (60, 60, 150)      # darker red for box halo
_C_CARD    = (35, 35, 40)       # info-card / header background
_C_TRACK   = (70, 70, 78)       # confidence-bar track
_C_TEXT    = (235, 235, 235)
_C_CTX     = (200, 200, 200)    # faint context boxes


def _rounded(out, p1, p2, color, r=12, t=2, fill=False):
    x1, y1 = p1; x2, y2 = p2
    if x2 - x1 < 2 * r or y2 - y1 < 2 * r:          # too small for rounding
        cv2.rectangle(out, p1, p2, color, -1 if fill else t, cv2.LINE_AA); return
    if fill:
        cv2.rectangle(out, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(out, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
            cv2.circle(out, (cx, cy), r, color, -1)
    else:
        cv2.line(out, (x1+r,y1), (x2-r,y1), color, t, cv2.LINE_AA)
        cv2.line(out, (x1+r,y2), (x2-r,y2), color, t, cv2.LINE_AA)
        cv2.line(out, (x1,y1+r), (x1,y2-r), color, t, cv2.LINE_AA)
        cv2.line(out, (x2,y1+r), (x2,y2-r), color, t, cv2.LINE_AA)
        for cx, cy, a in [(x1+r,y1+r,180),(x2-r,y1+r,270),(x1+r,y2-r,90),(x2-r,y2-r,0)]:
            cv2.ellipse(out, (cx,cy), (r,r), a, 0, 90, color, t, cv2.LINE_AA)


_CARD_W, _CARD_H = 168, 56


def _overlap_area(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def _place_card(pairbox, avoid, frame_w, frame_h):
    """Pick a card rect (x1,y1,x2,y2) near the pair that least overlaps `avoid`.
    `avoid` is a list of rects (hazard boxes + already-placed cards)."""
    x1, y1, x2, y2 = pairbox
    cw, ch, m = _CARD_W, _CARD_H, 8
    candidates = [
        (x2 + m,        y1),              # right of pair, top-aligned
        (x1 - cw - m,   y1),              # left
        (x1,            y1 - ch - m),     # above
        (x1,            y2 + m),          # below
        (x2 + m,        y2 - ch),         # right, bottom-aligned
        (x1 - cw - m,   y2 - ch),         # left, bottom-aligned
    ]
    best, best_score = None, None
    for cx, cy in candidates:
        cx = max(4, min(int(cx), frame_w - cw - 4))
        cy = max(4, min(int(cy), frame_h - ch - 4))
        rect = (cx, cy, cx + cw, cy + ch)
        score = sum(_overlap_area(rect, a) for a in avoid)
        if best_score is None or score < best_score:
            best, best_score = rect, score
            if score == 0:
                break
    return best


def _info_card(out, rule_name, prob, rect):
    """Dark rounded card with rule name + confidence bar at the given rect."""
    x, y = rect[0], rect[1]
    cw = rect[2] - rect[0]
    _rounded(out, (rect[0], rect[1]), (rect[2], rect[3]), _C_CARD, r=10, fill=True)
    cv2.putText(out, rule_name, (x + 12, y + 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.48, _C_TEXT, 1, cv2.LINE_AA)
    cv2.putText(out, f"{prob:.2f}", (x + cw - 46, y + 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.45, _C_ACCENT, 1, cv2.LINE_AA)
    bx1, bx2, by = x + 12, x + cw - 12, y + 36
    cv2.rectangle(out, (bx1, by), (bx2, by + 9), _C_TRACK, -1)
    cv2.rectangle(out, (bx1, by), (bx1 + int((bx2 - bx1) * float(prob)), by + 9),
                  _C_ACCENT, -1)



def draw(frame, dets, flagged):
    """Info-card rendering: light dim + faint context boxes + hazard cards."""
    h, w = frame.shape[:2]
    out = (frame * 0.6).astype(frame.dtype)          # light overall dim

    # restore brightness inside hazard boxes
    for boxA, boxB, _r, _p in flagged:
        for box in (boxA, boxB):
            x1, y1, x2, y2 = [max(0, int(box[0])), max(0, int(box[1])),
                              min(w, int(box[2])), min(h, int(box[3]))]
            out[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

    # faint context boxes (the detector saw lots; the guardrail selects)
    ctx = out.copy()
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        cv2.rectangle(ctx, (x1, y1), (x2, y2), _C_CTX, 1, cv2.LINE_AA)
    out = cv2.addWeighted(ctx, 0.45, out, 0.55, 0)

    # hazard pairs: glow outline + link first (so cards sit on top)
    haz_rects = []
    for boxA, boxB, rule_name, prob in flagged:
        ca = (int((boxA[0]+boxA[2])/2), int((boxA[1]+boxA[3])/2))
        cb = (int((boxB[0]+boxB[2])/2), int((boxB[1]+boxB[3])/2))
        cv2.line(out, ca, cb, _C_ACCENT, 2, cv2.LINE_AA)
        for box in (boxA, boxB):
            x1, y1, x2, y2 = [int(v) for v in box]
            _rounded(out, (x1, y1), (x2, y2), _C_GLOW, r=12, t=6)   # halo
            _rounded(out, (x1, y1), (x2, y2), _C_ACCENT, r=12, t=2) # crisp edge
            haz_rects.append((x1, y1, x2, y2))

    # header pill (top-left) — reserve its rect so cards avoid it too
    n = len(flagged)
    label = f"NS-Guard   -   {n} hazard{'s' if n != 1 else ''}" if n else "NS-Guard   -   clear"
    font, scale = cv2.FONT_HERSHEY_DUPLEX, 0.46
    (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
    header_rect = (12, 12, 12 + tw + 46, 46)

    # place + draw cards, avoiding boxes, header, and previously placed cards
    placed = [header_rect]
    for boxA, boxB, rule_name, prob in flagged:
        pairbox = (int(min(boxA[0], boxB[0])), int(min(boxA[1], boxB[1])),
                   int(max(boxA[2], boxB[2])), int(max(boxA[3], boxB[3])))
        rect = _place_card(pairbox, haz_rects + placed, w, h)
        _info_card(out, rule_name, prob, rect)
        placed.append(rect)

    # draw header pill on top
    _rounded(out, (header_rect[0], header_rect[1]), (header_rect[2], header_rect[3]),
             _C_CARD, r=9, fill=True)
    cv2.circle(out, (30, 29), 6, _C_ACCENT if n else (90, 180, 90), -1)
    cv2.putText(out, label, (44, 34), font, scale, _C_TEXT, 1, cv2.LINE_AA)
    return out


def process_frame(frame, detector, mlp, mean, std, conf, iou, flag_thresh, no_conf=False, top_n=None):
    """Full per-frame pipeline. Returns (annotated, flagged, timings_ms)."""
    h, w = frame.shape[:2]

    t0 = time.perf_counter()
    dets = run_detector(detector, frame, conf, iou)
    t1 = time.perf_counter()
    flagged = predict_pairs(dets, mlp, mean, std, w, h, flag_thresh, no_conf, top_n)
    t2 = time.perf_counter()

    annotated = draw(frame, dets, flagged)
    timings = {
        "detector_ms": (t1 - t0) * 1000,
        "guard_ms":    (t2 - t1) * 1000,   # pairing + MLP = the symbolic/relational overhead (ILO)
    }
    return annotated, flagged, timings


# ── sources ─────────────────────────────────────────────────────────────────
def load_models(args):
    from ultralytics import YOLO
    detector = YOLO(args.weights)
    ckpt = torch.load(args.model, map_location=DEVICE, weights_only=False)
    mlp = SafetyMLP(input_dim=ckpt["input_dim"]).to(DEVICE)
    mlp.load_state_dict(ckpt["model"])
    mlp.eval()
    return detector, mlp, ckpt["mean"], ckpt["std"], ckpt.get("no_confidence", False)


def run_images(paths, args, detector, mlp, mean, std, no_conf=False):
    os.makedirs(args.out_dir, exist_ok=True)
    for p in paths:
        frame = cv2.imread(p)
        if frame is None:
            print(f"skip (unreadable): {p}")
            continue
        annotated, flagged, t = process_frame(frame, detector, mlp, mean, std,
                                               args.conf, args.iou, args.flag_thresh, no_conf, args.top_n)
        out_path = os.path.join(args.out_dir, os.path.basename(p))
        cv2.imwrite(out_path, annotated)
        print(f"{os.path.basename(p)}: {len(flagged)} hazard pair(s) "
              f"| detector {t['detector_ms']:.1f}ms guard {t['guard_ms']:.2f}ms -> {out_path}")


def run_stream(cap, args, detector, mlp, mean, std, no_conf=False, save_video_to=None):
    writer = None
    if save_video_to:
        os.makedirs(args.out_dir, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save_video_to,
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    guard_times = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated, flagged, t = process_frame(frame, detector, mlp, mean, std,
                                               args.conf, args.iou, args.flag_thresh, no_conf, args.top_n)
        guard_times.append(t["guard_ms"])
        if writer:
            writer.write(annotated)
        if args.show:
            cv2.imshow("NS-Guard", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
        print(f"Saved annotated video -> {save_video_to}")
    if args.show:
        cv2.destroyAllWindows()
    if guard_times:
        print(f"Mean guard overhead (ILO): {np.mean(guard_times):.2f} ms/frame "
              f"over {len(guard_times)} frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="image file, folder, video file, or webcam index (e.g. 0)")
    ap.add_argument("--model", default="best_model.pt", help="trained MLP checkpoint")
    ap.add_argument("--weights", default="yolov8m.pt", help="YOLO weights")
    ap.add_argument("--conf", type=float, default=0.15, help="detector confidence floor")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    ap.add_argument("--flag-thresh", type=float, default=0.5,
                    help="MLP probability above which a pair is drawn as hazard "
                         "(lower e.g. 0.3 to over-warn in a safety setting)")
    ap.add_argument("--out-dir", default="./annotated")
    ap.add_argument("--top-n", type=int, default=None,
                    help="show only the N most confident hazards per image")
    ap.add_argument("--show", action="store_true", help="display window (webcam/video)")
    args = ap.parse_args()

    detector, mlp, mean, std, no_conf = load_models(args)
    src = args.source

    if src.isdigit():                                   # webcam index
        run_stream(cv2.VideoCapture(int(src)), args, detector, mlp, mean, std, no_conf)
    elif os.path.isdir(src):                            # folder of images
        paths = sorted(p for ext in IMG_EXTS
                       for p in glob(os.path.join(src, f"*{ext}")))
        print(f"{len(paths)} images in {src}")
        run_images(paths, args, detector, mlp, mean, std, no_conf)
    elif os.path.isfile(src) and src.lower().endswith(IMG_EXTS):   # single image
        run_images([src], args, detector, mlp, mean, std, no_conf)
    elif os.path.isfile(src):                           # assume video file
        out = os.path.join(args.out_dir, "annotated_" + os.path.basename(src))
        run_stream(cv2.VideoCapture(src), args, detector, mlp, mean, std, no_conf, save_video_to=out)
    else:
        raise SystemExit(f"Unrecognized source: {src}")


if __name__ == "__main__":
    main()
