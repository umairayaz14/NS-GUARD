"""
evaluate.py
Compares two baselines and the trained MLP against GT-derived TRUE labels:
  * Baseline (naive): single global distance threshold on detected boxes.
  * Baseline (multi): the FULL multi-factor rule on detected boxes
                      (the original symbolic NS-Guard) — the STRONG baseline.
  * MLP: learned head on detector features (geometry + confidence + class).

Reports overall + PER-RULE metrics, threshold-free AP, a threshold sweep, a PR
curve, and the disagreement analysis. Honors the confidence ablation stored in
the checkpoint.

Run:
    python evaluate.py --annotations ./annotations/instances_val2017.json \
        --detections detections.json --model best_model.pt --flag-thresh 0.5
"""

import argparse
import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import average_precision_score, precision_recall_curve

from data_pipeline import extract_pairs, apply_scaler, zero_confidence, PAIR_RULES, N_PAIR_TYPES
from model import SafetyMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RULE_NAMES = [f"{a}-{b}" for a, b, *_ in PAIR_RULES]


def _prf(preds, labels):
    preds, labels = np.array(preds), np.array(labels)
    tp = ((preds==1)&(labels==1)).sum(); fp = ((preds==1)&(labels==0)).sum()
    fn = ((preds==0)&(labels==1)).sum(); tn = ((preds==0)&(labels==0)).sum()
    pr = tp/(tp+fp+1e-8); rc = tp/(tp+fn+1e-8)
    return pr, rc, 2*pr*rc/(pr+rc+1e-8), fp/(fp+tn+1e-8)


def overall(preds, labels, name):
    p, r, f1, fpr = _prf(preds, labels)
    print(f"  {name:28s} | P {p:.3f} | R {r:.3f} | F1 {f1:.3f} | FPR {fpr:.3f}")


def image_level(preds, image_ids, labels, name):
    ip, il = defaultdict(int), defaultdict(int)
    for pred, i, lab in zip(preds, image_ids, labels):
        ip[i] = max(ip[i], int(pred)); il[i] = max(il[i], int(lab))
    ids = list(ip.keys())
    p, r, f1, fpr = _prf(np.array([ip[i] for i in ids]), np.array([il[i] for i in ids]))
    print(f"  {name:28s} | P {p:.3f} | R {r:.3f} | F1 {f1:.3f} | FPR {fpr:.3f}")


def per_rule(preds, labels, rule_idx, name, probs=None):
    print(f"\nPer-rule — {name}")
    for i, rname in enumerate(RULE_NAMES):
        m = rule_idx == i
        if m.sum() == 0:
            continue
        p, r, f1, fpr = _prf(preds[m], labels[m])
        ap = ""
        if probs is not None and labels[m].sum() > 0:
            ap = f" | AP {average_precision_score(labels[m], probs[m]):.3f}"
        print(f"  {rname:14s} (n={int(m.sum()):5d}, pos={int(labels[m].sum()):4d}) "
              f"| P {p:.3f} | R {r:.3f} | F1 {f1:.3f} | FPR {fpr:.3f}{ap}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--detections", default="detections.json")
    ap.add_argument("--model", default="best_model.pt")
    ap.add_argument("--flag-thresh", type=float, default=0.5)
    ap.add_argument("--split", choices=["all", "train", "val"], default="val",
                    help="which image split to score on (val = honest held-out; "
                         "all = full set, use for sparse per-rule numbers)")
    ap.add_argument("--plot", default="pr_eval.png")
    args = ap.parse_args()

    features, labels, image_ids, baselines = extract_pairs(args.annotations, args.detections)

    # Reproduce the SAME image split train.py used (seed 42, 20% val), then filter.
    if args.split != "all":
        from sklearn.model_selection import train_test_split
        uniq = np.unique(image_ids)
        tr_ids, va_ids = train_test_split(uniq, test_size=0.2, random_state=42)
        keep_ids = set(va_ids if args.split == "val" else tr_ids)
        m = np.array([i in keep_ids for i in image_ids])
        features, labels, image_ids = features[m], labels[m], image_ids[m]
        baselines = {k: v[m] for k, v in baselines.items()}
        print(f"\n[split={args.split}] {len(labels)} pairs, "
              f"{int(labels.sum())} positives")

    rule_idx = np.argmax(features[:, 12:12 + N_PAIR_TYPES], axis=1)

    bn = baselines["naive"].astype(int)
    bm = baselines["multi"].astype(int)

    print(f"\n=== Overall (pair-level)  [split={args.split}] ===")
    overall(bn, labels, "Baseline naive (1 threshold)")
    overall(bm, labels, "Baseline multi (full rule)")

    # ---- MLP ----
    ckpt = torch.load(args.model, map_location=DEVICE, weights_only=False)
    model = SafetyMLP(input_dim=ckpt["input_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["model"]); model.eval()
    feats = zero_confidence(features) if ckpt.get("no_confidence") else features
    if ckpt.get("no_confidence"):
        print("  (model trained WITHOUT confidence — ablation)")
    X = torch.tensor(apply_scaler(feats, ckpt["mean"], ckpt["std"]),
                     dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        probs = torch.sigmoid(model(X)).cpu().numpy()
    mlp = (probs > args.flag_thresh).astype(int)
    overall(mlp, labels, f"MLP @ thr={args.flag_thresh}")

    ap_score = average_precision_score(labels, probs)
    print(f"\nMLP Average Precision (threshold-free): {ap_score:.3f}")

    print("\n=== Image-level (OR aggregation) ===")
    image_level(bn,  image_ids, labels, "Baseline naive")
    image_level(bm,  image_ids, labels, "Baseline multi")
    image_level(mlp, image_ids, labels, f"MLP @ thr={args.flag_thresh}")

    # ---- per-rule ----
    per_rule(bn,  labels, rule_idx, "Baseline naive")
    per_rule(bm,  labels, rule_idx, "Baseline multi")
    per_rule(mlp, labels, rule_idx, f"MLP @ thr={args.flag_thresh}", probs=probs)

    # ---- threshold sweep ----
    print("\nMLP threshold sweep (pair-level):")
    print("  thr  |  P    |  R    |  F1   | FPR")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        p, r, f1, fpr = _prf((probs > t).astype(int), labels)
        print(f"  {t:.1f}  | {p:.3f} | {r:.3f} | {f1:.3f} | {fpr:.3f}")

    # ---- PR curve: MLP curve + both baselines as points ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prec, rec, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"MLP (AP={ap_score:.2f})")
    for preds, nm, c in [(bn, "Baseline naive", "red"), (bm, "Baseline multi", "orange")]:
        p, r, _, _ = _prf(preds, labels)
        plt.scatter([r], [p], c=c, zorder=5, label=nm)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("MLP vs baselines (pair-level)")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(args.plot, dpi=120, bbox_inches="tight")
    print(f"\nPR curve saved -> {args.plot}")

    # ---- disagreement vs the STRONG baseline ----
    dis = np.where(mlp != bm)[0]
    print(f"\nMLP vs STRONG baseline: disagree on {len(dis)} / {len(labels)} pairs")
    if len(dis):
        print(f"On disagreements: MLP correct {(mlp[dis]==labels[dis]).sum()}, "
              f"strong-baseline correct {(bm[dis]==labels[dis]).sum()}")


if __name__ == "__main__":
    main()