"""
sweep.py
Scans several pos_weight values, training a fresh MLP for each, and reports
for every value: best val F1 (at 0.5), Average Precision (AP, threshold-free),
and the best-F1 operating threshold. Saves the best-AP model to --out and a
PR-curve overlay plot.

The point: the auto pos_weight (~24) over-pushes recall and tanks precision.
This finds the value that best balances precision/recall, and AP gives you a
single honest number that doesn't depend on the 0.5 threshold.

Run:
    python sweep.py --annotations ./annotations/instances_val2017.json \
        --detections detections.json
"""

import argparse
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score

from train import load_split, train_one, val_probs, save_ckpt


def best_threshold(y, p):
    """Threshold that maximizes F1 on the PR curve."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2*prec*rec / (prec+rec+1e-8)
    i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return (thr[i] if len(thr) else 0.5), f1[i], prec[i], rec[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--detections", default="detections.json")
    ap.add_argument("--values", type=float, nargs="+",
                    default=[1, 3, 5, 8, 12, 24])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--out", default="best_model.pt")
    ap.add_argument("--plot", default="pr_sweep.png")
    args = ap.parse_args()

    data = load_split(args.annotations, args.detections)
    print(f"Train pairs {len(data['ytr'])} | Val pairs {len(data['yva'])} "
          f"| Val positives {int(data['yva'].sum())}\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))

    rows, best_ap, best = [], -1.0, None
    for pw in args.values:
        print(f"===== pos_weight = {pw} =====")
        state, f1_05 = train_one(data, pw, args.epochs, args.patience, verbose=False)
        y, p = val_probs(data, state)
        ap_score = average_precision_score(y, p)
        thr, f1_best, pr_best, rc_best = best_threshold(y, p)
        rows.append((pw, f1_05, ap_score, thr, f1_best, pr_best, rc_best))
        print(f"  F1@0.5 {f1_05:.3f} | AP {ap_score:.3f} | "
              f"bestF1 {f1_best:.3f} @thr {thr:.2f} (P {pr_best:.3f} R {rc_best:.3f})\n")

        prec, rec, _ = precision_recall_curve(y, p)
        plt.plot(rec, prec, label=f"pw={pw} (AP={ap_score:.2f})")

        if ap_score > best_ap:
            best_ap, best = ap_score, (state, pw)

    print("\n pos_weight |  F1@0.5 |   AP   | bestF1 | thr  |  P    |  R")
    print("-----------+---------+--------+--------+------+-------+------")
    for pw, f105, apx, thr, f1b, prb, rcb in rows:
        print(f"   {pw:6.1f}  |  {f105:.3f}  | {apx:.3f}  | {f1b:.3f}  | {thr:.2f} | "
              f"{prb:.3f} | {rcb:.3f}")

    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("PR curves across pos_weight"); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(args.plot, dpi=120, bbox_inches="tight")
    print(f"\nPR overlay saved -> {args.plot}")

    state, pw = best
    save_ckpt(args.out, state, data)
    print(f"Best model (pos_weight={pw}, AP={best_ap:.3f}) saved -> {args.out}")


if __name__ == "__main__":
    main()
