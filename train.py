"""
train.py
Trains SafetyMLP on detector features (GT labels). Standardizes features
(fit on TRAIN only, saved in the checkpoint), weighted loss for imbalance,
early stopping on val F1.

  --pos-weight   override the automatic n_neg/n_pos. The auto value can be huge
                 (e.g. 24) and over-pushes recall at the cost of precision;
                 lower it (try 3-8) for a better precision/recall balance.
                 Use sweep.py to scan several values at once.

Run:
    python train.py --annotations ./annotations/instances_val2017.json \
        --detections detections.json --pos-weight 5
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from data_pipeline import (extract_pairs, PairDataset, fit_scaler, apply_scaler,
                           zero_confidence, HAZARD_MODE)
from model import SafetyMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_split(annotations, detections, test_size=0.2, seed=42, no_confidence=False):
    """Extract pairs, split by image, standardize (fit on train). Returns a dict."""
    features, labels, image_ids, _ = extract_pairs(annotations, detections)
    unique_ids = np.unique(image_ids)
    train_ids, val_ids = train_test_split(unique_ids, test_size=test_size, random_state=seed)
    tr = np.isin(image_ids, train_ids); va = np.isin(image_ids, val_ids)
    Xtr, ytr = features[tr], labels[tr]
    Xva, yva = features[va], labels[va]
    if no_confidence:                      # ablation: remove the confidence signal
        Xtr, Xva = zero_confidence(Xtr), zero_confidence(Xva)
    mean, std = fit_scaler(Xtr)
    return {"Xtr": apply_scaler(Xtr, mean, std), "ytr": ytr,
            "Xva": apply_scaler(Xva, mean, std), "yva": yva,
            "mean": mean, "std": std, "input_dim": features.shape[1],
            "no_confidence": no_confidence}


def resolve_pos_weight(y, override):
    n_neg = (y == 0).sum(); n_pos = (y == 1).sum()
    if n_pos == 0:
        raise ValueError("No positive examples — loosen thresholds or lower --conf.")
    auto = n_neg / n_pos
    w = auto if (override is None or override < 0) else override
    print(f"pos_weight = {w:.2f}  (auto would be {auto:.2f})")
    return torch.tensor([w], dtype=torch.float32).to(DEVICE)


def _metrics(P, L):
    P, L = np.array(P), np.array(L)
    tp = ((P==1)&(L==1)).sum(); fp = ((P==1)&(L==0)).sum()
    fn = ((P==0)&(L==1)).sum(); tn = ((P==0)&(L==0)).sum()
    pr = tp/(tp+fp+1e-8); rc = tp/(tp+fn+1e-8)
    return {"precision":pr, "recall":rc, "f1":2*pr*rc/(pr+rc+1e-8), "fpr":fp/(fp+tn+1e-8)}


def evaluate_loader(model, loader, criterion):
    model.eval(); total, P, L = 0.0, [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            total += criterion(logits, y).item()*len(y)
            P.extend((torch.sigmoid(logits)>0.5).long().cpu().numpy())
            L.extend(y.long().cpu().numpy())
    m = _metrics(P, L); m["loss"] = total/len(L); return m


def train_one(data, pos_weight_override=None, epochs=50, patience=8,
              lr=1e-3, batch_size=256, verbose=True):
    """Train a model on a prepared split. Returns (best_state, best_f1)."""
    train_loader = DataLoader(PairDataset(data["Xtr"], data["ytr"]),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(PairDataset(data["Xva"], data["yva"]), batch_size=batch_size)

    model = SafetyMLP(input_dim=data["input_dim"]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=resolve_pos_weight(data["ytr"], pos_weight_override))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1, best_epoch, best_state, since = 0.0, 0, None, 0
    for epoch in range(1, epochs+1):
        model.train()
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(); criterion(model(X), y).backward(); optimizer.step()
        m = evaluate_loader(model, val_loader, criterion)
        if verbose:
            print(f"Epoch {epoch:02d} | Loss {m['loss']:.4f} | P {m['precision']:.3f} | "
                  f"R {m['recall']:.3f} | F1 {m['f1']:.3f} | FPR {m['fpr']:.3f}")
        if m["f1"] > best_f1 + 1e-4:
            best_f1, best_epoch, since = m["f1"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                if verbose: print(f"Early stop at epoch {epoch}.")
                break
    if verbose: print(f"Best F1 {best_f1:.3f} at epoch {best_epoch}")
    return best_state, best_f1


def val_probs(data, state):
    """Return (y_val, probs) for the given trained state — used for AP / PR curves."""
    model = SafetyMLP(input_dim=data["input_dim"]).to(DEVICE)
    model.load_state_dict(state); model.eval()
    X = torch.tensor(data["Xva"], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        p = torch.sigmoid(model(X)).cpu().numpy()
    return data["yva"], p


def save_ckpt(path, state, data, hazard_mode=HAZARD_MODE):
    torch.save({"model": state, "mean": data["mean"], "std": data["std"],
                "input_dim": data["input_dim"], "hazard_mode": hazard_mode,
                "no_confidence": data.get("no_confidence", False)}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--detections", default="detections.json")
    ap.add_argument("--out", default="best_model.pt")
    ap.add_argument("--pos-weight", type=float, default=-1, help="override auto pos_weight")
    ap.add_argument("--no-confidence", action="store_true",
                    help="ablation: zero the confidence features")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    data = load_split(args.annotations, args.detections, no_confidence=args.no_confidence)
    print(f"\nTrain pairs: {len(data['ytr'])}  |  Val pairs: {len(data['yva'])}"
          f"{'  | CONFIDENCE ABLATED' if args.no_confidence else ''}")
    state, _ = train_one(data, args.pos_weight, args.epochs, args.patience,
                         args.lr, args.batch_size)
    save_ckpt(args.out, state, data)
    print(f"Model + scaler saved to {args.out}")


if __name__ == "__main__":
    main()
