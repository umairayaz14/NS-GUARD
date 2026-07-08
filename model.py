"""
model.py
Small MLP for pairwise relational safety classification.

Input : feature vector built in data_pipeline.extract_pairs
        = 10 geometric + 2 confidence + N_PAIR_TYPES one-hot   (default 15)
Output: single logit (hazard). Sigmoid is applied by BCEWithLogitsLoss / at eval.
"""

import torch
import torch.nn as nn


class SafetyMLP(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            # no sigmoid here — BCEWithLogitsLoss handles it
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
