"""
Stage 4a - the four trajectory representations under comparison.

All four predict the SAME target (non-improvement in seeker-reported distress) from the
SAME turn-score sequence, so the only thing that varies is how the sequence is collapsed.
This is the ablation gap 3.2 says is missing from the literature.

  A  last-turn score only          <- the pointwise baseline the field actually deploys
  B  mean score across turns       <- order-free aggregate
  C  engineered trajectory features<- explicit slope / recency / volatility
  D1 learned GRU over scores       <- learned sequence model
  D2 learned GRU over scores + PCA-reduced turn embeddings
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (DEVICE, EMB_PCA_DIM, GRU_DROPOUT, GRU_EPOCHS, GRU_HIDDEN,
                        GRU_LR)

_trapz = getattr(np, "trapezoid", None) or np.trapz

FEATURE_NAMES = [
    "last", "mean", "max", "min", "std", "first", "last_minus_first",
    "ols_slope", "slope_last3", "third_diff", "frac_gt50", "frac_gt60",
    "max_rise_run", "log_n_turns", "auc_trapz",
]


def traj_features(s):
    """Engineered trajectory descriptors for one score sequence."""
    s = np.asarray(s, dtype=float)
    T = len(s)
    t = np.linspace(0, 1, T)
    slope = np.polyfit(t, s, 1)[0] if T > 1 else 0.0
    last3 = np.polyfit(np.arange(min(3, T)), s[-min(3, T):], 1)[0] if T >= 3 else 0.0
    k = max(1, T // 3)
    third_diff = s[-k:].mean() - s[:k].mean()
    d = np.diff(s)
    run = best = 0
    for x in d:
        run = run + 1 if x > 0 else 0
        best = max(best, run)
    return np.array([
        s[-1], s.mean(), s.max(), s.min(), s.std(), s[0], s[-1] - s[0],
        slope, last3, third_diff, (s > 0.5).mean(), (s > 0.6).mean(),
        best, np.log(T), _trapz(s, t) if T > 1 else s[0],
    ], dtype=float)


# ------------------------------------------------------------------ A / B / C
class ScalarModel:
    """Logistic regression on 1 or k trajectory features. Fitting a model even for the
    single-feature baselines keeps every arm on a comparable, calibrated probability scale."""

    def __init__(self, kind):
        self.kind = kind
        self.clf = make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=2000, C=1.0))

    def _X(self, seqs):
        if self.kind == "last":
            return np.array([[np.asarray(s)[-1]] for s in seqs])
        if self.kind == "mean":
            return np.array([[float(np.mean(s))] for s in seqs])
        if self.kind == "features":
            return np.array([traj_features(s) for s in seqs])
        raise ValueError(self.kind)

    def fit(self, seqs, embs, y):
        self.clf.fit(self._X(seqs), y)
        return self

    def predict_proba(self, seqs, embs):
        return self.clf.predict_proba(self._X(seqs))[:, 1]


# ----------------------------------------------------------------------- D
class _GRU(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gru = nn.GRU(in_dim, GRU_HIDDEN, batch_first=True, bidirectional=False)
        self.drop = nn.Dropout(GRU_DROPOUT)
        self.head = nn.Linear(GRU_HIDDEN, 1)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.head(self.drop(h[-1])).squeeze(-1)


class SeqModel:
    """GRU over the turn-score sequence. Small on purpose: ~1.3k conversations is not
    enough data to justify a transformer over turns, and over-parameterising here would
    make a null result uninterpretable."""

    def __init__(self, use_emb=False, seed=0):
        self.use_emb = use_emb
        self.seed = seed
        self.pca = None
        self.net = None
        self.T_scale = 1.0

    def _tensors(self, seqs, embs, fit_pca=False):
        feats = []
        if self.use_emb:
            if fit_pca:
                self.pca = PCA(n_components=EMB_PCA_DIM,
                               random_state=self.seed).fit(np.concatenate(embs))
            for s, e in zip(seqs, embs):
                z = self.pca.transform(e)
                feats.append(np.concatenate([np.asarray(s)[:, None], z], axis=1))
        else:
            feats = [np.asarray(s, dtype=float)[:, None] for s in seqs]
        lengths = torch.tensor([len(f) for f in feats])
        maxlen = int(lengths.max())
        X = np.zeros((len(feats), maxlen, feats[0].shape[1]), dtype=np.float32)
        for i, f in enumerate(feats):
            X[i, :len(f)] = f
        return torch.tensor(X), lengths

    def fit(self, seqs, embs, y):
        """Holds out an inner 15% of the training fold purely to fit the temperature,
        so D is reported on the same calibrated footing as A/B/C (gap 3.3)."""
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(y))
        n_cal = max(20, int(0.15 * len(y)))
        cal_idx, fit_idx = idx[:n_cal], idx[n_cal:]
        seqs_f = [seqs[i] for i in fit_idx]
        embs_f = [embs[i] for i in fit_idx] if embs is not None else None
        self._fit_core(seqs_f, embs_f, y[fit_idx])
        self.T_scale = 1.0
        logit_cal = self._raw_logits([seqs[i] for i in cal_idx],
                                     [embs[i] for i in cal_idx] if embs is not None else None)
        self.T_scale = _fit_temperature_1d(logit_cal, y[cal_idx])
        return self

    def _fit_core(self, seqs, embs, y):
        X, L = self._tensors(seqs, embs, fit_pca=True)
        yt = torch.tensor(y, dtype=torch.float32)
        self.net = _GRU(X.shape[2]).to(DEVICE)
        opt = torch.optim.Adam(self.net.parameters(), lr=GRU_LR, weight_decay=1e-4)
        pos_w = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)], dtype=torch.float32).to(DEVICE)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        X, yt = X.to(DEVICE), yt.to(DEVICE)
        n = len(y)
        for ep in range(GRU_EPOCHS):
            self.net.train()
            perm = torch.randperm(n)
            for i in range(0, n, 32):
                idx = perm[i:i + 32]
                opt.zero_grad()
                out = self.net(X[idx], L[idx])
                loss = lossf(out, yt[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
        return self

    @torch.no_grad()
    def _raw_logits(self, seqs, embs):
        self.net.eval()
        X, L = self._tensors(seqs, embs, fit_pca=False)
        return self.net(X.to(DEVICE), L).cpu().numpy()

    def predict_proba(self, seqs, embs):
        logits = self._raw_logits(seqs, embs) / self.T_scale
        return 1 / (1 + np.exp(-logits))


def _fit_temperature_1d(logits, y, grid=None):
    """Scalar temperature minimising binary NLL. Grid search: one parameter, tiny data."""
    grid = grid if grid is not None else np.linspace(0.25, 5.0, 96)
    best_T, best_nll = 1.0, np.inf
    for T in grid:
        p = np.clip(1 / (1 + np.exp(-logits / T)), 1e-6, 1 - 1e-6)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T


def build_models(seed):
    return {
        "A_last_turn": ScalarModel("last"),
        "B_mean": ScalarModel("mean"),
        "C_features": ScalarModel("features"),
        "D1_gru_scores": SeqModel(use_emb=False, seed=seed),
        "D2_gru_scores_emb": SeqModel(use_emb=True, seed=seed),
    }
