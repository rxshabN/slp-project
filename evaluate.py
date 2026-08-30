"""
Stage 4b - the experiment.

Metrics are chosen to match the triage task (gap 3.3), not the classification convention:
  primary   recall@budget  - of the conversations that actually deteriorated, what
                             fraction lands in the top b% a counsellor can open
  primary   ECE            - are the confidences shown to that counsellor honest
  secondary AUROC, AUPRC, Brier

Splits are at the conversation level. The OOD arm holds out whole problem types.

Run:  python -m src.evaluate --seed 13
Outputs: results/trajectory_seed{S}.json, results/reliability_seed{S}.png,
         results/recall_budget_seed{S}.png
"""
import argparse
import json
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from src.config import (ALERT_BUDGETS, ARTIFACTS, ECE_BINS, N_BOOTSTRAP, N_FOLDS,
                        N_REPEATS, OOD_PROBLEM_TYPES, RESULTS)
from src.trajectory_models import build_models


# ------------------------------------------------------------------- metrics
def recall_at_budget(y, p, budget):
    """Of the true positives, how many are inside the top-`budget` fraction by score."""
    n = len(y)
    k = max(1, int(round(budget * n)))
    order = np.argsort(-p)
    flagged = np.zeros(n, dtype=bool)
    flagged[order[:k]] = True
    tp = int((flagged & (y == 1)).sum())
    return tp / max(int((y == 1).sum()), 1)


def precision_at_budget(y, p, budget):
    n = len(y)
    k = max(1, int(round(budget * n)))
    order = np.argsort(-p)[:k]
    return float(y[order].mean())


def ece(y, p, bins=ECE_BINS):
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(y)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum() == 0:
            continue
        e += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(e)


def all_metrics(y, p):
    out = {"auroc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
           "auprc": float(average_precision_score(y, p)),
           "brier": float(brier_score_loss(y, p)),
           "ece": ece(y, p),
           "base_rate": float(y.mean())}
    for b in ALERT_BUDGETS:
        out[f"recall@{int(b * 100)}"] = recall_at_budget(y, p, b)
        out[f"precision@{int(b * 100)}"] = precision_at_budget(y, p, b)
    return out


def bootstrap_ci(y, p, fn, n=N_BOOTSTRAP, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap_diff(y, p1, p2, fn, n=N_BOOTSTRAP, seed=0):
    """Two-sided p for fn(p2) - fn(p1) != 0, resampling conversations jointly."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        diffs.append(fn(y[idx], p2[idx]) - fn(y[idx], p1[idx]))
    diffs = np.array(diffs)
    obs = fn(y, p2) - fn(y, p1)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(obs), float(min(1.0, p)), (float(np.percentile(diffs, 2.5)),
                                            float(np.percentile(diffs, 97.5)))


# ------------------------------------------------------------------- plotting
def reliability_plot(curves, path):
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, (y, p) in curves.items():
        edges = np.linspace(0, 1, 11)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p > lo) & (p <= hi)
            if m.sum() < 10:
                continue
            xs.append(p[m].mean())
            ys.append(y[m].mean())
        plt.plot(xs, ys, "o-", label=name, ms=4)
    plt.xlabel("mean predicted probability")
    plt.ylabel("observed non-improvement rate")
    plt.title("Calibration, pooled out-of-fold")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def budget_plot(curves, path):
    grid = np.linspace(0.05, 0.6, 24)
    plt.figure(figsize=(5.5, 4))
    for name, (y, p) in curves.items():
        plt.plot(grid, [recall_at_budget(y, p, b) for b in grid], label=name)
    plt.plot(grid, grid, "k--", lw=1, label="random ordering")
    plt.xlabel("alert budget (fraction of conversations reviewed)")
    plt.ylabel("recall on non-improving conversations")
    plt.title("Recall under a fixed reviewer capacity")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# ------------------------------------------------------------------------ run
def load(seed):
    with open(ARTIFACTS / f"scores_seed{seed}.pkl", "rb") as f:
        d = pickle.load(f)
    meta = d["meta"]
    y = np.array([m["label"] for m in meta])
    ptype = np.array([m["problem_type"] for m in meta])
    return d["scores"], d["embeddings"], y, ptype, meta


def run_cv(scores, embs, y, seed):
    cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=seed)
    oof = {k: np.zeros((N_REPEATS, len(y))) for k in build_models(seed)}
    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        rep = fold // N_FOLDS
        models = build_models(seed * 100 + fold)
        for name, m in models.items():
            m.fit([scores[i] for i in tr], [embs[i] for i in tr], y[tr])
            oof[name][rep, te] = m.predict_proba([scores[i] for i in te],
                                                 [embs[i] for i in te])
        print(f"  fold {fold + 1}/{N_FOLDS * N_REPEATS} done", flush=True)
    return oof


def run_ood(scores, embs, y, ptype, seed):
    mask_ood = np.isin(ptype, OOD_PROBLEM_TYPES)
    if mask_ood.sum() < 20 or (~mask_ood).sum() < 50:
        print("  ! OOD split too small, skipping")
        return None
    tr = np.where(~mask_ood)[0]
    te = np.where(mask_ood)[0]
    out = {}
    for name, m in build_models(seed).items():
        m.fit([scores[i] for i in tr], [embs[i] for i in tr], y[tr])
        p = m.predict_proba([scores[i] for i in te], [embs[i] for i in te])
        out[name] = all_metrics(y[te], p) | {"n_test": int(len(te))}
    return out


def main(seed):
    scores, embs, y, ptype, meta = load(seed)
    print(f"{len(y)} conversations, {y.mean():.1%} non-improving, "
          f"median {np.median([len(s) for s in scores]):.0f} seeker turns")

    oof = run_cv(scores, embs, y, seed)

    report = {"seed": seed, "n": int(len(y)), "base_rate": float(y.mean()), "cv": {}}
    pooled = {}
    for name, mat in oof.items():
        per_rep = [all_metrics(y, mat[r]) for r in range(N_REPEATS)]
        agg = {k: {"mean": float(np.mean([d[k] for d in per_rep])),
                   "sd": float(np.std([d[k] for d in per_rep]))} for k in per_rep[0]}
        p_pool = mat.mean(0)
        pooled[name] = p_pool
        lo, hi = bootstrap_ci(y, p_pool, lambda a, b: recall_at_budget(a, b, 0.20), seed=seed)
        agg["recall@20_ci95"] = [lo, hi]
        report["cv"][name] = agg

    # significance: every trajectory arm against the pointwise baseline, Bonferroni
    base = pooled["A_last_turn"]
    comps = [k for k in pooled if k != "A_last_turn"]
    report["vs_baseline"] = {}
    for k in comps:
        for metric, fn in [("recall@20", lambda a, b: recall_at_budget(a, b, 0.20)),
                           ("auroc", lambda a, b: roc_auc_score(a, b))]:
            obs, p, ci = paired_bootstrap_diff(y, base, pooled[k], fn, seed=seed)
            report["vs_baseline"][f"{k}|{metric}"] = {
                "delta": obs, "p_raw": p,
                "p_bonferroni": min(1.0, p * len(comps) * 2), "ci95": ci}

    report["ood"] = run_ood(scores, embs, y, ptype, seed)

    curves = {k: (y, v) for k, v in pooled.items()}
    reliability_plot(curves, RESULTS / f"reliability_seed{seed}.png")
    budget_plot(curves, RESULTS / f"recall_budget_seed{seed}.png")
    (RESULTS / f"trajectory_seed{seed}.json").write_text(json.dumps(report, indent=2))

    print("\n=== recall@20% (mean over repeats) ===")
    for k, v in report["cv"].items():
        print(f"  {k:20s} {v['recall@20']['mean']:.3f} ± {v['recall@20']['sd']:.3f}   "
              f"ECE {v['ece']['mean']:.3f}   AUROC {v['auroc']['mean']:.3f}")
    print("\n=== vs pointwise baseline (Bonferroni-corrected) ===")
    for k, v in report["vs_baseline"].items():
        print(f"  {k:34s} Δ={v['delta']:+.3f}  p={v['p_bonferroni']:.4f}")
    print(f"\nwrote results/trajectory_seed{seed}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=13)
    main(ap.parse_args().seed)
