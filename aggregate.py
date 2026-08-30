"""
Stage 6 - aggregate across the five seeds into the tables that go in the paper.

Run:  python -m src.aggregate
Outputs: results/table_main.csv, results/table_ood.csv, results/table_main.tex
"""
import json

import numpy as np
import pandas as pd

from src.config import RESULTS, SEEDS

ORDER = ["A_last_turn", "B_mean", "C_features", "D1_gru_scores", "D2_gru_scores_emb"]
PRETTY = {"A_last_turn": "A. Last-turn score (pointwise)",
          "B_mean": "B. Mean score",
          "C_features": "C. Engineered trajectory features",
          "D1_gru_scores": "D1. GRU over score sequence",
          "D2_gru_scores_emb": "D2. GRU over scores + embeddings"}


def main():
    runs = []
    for s in SEEDS:
        f = RESULTS / f"trajectory_seed{s}.json"
        if f.exists():
            runs.append(json.loads(f.read_text()))
    if not runs:
        raise SystemExit("No results found. Run src.evaluate for at least one seed first.")
    print(f"aggregating {len(runs)} seeds")

    rows = []
    for m in ORDER:
        if m not in runs[0]["cv"]:
            continue
        r = {"Model": PRETTY[m]}
        for metric in ["recall@10", "recall@20", "recall@30", "precision@20",
                       "auroc", "auprc", "ece", "brier"]:
            vals = [run["cv"][m][metric]["mean"] for run in runs]
            r[metric] = f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"
        rows.append(r)
    main_tbl = pd.DataFrame(rows)
    main_tbl.to_csv(RESULTS / "table_main.csv", index=False)
    print("\n" + main_tbl.to_string(index=False))

    # significance, averaged p across seeds via Fisher's method
    sig = []
    for key in runs[0].get("vs_baseline", {}):
        ps = [run["vs_baseline"][key]["p_bonferroni"] for run in runs]
        deltas = [run["vs_baseline"][key]["delta"] for run in runs]
        chi = -2 * np.sum(np.log(np.clip(ps, 1e-12, 1)))
        from scipy.stats import chi2
        sig.append({"comparison": key,
                    "mean_delta": f"{np.mean(deltas):+.3f}",
                    "fisher_p": f"{chi2.sf(chi, 2 * len(ps)):.4g}"})
    if sig:
        sig_tbl = pd.DataFrame(sig)
        sig_tbl.to_csv(RESULTS / "table_significance.csv", index=False)
        print("\n" + sig_tbl.to_string(index=False))

    ood_rows = []
    for m in ORDER:
        vals = [run["ood"][m] for run in runs if run.get("ood") and m in run["ood"]]
        if not vals:
            continue
        row = {"Model": PRETTY[m]}
        for metric in ["recall@20", "auroc", "ece"]:
            row[metric] = f"{np.mean([v[metric] for v in vals]):.3f} ± {np.std([v[metric] for v in vals]):.3f}"
        ood_rows.append(row)
    if ood_rows:
        ood = pd.DataFrame(ood_rows)
        ood.to_csv(RESULTS / "table_ood.csv", index=False)
        print("\nOut-of-distribution (held-out problem types):\n" + ood.to_string(index=False))

    (RESULTS / "table_main.tex").write_text(main_tbl.to_latex(index=False, escape=False))
    print("\nwrote results/table_main.csv, table_main.tex")


if __name__ == "__main__":
    main()
