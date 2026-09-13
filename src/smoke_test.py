"""
Synthetic end-to-end check of stages 4a/4b. No corpora, no GPU, ~1 minute.
Generates score sequences where trajectory genuinely carries signal that the last turn
does not, so a correct implementation must show D > A. If this passes, the experiment
harness is wired correctly and any null result on real data is a finding, not a bug.

Run:  python -m src.smoke_test
"""
import pickle

import numpy as np

from src.config import ARTIFACTS
import src.config as cfg


def synth(n=600, seed=0):
    rng = np.random.default_rng(seed)
    scores, embs, meta = [], [], []
    for i in range(n):
        T = rng.integers(5, 16)
        escalating = rng.random() < 0.4
        base = rng.uniform(0.25, 0.5)
        slope = rng.uniform(0.02, 0.05) if escalating else rng.uniform(-0.04, 0.0)
        s = np.clip(base + slope * np.arange(T) + rng.normal(0, 0.10, T), 0.01, 0.99)
        # the last turn is deliberately uninformative: everyone signs off calmer
        s[-1] = np.clip(base + rng.normal(0, 0.08), 0.01, 0.99)
        # label follows the trajectory, with 15% label noise
        lab = int(escalating) if rng.random() > 0.15 else int(not escalating)
        scores.append(s)
        embs.append(rng.normal(0, 1, (T, 32)) + s[:, None])
        meta.append({"conv_id": f"syn_{i}", "problem_type": "A" if i % 5 else "job crisis",
                     "label": lab, "delta": 0, "initial_intensity": 4,
                     "final_intensity": 4, "n_seeker_turns": int(T)})
    return scores, embs, meta


if __name__ == "__main__":
    cfg.N_REPEATS = 2
    cfg.GRU_EPOCHS = 30
    cfg.N_BOOTSTRAP = 200
    import src.evaluate as ev
    import src.trajectory_models as tm
    tm.GRU_EPOCHS = 30
    ev.N_REPEATS = 2
    ev.N_BOOTSTRAP = 200

    s, e, m = synth()
    with open(ARTIFACTS / "scores_seed999.pkl", "wb") as f:
        pickle.dump({"scores": s, "embeddings": e, "meta": m}, f)
    ev.main(999)
    print("\nSmoke test finished. Expect D1/D2 and C to beat A_last_turn clearly.")
