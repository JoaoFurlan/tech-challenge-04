"""The promotion gate: a paired bootstrap on the held-out gate slice, not a hand-picked margin.

A week of ~400 defaulters moves AUC-PR by +-0.04 and recalibration never moves it, so a fixed margin would be either
arbitrary or fooled by noise. The challenger is promoted only if
  1. its expected cost (C_FN = 5 x C_FP, at each model's own decision threshold) beats the champion's with the whole
     interval above zero (95% for one candidate; with two, each gets 97.5% so the family-wise error stays 5%), AND
  2. its AUC-PR is not significantly worse (the 95% interval of challenger - champion does not lie below zero), AND
  3. its Brier score beats the constant-prior predictor (the usual sanity gate).
"""
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss

from credit_drift.training.metrics import DEFAULT_RATIO, cost_at_threshold, prior_brier

N_BOOTSTRAP = 1000


def paired_gate(y: np.ndarray, p_champion: np.ndarray, p_challenger: np.ndarray, threshold_champion: float,
                threshold_challenger: float, n: int = N_BOOTSTRAP, seed: int = 42, alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    cost_gain, pr_diff = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        cost_gain.append(cost_at_threshold(y[i], p_champion[i], threshold_champion, c_fn=DEFAULT_RATIO)
                         - cost_at_threshold(y[i], p_challenger[i], threshold_challenger, c_fn=DEFAULT_RATIO))
        pr_diff.append(average_precision_score(y[i], p_challenger[i]) - average_precision_score(y[i], p_champion[i]))
    bounds = [alpha / 2, 1 - alpha / 2]  # alpha is split across the candidates being gated (Bonferroni)
    cost_ci = [float(q) for q in np.quantile(cost_gain, bounds)]
    pr_ci = [float(q) for q in np.quantile(pr_diff, bounds)]
    brier = float(brier_score_loss(y, p_challenger))
    checks = {"cost_improves": cost_ci[0] > 0, "auc_pr_not_worse": pr_ci[1] >= 0, "beats_prior_brier": brier < prior_brier(y)}
    return {
        "rows": len(y), "defaulters": int(y.sum()),
        "cost_champion": cost_at_threshold(y, p_champion, threshold_champion),
        "cost_challenger": cost_at_threshold(y, p_challenger, threshold_challenger),
        "cost_gain": float(np.mean(cost_gain)), "cost_gain_ci": cost_ci,
        "auc_pr_champion": float(average_precision_score(y, p_champion)),
        "auc_pr_challenger": float(average_precision_score(y, p_challenger)),
        "auc_pr_diff": float(np.mean(pr_diff)), "auc_pr_diff_ci": pr_ci,
        "brier_champion": float(brier_score_loss(y, p_champion)), "brier_challenger": brier,
        "checks": checks, "passed": all(checks.values()), "confidence": 1 - alpha,
    }


def choose_candidate(gates: dict[str, dict]) -> str:
    """Among the candidates that passed, the largest cost gain; if none passed, the largest cost gain anyway (it stays
    the shadow challenger). No preselection on the small calibration slice: every candidate faces the gate."""
    passed = {k: g for k, g in gates.items() if g["passed"]} or gates
    return max(passed, key=lambda k: (passed[k]["cost_gain"], k == "A"))
