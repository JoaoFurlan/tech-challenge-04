"""Metrics used to rank, select and report models (docs/decisions.md, #8)."""
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

COST_RATIOS = (3.0, 5.0, 10.0)  # C_FN / C_FP; 5 is the assumption used to pick the champion
DEFAULT_RATIO = 5.0


def expected_cost(y, p, c_fn: float = DEFAULT_RATIO, c_fp: float = 1.0) -> float:
    """min over thresholds of C_FN*pi*(1-TPR) + C_FP*(1-pi)*FPR (course formula, per application)."""
    y = np.asarray(y)
    fpr, tpr, _ = roc_curve(y, p)
    pi = y.mean()
    return float(np.min(c_fn * pi * (1 - tpr) + c_fp * (1 - pi) * fpr))


def cost_at_threshold(y, p, threshold: float, c_fn: float = DEFAULT_RATIO, c_fp: float = 1.0) -> float:
    y, flagged = np.asarray(y), np.asarray(p) >= threshold
    fn = np.mean((y == 1) & ~flagged)
    fp = np.mean((y == 0) & flagged)
    return float(c_fn * fn + c_fp * fp)


def prior_brier(y) -> float:
    """Brier score of the constant predictor pi: the bar a useful model must beat."""
    pi = float(np.mean(y))
    return pi * (1 - pi)


def summarize(y, p) -> dict:
    return {
        "auc_pr": float(average_precision_score(y, p)),
        "auc_roc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "mean_pred": float(np.mean(p)),
        **{f"cost_{int(r)}": expected_cost(y, p, c_fn=r) for r in COST_RATIOS},
    }


def bootstrap_ci(y, p, metric=average_precision_score, n: int = 500, seed: int = 42, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        scores.append(metric(y[idx], p[idx]))
    return float(np.quantile(scores, alpha / 2)), float(np.quantile(scores, 1 - alpha / 2))
