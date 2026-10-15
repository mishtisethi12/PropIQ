"""
PropIQ - Specialized Machine Learning Model Definitions
Defines custom estimators and bootstrap confidence interval evaluators:
- NaiveBayesBinnedRegressor: Expected-value continuous regression via binned target classification
- Bootstrap CI Evaluator for MAE and R2
"""

import numpy as np
from typing import Dict, Any, List
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mean_absolute_error, r2_score


class NaiveBayesBinnedRegressor(BaseEstimator, RegressorMixin):
    """
    Naive Bayes for continuous numerical target prediction via binned target classification.
    Discretizes target into quantile bins, trains GaussianNB, and predicts continuous
    expected values across posterior class probabilities:
        E[y | x] = sum_c P(C=c | x) * mean(y | C=c)
    """
    def __init__(self, n_bins: int = 15, strategy: str = "quantile"):
        self.n_bins = n_bins
        self.strategy = strategy
        self.discretizer = None
        self.classifier = None
        self.bin_means_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        y_arr = np.asarray(y, dtype=np.float64)
        self.discretizer = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode="ordinal",
            strategy=self.strategy,
            subsample=None
        )
        y_binned = self.discretizer.fit_transform(y_arr.reshape(-1, 1)).flatten().astype(int)

        # Compute empirical conditional mean of continuous target for each bin
        unique_bins = np.unique(y_binned)
        self.bin_means_ = np.zeros(self.n_bins, dtype=np.float64)
        for b in unique_bins:
            mask = (y_binned == b)
            self.bin_means_[b] = float(np.mean(y_arr[mask]))

        self.classifier = GaussianNB()
        self.classifier.fit(X, y_binned)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.classifier.predict_proba(X)
        active_means = self.bin_means_[self.classifier.classes_]
        return np.dot(probs, active_means)


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Computes empirical bootstrap confidence interval for MAE and R2.
    Resamples test set with replacement n_bootstraps times.
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    boot_maes = np.empty(n_bootstraps, dtype=np.float64)
    boot_r2s = np.empty(n_bootstraps, dtype=np.float64)

    alpha = 1.0 - confidence
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    for i in range(n_bootstraps):
        sample_indices = rng.randint(0, n_samples, n_samples)
        y_t_samp = y_true[sample_indices]
        y_p_samp = y_pred[sample_indices]

        boot_maes[i] = mean_absolute_error(y_t_samp, y_p_samp)
        if np.var(y_t_samp) > 1e-9:
            boot_r2s[i] = r2_score(y_t_samp, y_p_samp)
        else:
            boot_r2s[i] = 1.0

    mae_lower = float(np.percentile(boot_maes, lower_pct))
    mae_upper = float(np.percentile(boot_maes, upper_pct))
    r2_lower = float(np.percentile(boot_r2s, lower_pct))
    r2_upper = float(np.percentile(boot_r2s, upper_pct))

    return {
        "mae_ci_lower": round(mae_lower, 2),
        "mae_ci_upper": round(mae_upper, 2),
        "mae_ci_margin": round((mae_upper - mae_lower) / 2.0, 2),
        "r2_ci_lower": round(max(r2_lower, 0.0), 4),
        "r2_ci_upper": round(min(r2_upper, 1.0), 4),
        "bootstrap_samples": n_bootstraps,
        "confidence_level": confidence
    }
