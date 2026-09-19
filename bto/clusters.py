"""Effective number of trials by clustering (Bailey and Lopez de Prado, 2014,
recommend clustering correlated trials before counting them).

Trials are clustered hierarchically on the distance ``sqrt((1 - rho) / 2)``;
the number of clusters is chosen by the silhouette score over ``k = 2..K``.
The cluster count is the strict end of the deflation (few independent bets),
the participation ratio in :mod:`bto.psr` the lenient end; both are reported.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def correlation_distance(trial_returns: np.ndarray) -> np.ndarray:
    x = np.asarray(trial_returns, dtype=float)
    x = x[~np.isnan(x).any(axis=1)]
    corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    d = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(d, 0.0)
    return d


def silhouette(dist: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient from a precomputed distance matrix.
    Singleton clusters get a coefficient of 0, the usual convention, so that
    splitting everything into its own cluster is never rewarded."""
    n = len(labels)
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(uniq) >= n:
        return -1.0
    s = np.zeros(n)
    for i in range(n):
        own = labels == labels[i]
        own[i] = False
        if not own.any():
            continue  # singleton: s_i = 0
        a = dist[i, own].mean()
        b = min(dist[i, labels == c].mean() for c in uniq if c != labels[i])
        s[i] = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
    return float(s.mean())


def cluster_trials(trial_returns: np.ndarray, max_clusters: int | None = None) -> tuple[int, np.ndarray]:
    """Return ``(n_clusters, labels)``. ``labels`` are 1-based cluster ids per
    trial. With fewer than three trials every trial is its own cluster."""
    x = np.asarray(trial_returns, dtype=float)
    n = x.shape[1]
    if n < 3:
        return n, np.arange(1, n + 1)
    dist = correlation_distance(x)
    z = linkage(squareform(dist, checks=False), method="average")
    kmax = min(max_clusters or n - 1, n - 1)
    best_k, best_s, best_labels = 1, -np.inf, np.ones(n, dtype=int)
    for k in range(2, kmax + 1):
        labels = fcluster(z, k, criterion="maxclust")
        if len(np.unique(labels)) != k:
            continue
        s = silhouette(dist, labels)
        if s > best_s + 1e-12:
            best_k, best_s, best_labels = k, s, labels
    return best_k, best_labels
