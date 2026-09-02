"""
Helpers for the counterfactual half of the paper (`Code_Counterfactuals.ipynb`).

Grouped as:
    1. Data loading      - South German Credit (UCI id 522), downloaded on demand
    2. Decision maps     - 2-D prediction grids and the Figure-1 panels
    3. Stability regions - leaf constraints of a decision tree
    4. Verification      - Monte-Carlo checks of the regions
"""

import io
import ssl
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# 1. Data loading
# =====================================================================
SOUTH_GERMAN_CREDIT_URL = (
    "https://archive.ics.uci.edu/static/public/522/south+german+credit.zip"
)

# The .asc file ships with the original German variable names; these are the
# English equivalents from the accompanying code table, in file order.
SOUTH_GERMAN_CREDIT_COLUMNS = [
    "status_checking", "duration", "credit_history", "purpose",
    "credit_amount", "savings", "employment_duration", "installment_rate",
    "personal_status_sex", "other_debtors", "present_residence_since",
    "property", "age", "other_installment_plans",
    "housing", "existing_credits", "job",
    "num_dependents", "own_telephone", "foreign_worker",
    "credit_risk",
]

# Valid codes of the ordinal/categorical features, again from the code table.
# `duration`, `credit_amount` and `age` are genuinely numeric and are absent.
SOUTH_GERMAN_CREDIT_ORDINAL_RANGES = {
    "status_checking": (1, 4),
    "credit_history": (0, 4),
    "purpose": (0, 10),
    "savings": (1, 5),
    "employment_duration": (1, 5),
    "installment_rate": (1, 4),
    "personal_status_sex": (1, 4),
    "other_debtors": (1, 3),
    "present_residence_since": (1, 4),
    "property": (1, 4),
    "other_installment_plans": (1, 3),
    "housing": (1, 3),
    "existing_credits": (1, 4),
    "job": (1, 4),
    "num_dependents": (1, 2),
    "own_telephone": (1, 2),
    "foreign_worker": (1, 2),
}


def _urlopen(url):
    """`urlopen` that also works where the system CA bundle is missing."""
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:                      # fall back to the system bundle
        context = None
    return urllib.request.urlopen(url, context=context)


def load_south_german_credit(cache_dir="data"):
    """
    South German Credit data set as a DataFrame with English column names.

    The archive is fetched from the UCI repository the first time and cached
    in `cache_dir`, so the notebook runs without any manual download.
    """
    cache_dir = Path(cache_dir)
    asc_path = cache_dir / "SouthGermanCredit.asc"

    if not asc_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        with _urlopen(SOUTH_GERMAN_CREDIT_URL) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        archive.extract("SouthGermanCredit.asc", cache_dir)

    # header=0 drops the German header row, `names` supplies the English one
    return pd.read_csv(asc_path, sep=r"\s+", header=0,
                       names=SOUTH_GERMAN_CREDIT_COLUMNS)


# =====================================================================
# 2. Decision maps
# =====================================================================
def decision_grid(model, base_point, x_idx, y_idx, x_range, y_range, n_grid=300):
    """
    Predictions of `model` on a 2-D slice through `base_point`.

    Features `x_idx` and `y_idx` sweep `x_range` / `y_range`; every other
    feature is held at its value in `base_point` (the median test point in
    the paper). Returns `(xx, yy, Z)` ready for `contourf`.
    """
    xx, yy = np.meshgrid(np.linspace(*x_range, n_grid),
                         np.linspace(*y_range, n_grid))

    grid_points = np.tile(np.asarray(base_point, dtype=float), (xx.size, 1))
    grid_points[:, x_idx] = xx.ravel()
    grid_points[:, y_idx] = yy.ravel()

    return xx, yy, model.predict(grid_points).reshape(xx.shape)


INDIVIDUAL_STYLE = dict(color="#FFD700", marker="o",
                        edgecolors="black", linewidths=1.5, zorder=5)
COUNTERFACTUAL_STYLE = dict(color="#8C3BDD", marker="s",
                            edgecolors="black", linewidths=1.5, zorder=5)


def plot_decision_map(xx, yy, Z, individual, counterfactual, title,
                      xlabel="Hours Worked", ylabel="Education",
                      alpha=0.85, marker_size=300, arrow=False,
                      xlim=(15, 60), ylim=(14, 24), figsize=(8, 6)):
    """
    One panel of Figure 1: decision regions of the model plus the individual
    and its counterfactual.

    `individual` and `counterfactual` are `(x, y)` pairs in the plotted
    coordinates. `alpha=0` hides the decision regions (Figure 1a), `arrow=True`
    draws the action from the individual to the counterfactual.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.contourf(xx, yy, Z, alpha=alpha, cmap="RdBu_r")

    if arrow:
        ax.annotate("", xy=counterfactual, xytext=individual,
                    arrowprops=dict(arrowstyle="->", color="black", lw=2),
                    zorder=4)

    ax.scatter(*individual, s=marker_size, label="Individual",
               **INDIVIDUAL_STYLE)
    ax.scatter(*counterfactual, s=marker_size, label="Counterfactual",
               **COUNTERFACTUAL_STYLE)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


# =====================================================================
# 3. Stability regions
# =====================================================================
def get_leaf_constraints(tree, x):
    """
    Per-feature bounds implied by the leaf of `tree` that contains `x`.

    Walks the decision path and keeps, for every feature tested along it, the
    tightest threshold on the side `x` falls. Features not tested on the path
    are absent from the result, i.e. unconstrained by the leaf.
    """
    tree_ = tree.tree_
    feature = tree_.feature
    threshold = tree_.threshold

    node_indicator = tree.decision_path(x.reshape(1, -1))
    leaf_id = tree.apply(x.reshape(1, -1))[0]

    bounds = {}
    for node_id in node_indicator.indices:
        if node_id == leaf_id:
            continue

        f = feature[node_id]
        if f == -2:
            continue  # leaf node

        thr = threshold[node_id]
        if f not in bounds:
            bounds[f] = [-np.inf, np.inf]

        # tighten the side of the threshold that x goes to
        if x[f] <= thr:
            bounds[f][1] = min(bounds[f][1], thr)
        else:
            bounds[f][0] = max(bounds[f][0], thr)

    return bounds


# =====================================================================
# 4. Verification
# =====================================================================
def domain_bounds(domain):
    """(lo, hi) of a feature domain given either as an interval or a value list."""
    return domain if isinstance(domain, tuple) else (min(domain), max(domain))


def sample_from_region(x, region, feature_to_idx, domains=None):
    """
    A random point of `region`, i.e. `x` with the region's features resampled.

    A region maps a feature name either to a `(lo, hi)` interval (sampled
    uniformly, clipped to `domains`) or to a list of admissible values
    (sampled uniformly at random). Empty intervals leave the feature untouched.
    """
    x = x.copy()
    for f, values in region.items():
        idx = feature_to_idx[f]

        if isinstance(values, tuple):
            lo, hi = values
            if domains is not None and f in domains:
                domain_lo, domain_hi = domain_bounds(domains[f])
                lo, hi = max(lo, domain_lo), min(hi, domain_hi)
            if lo > hi:
                continue  # empty interval
            x[idx] = np.random.uniform(lo, hi)
        else:
            x[idx] = np.random.choice(values)

    return x


def verify_positive_region(tree, x_cf, region, feature_to_idx,
                           domains=None, n_samples=10000):
    """True if every sampled point of the region around `x_cf` stays positive."""
    for _ in range(n_samples):
        x = sample_from_region(x_cf, region, feature_to_idx, domains)
        if tree.predict(x.reshape(1, -1))[0] != 1:
            return False
    return True


def verify_negative_region(tree, x0, shift, region, feature_to_idx,
                           domains=None, n_samples=10000):
    """
    True if every sampled point of the region around `x0` both

    1. stays negative, and
    2. is flipped to positive by the counterfactual action `shift`.
    """
    for _ in range(n_samples):
        x = sample_from_region(x0, region, feature_to_idx, domains)

        pred = tree.predict(x.reshape(1, -1))[0]
        if pred != 0:
            print(f"Failed: original prediction = {pred}")
            return False

        pred_cf = tree.predict((x + shift).reshape(1, -1))[0]
        if pred_cf != 1:
            print(f"Failed: CF prediction = {pred_cf}")
            return False

    return True


def print_region(region, header):
    """Readable dump of a stability region."""
    print(header)
    for f, values in region.items():
        print(f"{f}: {values}")
