import numpy as np
import pandas as pd

import shap
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# =====================================================================
# 1. Prediction helper 
# =====================================================================
def raw_prediction(model, X_input):
    """Raw model output, consistent with `model_output="raw"` SHAP values."""
    if isinstance(model, CatBoostClassifier):
        return model.predict(X_input, prediction_type="RawFormulaVal")
    return model.predict_proba(X_input)[:, 1]


# =====================================================================
# 2. SHAP values 
# =====================================================================
def compute_shap_values(model, X):
    """TreeSHAP values as a 2-D array (n_samples, n_features)."""
    explainer = shap.TreeExplainer(model, data=X, model_output="raw")
    values = explainer(X, check_additivity=False).values
    if isinstance(model, RandomForestClassifier):
        values = values[:, :, 1]           # positive-class column
    return values


# =====================================================================
# 3. Neighborhood 
# =====================================================================
def find_neighbors(X, cat_cols, idx_point, n_neighbors=50):
    """
    Indices of the `n_neighbors` nearest rows to `idx_point`.

    Categorical features are binarized to "same as the instance? yes/no"
    before standardizing, and distance is Chebyshev — exactly as in the paper.
    """
    X_bin = X.copy()
    ref = X.loc[idx_point, cat_cols]
    for col in cat_cols:
        X_bin[col] = (X_bin[col] == ref[col]).astype(int)

    X_scaled = StandardScaler().fit_transform(X_bin)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="chebyshev").fit(X_scaled)
    _, indices = nbrs.kneighbors(X_scaled[idx_point].reshape(1, -1))
    return indices[0]


# =====================================================================
# 4. Plot building blocks 
# =====================================================================
def _feat_grid(X, neighbor_idx, feat, n=100):
    """Evenly spaced values across the neighbors' range of `feat`."""
    vals = X.iloc[neighbor_idx][feat].values
    return np.linspace(vals.min(), vals.max(), n)


def plot_shap_scatter(ax, X, shap_values, neighbor_idx, idx_point, feat,
                      xlabel=None, ylabel=None, fontsize=30):
    """
    Base of every figure: SHAP values of the neighbors (blue) plus the
    highlighted instance (red). Returns (x, y) of the neighbor points so
    callers can reuse them for regression / envelope overlays.

    `xlabel` / `ylabel` override the default axis text (e.g. a spelled-out
    feature name, or a two-line label for the narrow card figures).
    """
    feat_idx = X.columns.get_loc(feat)
    x = X.iloc[neighbor_idx][feat].values
    y = shap_values[neighbor_idx, feat_idx]

    ax.scatter(x, y, color="steelblue", zorder=5, label="SHAP values")
    ax.scatter(X.iloc[idx_point][feat], shap_values[idx_point, feat_idx],
               marker="o", color="red", s=50, zorder=6)

    ax.set_xlabel(xlabel or feat, fontsize=fontsize)
    ax.set_ylabel(ylabel or f"SHAP value for {feat}", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)
    return x, y


def add_shap_regression(ax, x, y, feat, max_depth=3, random_state=42):
    """Smooth blue SHAP-regression line (Figures 5, 6, 8, 9)."""
    reg = RandomForestRegressor(max_depth=max_depth, random_state=random_state)
    reg.fit(pd.DataFrame({feat: x}), y)
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = reg.predict(pd.DataFrame({feat: x_smooth}))
    ax.plot(x_smooth, y_smooth, color="steelblue", lw=2, zorder=4,
            label="SHAP regression")


def add_ice_instance(ax, model, X, shap_values, neighbor_idx, idx_point, feat):
    """Single red ICE curve for the instance, shifted onto its SHAP point
    (Figures 4b, 4d). Grid spans the neighbors' range of `feat`."""
    feat_idx = X.columns.get_loc(feat)
    row = X.iloc[[idx_point]].copy()
    base = raw_prediction(model, row)[0]
    feat_vals = _feat_grid(X, neighbor_idx, feat)

    preds = np.empty(len(feat_vals))
    for k, v in enumerate(feat_vals):
        r = row.copy()
        r[feat] = v
        preds[k] = raw_prediction(model, r)[0]
    preds += shap_values[idx_point, feat_idx] - base

    ax.plot(feat_vals, preds, color="red", lw=1.5, zorder=7,
            label="Change in prediction")


def add_ice_envelope(ax, model, X, shap_values, neighbor_idx, feat):
    """Dashed max/min envelope of ICE curves over all neighbors
    (Figures 6, 8, 9)."""
    feat_idx = X.columns.get_loc(feat)
    neighbors = X.iloc[neighbor_idx]
    feat_vals = _feat_grid(X, neighbor_idx, feat)

    shifts = shap_values[neighbor_idx, feat_idx] - raw_prediction(model, neighbors)
    ice = np.zeros((len(neighbor_idx), len(feat_vals)))
    for k, v in enumerate(feat_vals):
        tmp = neighbors.copy()
        tmp[feat] = v
        ice[:, k] = raw_prediction(model, tmp)
    ice += shifts[:, None]

    ax.plot(feat_vals, ice.max(axis=0), color="steelblue", lw=1.5,
            linestyle="--", zorder=2, label="Max change in prediction")
    ax.plot(feat_vals, ice.min(axis=0), color="steelblue", lw=1.5,
            linestyle="--", zorder=2)
