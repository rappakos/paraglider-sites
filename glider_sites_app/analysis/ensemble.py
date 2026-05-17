# analysis/ensemble.py
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from glider_sites_app.analysis.data_preparation import prepare_training_data
from glider_sites_app.analysis.model_loader import save_ensemble_results
from glider_sites_app.analysis.svm import FLYABLE_THRESHOLD

SEED = 431
k_folds = 5

# Steepness of the sigmoid that converts SVR log-counts to [0, 1] probabilities.
# At scale=0.5, the transition from p≈0.12 to p≈0.88 spans roughly one log-unit.
SIGMOID_SCALE = 0.5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def svm_to_prob(log_count: np.ndarray, threshold: float = FLYABLE_THRESHOLD, scale: float = SIGMOID_SCALE) -> np.ndarray:
    """Convert SVR log-count predictions to [0, 1] flyability probabilities.

    Uses a sigmoid centred at `threshold` with steepness controlled by `scale`.

    Args:
        log_count: Array of predicted log1p(flight_count) values from SVR.
        threshold: Centre of the sigmoid (defaults to FLYABLE_THRESHOLD from svm.py).
        scale: Width of the transition region in log-count units.

    Returns:
        Array of probabilities in [0, 1].
    """
    return 1.0 / (1.0 + np.exp(-(log_count - threshold) / scale))


def _make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, max_depth=5, random_state=SEED, n_jobs=-1)


def _make_svm_pipe() -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=10.0, epsilon=0.1, gamma='scale')),
    ])


# ---------------------------------------------------------------------------
# OOF prediction collector (shared by scan and train)
# ---------------------------------------------------------------------------

async def _collect_oof_predictions(site_name: str):
    """Collect out-of-fold RF probabilities and SVR log-count predictions.

    Both models are trained on identical KFold splits so comparisons are fair.

    Returns:
        Tuple (y_true, rf_proba_oof, svm_log_oof) — aligned 1-D arrays.
        y_true is binary (flight_count > 0).
        Returns None if insufficient data.
    """
    df = await prepare_training_data(site_name)
    if len(df) < 50:
        logger.error(f"Insufficient data: only {len(df)} days available")
        return None

    features = [
        'avg_wind_speed', 'avg_wind_alignment', 'max_wind_gust', 'min_wind_speed',
        'total_sunshine', 'total_precipitation', 'max_lapse_rate',
    ]
    X = df[features].values
    y_binary = (df['flight_count'].values > 0).astype(int)
    y_log = np.log1p(df['flight_count'].values)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    rf_proba_oof = np.zeros(len(X))
    svm_log_oof = np.zeros(len(X))

    for train_idx, val_idx in kf.split(X):
        rf = _make_rf()
        rf.fit(X[train_idx], y_binary[train_idx])
        rf_proba_oof[val_idx] = rf.predict_proba(X[val_idx])[:, 1]

        svm = _make_svm_pipe()
        svm.fit(X[train_idx], y_log[train_idx])
        svm_log_oof[val_idx] = svm.predict(X[val_idx])

    return y_binary, rf_proba_oof, svm_log_oof


# ---------------------------------------------------------------------------
# Alpha sweep
# ---------------------------------------------------------------------------

async def scan_ensemble_alpha(
    site_name: str,
    alphas: np.ndarray | None = None,
) -> pd.DataFrame:
    """Sweep blend weights and report F1 for each alpha.

    ensemble_prob = alpha * rf_prob + (1 - alpha) * svm_prob

    Uses out-of-fold predictions so there is no data leakage.
    Also logs standalone RF (alpha=1) and SVM (alpha=0) F1 for reference.

    Args:
        site_name: Site to evaluate.
        alphas: Array of alpha values to test. Defaults to np.arange(0, 1.05, 0.05).

    Returns:
        DataFrame with columns [alpha, f1] sorted by f1 desc.
    """
    if alphas is None:
        alphas = np.arange(0.0, 1.05, 0.05)

    oof = await _collect_oof_predictions(site_name)
    if oof is None:
        return pd.DataFrame()

    y_true, rf_proba_oof, svm_log_oof = oof
    svm_prob_oof = svm_to_prob(svm_log_oof)

    rows = []
    for alpha in alphas:
        ensemble_prob = alpha * rf_proba_oof + (1.0 - alpha) * svm_prob_oof
        f1 = f1_score(y_true, ensemble_prob >= 0.5, zero_division=0)
        rows.append({'alpha': round(float(alpha), 3), 'f1': round(f1, 4)})

    result_df = pd.DataFrame(rows).sort_values('f1', ascending=False)

    # Log standalone reference scores
    rf_f1 = f1_score(y_true, rf_proba_oof >= 0.5, zero_division=0)
    svm_f1 = f1_score(y_true, svm_prob_oof >= 0.5, zero_division=0)
    logger.info(f"Standalone RF F1 (alpha=1.0): {rf_f1:.4f}")
    logger.info(f"Standalone SVM F1 (alpha=0.0): {svm_f1:.4f}")
    logger.info(f"Alpha sweep for {site_name}:\n{result_df.to_string(index=False)}")

    return result_df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

async def train_ensemble(site_name: str, save: bool = False) -> dict | None:
    """Train the full ensemble (RF classifier + SVR) on all available data.

    Finds the optimal blend weight alpha via OOF sweep, then fits both models
    on the full dataset.

    Args:
        site_name: Site to train for.
        save: If True, persist the ensemble artefact via save_ensemble_results().

    Returns:
        Dict with keys: site_name, rf_model, svm_model, alpha, sigmoid_scale,
        flyable_threshold, feature_importance_rf, feature_importance_svm.
        Returns None on insufficient data.
    """
    df = await prepare_training_data(site_name)
    if len(df) < 50:
        logger.error(f"Insufficient data: only {len(df)} days available")
        return None

    features = [
        'avg_wind_speed', 'avg_wind_alignment', 'max_wind_gust', 'min_wind_speed',
        'total_sunshine', 'total_precipitation', 'max_lapse_rate',
    ]
    X = df[features].values
    y_binary = (df['flight_count'].values > 0).astype(int)
    y_log = np.log1p(df['flight_count'].values)

    # --- Find best alpha via OOF sweep ---
    alpha_df = await scan_ensemble_alpha(site_name)
    best_alpha = float(alpha_df.iloc[0]['alpha'])
    best_f1 = float(alpha_df.iloc[0]['f1'])
    rf_f1 = float(alpha_df[alpha_df['alpha'] == 1.0]['f1'].values[0])
    svm_f1 = float(alpha_df[alpha_df['alpha'] == 0.0]['f1'].values[0])
    logger.info(
        f"Best alpha={best_alpha:.2f} → ensemble F1={best_f1:.4f} "
        f"(RF={rf_f1:.4f}, SVM={svm_f1:.4f})"
    )

    # --- Fit final models on full dataset ---
    logger.info(f"Fitting final RF + SVM on full dataset ({len(X)} days)")

    rf_model = _make_rf()
    rf_model.fit(X, y_binary)

    svm_model = _make_svm_pipe()
    svm_model.fit(X, y_log)

    # Feature importance: RF native, SVM permutation (scored on binary F1)
    rf_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_,
    }).sort_values('importance', ascending=False)

    perm = permutation_importance(
        svm_model, X, y_log,
        n_repeats=10,
        random_state=SEED,
        n_jobs=-1,
        scoring='neg_root_mean_squared_error',
    )
    svm_importance = pd.DataFrame({
        'feature': features,
        'importance': perm.importances_mean,
        'importance_std': perm.importances_std,
    }).sort_values('importance', ascending=False)

    logger.info(f"RF feature importance:\n{rf_importance.to_string(index=False)}")
    logger.info(f"SVM permutation importance:\n{svm_importance[['feature', 'importance']].to_string(index=False)}")

    result = {
        'site_name': site_name,
        'rf_model': rf_model,
        'svm_model': svm_model,
        'alpha': best_alpha,
        'sigmoid_scale': SIGMOID_SCALE,
        'flyable_threshold': FLYABLE_THRESHOLD,
        'feature_importance_rf': rf_importance,
        'feature_importance_svm': svm_importance,
    }

    if save:
        save_ensemble_results(site_name, result)

    return result


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_ensemble(ensemble_data: dict, X: np.ndarray) -> pd.DataFrame:
    """Generate ensemble predictions for a feature matrix.

    Args:
        ensemble_data: Dict returned by train_ensemble() or load_ensemble_model().
        X: Feature matrix with the 7 standard features (numpy array or DataFrame).

    Returns:
        DataFrame with columns:
            ensemble_prob  – blended probability [0, 1]
            is_flyable     – bool (ensemble_prob >= 0.5)
            rf_prob        – raw RF predict_proba output
            svm_prob       – sigmoid-converted SVR output
    """
    if hasattr(X, 'values'):
        X = X.values

    alpha = ensemble_data['alpha']
    threshold = ensemble_data['flyable_threshold']
    scale = ensemble_data['sigmoid_scale']

    rf_prob = ensemble_data['rf_model'].predict_proba(X)[:, 1]
    svm_log = ensemble_data['svm_model'].predict(X)
    svm_prob = svm_to_prob(svm_log, threshold=threshold, scale=scale)

    ensemble_prob = alpha * rf_prob + (1.0 - alpha) * svm_prob

    return pd.DataFrame({
        'ensemble_prob': ensemble_prob,
        'is_flyable': ensemble_prob >= 0.5,
        'rf_prob': rf_prob,
        'svm_prob': svm_prob,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import asyncio
    do_save = True
    asyncio.run(train_ensemble('Rammelsberg NW', save=do_save))
    asyncio.run(scan_ensemble_alpha('Rammelsberg NW'))
