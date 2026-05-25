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
from glider_sites_app.analysis.svm import BASE_FEATURES, SVM_FEATURES, FLYABLE_THRESHOLD, compute_svm_features

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
    # Drop rows with NaN in any feature column (SVR does not accept missing values)
    cols_needed = ['avg_wind_speed', 'avg_wind_alignment', 'max_wind_gust', 'min_wind_speed',
                   'total_sunshine', 'total_precipitation', 'max_lapse_rate']
    df = df.dropna(subset=cols_needed).reset_index(drop=True)
    if len(df) < 50:
        logger.error(f"Insufficient data: only {len(df)} days available")
        return None

    X_rf = df[BASE_FEATURES].values
    X_svm = compute_svm_features(df)
    y_binary = (df['flight_count'].values > 0).astype(int)
    y_log = np.log1p(df['flight_count'].values)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    rf_proba_oof = np.zeros(len(X_rf))
    svm_log_oof = np.zeros(len(X_rf))

    for train_idx, val_idx in kf.split(X_rf):
        rf = _make_rf()
        rf.fit(X_rf[train_idx], y_binary[train_idx])
        rf_proba_oof[val_idx] = rf.predict_proba(X_rf[val_idx])[:, 1]

        svm = _make_svm_pipe()
        svm.fit(X_svm[train_idx], y_log[train_idx])
        svm_log_oof[val_idx] = svm.predict(X_svm[val_idx])

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
    # Drop rows with NaN in any feature column (SVR does not accept missing values)
    cols_needed = ['avg_wind_speed', 'avg_wind_alignment', 'max_wind_gust', 'min_wind_speed',
                   'total_sunshine', 'total_precipitation', 'max_lapse_rate']
    df = df.dropna(subset=cols_needed).reset_index(drop=True)
    if len(df) < 50:
        logger.error(f"Insufficient data: only {len(df)} days available")
        return None

    X_rf = df[BASE_FEATURES].values
    X_svm = compute_svm_features(df)
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
    logger.info(f"Fitting final RF + SVM on full dataset ({len(X_rf)} days)")

    rf_model = _make_rf()
    rf_model.fit(X_rf, y_binary)

    svm_model = _make_svm_pipe()
    svm_model.fit(X_svm, y_log)

    # Feature importance: RF native, SVM permutation
    rf_importance = pd.DataFrame({
        'feature': BASE_FEATURES,
        'importance': rf_model.feature_importances_,
    }).sort_values('importance', ascending=False)

    perm = permutation_importance(
        svm_model, X_svm, y_log,
        n_repeats=10,
        random_state=SEED,
        n_jobs=-1,
        scoring='neg_root_mean_squared_error',
    )
    svm_importance = pd.DataFrame({
        'feature': SVM_FEATURES,
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
        'rf_features': BASE_FEATURES,
        'svm_features': SVM_FEATURES,
        'feature_importance_rf': rf_importance,
        'feature_importance_svm': svm_importance,
    }

    if save:
        save_ensemble_results(site_name, result)

    return result


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_ensemble(ensemble_data: dict, X: np.ndarray, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Generate ensemble predictions for a feature matrix.

    Args:
        ensemble_data: Dict returned by train_ensemble() or load_ensemble_model().
        X: Feature matrix with the 7 base features (for RF).
        df: Optional DataFrame with weather columns. If provided, SVM interaction
            features are computed from it. If None, X must already contain all
            SVM features (10 columns) for models trained with interactions.

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

    # RF always uses base features (first 7 columns)
    X_rf = X[:, :len(BASE_FEATURES)] if X.shape[1] > len(BASE_FEATURES) else X
    rf_prob = ensemble_data['rf_model'].predict_proba(X_rf)[:, 1]

    # SVM uses extended features if model was trained with them
    has_interactions = ensemble_data.get('svm_features') == SVM_FEATURES
    if has_interactions and df is not None:
        X_svm = compute_svm_features(df)
    elif has_interactions and X.shape[1] == len(SVM_FEATURES):
        X_svm = X
    else:
        # Backward compat: old models trained on 7 features
        X_svm = X_rf

    svm_log = ensemble_data['svm_model'].predict(X_svm)
    svm_prob = svm_to_prob(svm_log, threshold=threshold, scale=scale)

    ensemble_prob = alpha * rf_prob + (1.0 - alpha) * svm_prob

    return pd.DataFrame({
        'ensemble_prob': ensemble_prob,
        'is_flyable': ensemble_prob >= 0.5,
        'rf_prob': rf_prob,
        'svm_prob': svm_prob,
    })


# ---------------------------------------------------------------------------
# Cross-site comparison
# ---------------------------------------------------------------------------

ALL_SITES = ['Börry', 'Königszinne', 'Rammelsberg NW', 'Porta', 'Dielmissen','Brunsberg']

async def compare_all_sites(alpha_fixed: float = 0.3):
    """Loop over all sites and print a tabular overview of ensemble performance.

    Columns: site, days, flying_days, RF_F1, SVM_F1, Ens_F1(fixed), best_α, best_F1
    """
    rows = []
    for site_name in ALL_SITES:
        oof = await _collect_oof_predictions(site_name)
        if oof is None:
            logger.warning(f"Skipping {site_name} — insufficient data")
            continue

        y_true, rf_proba_oof, svm_log_oof = oof
        svm_prob_oof = svm_to_prob(svm_log_oof)

        rf_f1 = f1_score(y_true, rf_proba_oof >= 0.5, zero_division=0)
        svm_f1 = f1_score(y_true, svm_prob_oof >= 0.5, zero_division=0)

        # Fixed alpha ensemble
        ens_prob_fixed = alpha_fixed * rf_proba_oof + (1.0 - alpha_fixed) * svm_prob_oof
        ens_f1_fixed = f1_score(y_true, ens_prob_fixed >= 0.5, zero_division=0)

        # Best alpha sweep
        best_alpha, best_f1 = alpha_fixed, ens_f1_fixed
        for a in np.arange(0.0, 1.05, 0.05):
            ens_prob = a * rf_proba_oof + (1.0 - a) * svm_prob_oof
            f1 = f1_score(y_true, ens_prob >= 0.5, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_alpha = a

        rows.append({
            'site': site_name,
            'days': len(y_true),
            'flying_days': int(y_true.sum()),
            'RF_F1': round(rf_f1, 4),
            'SVM_F1': round(svm_f1, 4),
            f'Ens_F1(α={alpha_fixed})': round(ens_f1_fixed, 4),
            'best_α': round(float(best_alpha), 2),
            'best_F1': round(best_f1, 4),
        })

    df = pd.DataFrame(rows)
    logger.info(f"\n{'='*80}\nCross-site ensemble comparison (fixed α={alpha_fixed}):\n{'='*80}\n{df.to_string(index=False)}\n{'='*80}")
    return df


async def diagnose_competitor_effect(site_name: str = 'Dielmissen'):
    """Show how competitor flights correlate with missing flights at this site.

    For days where this site had 0 flights, how many had competitor flights?
    Breaks down by wind alignment quality.
    """
    df = await prepare_training_data(site_name)

    no_flights = df[df['flight_count'] == 0]
    has_flights = df[df['flight_count'] > 0]

    total_zero = len(no_flights)
    comp_flew_zero = (no_flights['competitor_flights'] > 0).sum()

    total_pos = len(has_flights)
    comp_flew_pos = (has_flights['competitor_flights'] > 0).sum()

    logger.info(f"\n{'='*60}\nCompetitor flight diagnosis for {site_name}\n{'='*60}")
    logger.info(f"Total days: {len(df)} (flying: {total_pos}, zero: {total_zero})")
    logger.info(f"Zero-flight days where competitors flew: {comp_flew_zero}/{total_zero} ({100*comp_flew_zero/max(total_zero,1):.1f}%)")
    logger.info(f"Flying days where competitors also flew: {comp_flew_pos}/{total_pos} ({100*comp_flew_pos/max(total_pos,1):.1f}%)")

    # Break down by wind alignment bins
    bins = [-1, 0.0, 0.3, 0.6, 1.0]
    labels = ['Against(<0)', 'Cross(0-0.3)', 'OK(0.3-0.6)', 'Good(>0.6)']
    df['align_bin'] = pd.cut(df['avg_wind_alignment'], bins=bins, labels=labels)

    pivot = df.groupby('align_bin', observed=True).agg(
        days=('flight_count', 'count'),
        flying_days=('flight_count', lambda x: (x > 0).sum()),
        zero_with_comp=('flight_count', lambda x: 0),  # placeholder
    ).reset_index()

    # Recompute zero_with_comp properly
    rows = []
    for label in labels:
        subset = df[df['align_bin'] == label]
        zero_subset = subset[subset['flight_count'] == 0]
        rows.append({
            'alignment': label,
            'days': len(subset),
            'flying_days': int((subset['flight_count'] > 0).sum()),
            'zero_days': len(zero_subset),
            'zero_w_comp': int((zero_subset['competitor_flights'] > 0).sum()),
        })

    diag_df = pd.DataFrame(rows)
    diag_df['fly_rate'] = (diag_df['flying_days'] / diag_df['days'].clip(lower=1)).round(3)
    diag_df['zero_comp_rate'] = (diag_df['zero_w_comp'] / diag_df['zero_days'].clip(lower=1)).round(3)
    logger.info(f"\nBy wind alignment:\n{diag_df.to_string(index=False)}")
    logger.info(f"{'='*60}")

    return diag_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import asyncio
    #do_save = False
    #site = 'Dielmissen'
    #asyncio.run(train_ensemble(site, save=do_save))
    #asyncio.run(compare_all_sites(alpha_fixed=0.3))
    asyncio.run(diagnose_competitor_effect('Dielmissen'))
