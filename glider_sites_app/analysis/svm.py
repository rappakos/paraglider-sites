# analysis/svm.py
import logging

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

from glider_sites_app.analysis.data_preparation import prepare_training_data
from glider_sites_app.analysis.model_loader import save_svm_results


SEED = 431
k_folds = 5

# Threshold on log1p(flight_count) scale used to derive binary flyability from regression output.
# A prediction above this value is treated as "flyable".
# Intended to be tuned globally (same value across all sites) once models for all sites exist.
FLYABLE_THRESHOLD = 0.5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def train_svm_regressor(site_name: str, save: bool = False):
    """Train an SVR pipeline to predict log1p(daily flight count).

    Uses the same 7 features as the Random Forest model.  A StandardScaler is
    embedded inside the sklearn Pipeline so that the scaler is fitted only on
    training folds during cross-validation (no data leakage).

    Flyability F1 is computed by thresholding predicted log-counts at
    FLYABLE_THRESHOLD.  The threshold is a module-level constant intended to be
    tuned globally across all sites.
    """

    # Prepare data
    df = await prepare_training_data(site_name)

    if len(df) < 50:
        logger.error(f"Insufficient data: only {len(df)} days available")
        return None

    # Features and target (same feature set as Random Forest)
    features = [
        'avg_wind_speed',
        'avg_wind_alignment',
        'max_wind_gust',
        'min_wind_speed',
        'total_sunshine',
        'total_precipitation',
        'max_lapse_rate',
    ]
    X = df[features].values
    y = np.log1p(df['flight_count'].values)

    # Pipeline: StandardScaler is mandatory for SVM (sensitive to feature scale)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=10.0, epsilon=0.1, gamma='scale')),
    ])

    # --- K-fold cross-validation ---

    # RMSE
    rmse_scores = cross_val_score(
        pipe, X, y, cv=k_folds, scoring='neg_root_mean_squared_error', n_jobs=-1
    )
    logger.info(f"{k_folds}-fold CV RMSE scores: {-rmse_scores}")
    logger.info(f"Mean CV RMSE: {-rmse_scores.mean():.3f} (+/- {rmse_scores.std() * 2:.3f})")

    # F1 — manual KFold loop so we can apply FLYABLE_THRESHOLD to fold predictions
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    f1_scores = []
    for train_idx, val_idx in kf.split(X):
        fold_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel='rbf', C=10.0, epsilon=0.1, gamma='scale')),
        ])
        fold_pipe.fit(X[train_idx], y[train_idx])
        y_pred_fold = fold_pipe.predict(X[val_idx])
        f1 = f1_score(
            y[val_idx] > FLYABLE_THRESHOLD,
            y_pred_fold > FLYABLE_THRESHOLD,
            zero_division=0,
        )
        f1_scores.append(f1)

    f1_scores = np.array(f1_scores)
    logger.info(f"{k_folds}-fold CV F1 scores (threshold={FLYABLE_THRESHOLD}): {f1_scores}")
    logger.info(f"Mean CV F1: {f1_scores.mean():.3f} (+/- {f1_scores.std() * 2:.3f})")

    # --- Train final model on full dataset ---
    logger.info(f"Training final SVR pipeline on full dataset ({len(X)} days)")
    pipe.fit(X, y)

    # --- Permutation importance (SVR has no native feature_importances_) ---
    perm = permutation_importance(
        pipe, X, y,
        n_repeats=10,
        random_state=SEED,
        n_jobs=-1,
        scoring='neg_root_mean_squared_error',
    )
    importance = pd.DataFrame({
        'feature': features,
        'importance': perm.importances_mean,
        'importance_std': perm.importances_std,
    }).sort_values('importance', ascending=False)

    logger.info(f"Permutation importance:\n{importance[['feature', 'importance']].to_string(index=False)}")

    result = {
        'site_name': site_name,
        'model': pipe,
        'features': features,
        'feature_importance': importance,
        'flyable_threshold': FLYABLE_THRESHOLD,
    }

    if save:
        save_svm_results(site_name, result)

    return result


if __name__ == '__main__':
    import asyncio
    do_save = True
    asyncio.run(train_svm_regressor('Rammelsberg NW', save=do_save))
