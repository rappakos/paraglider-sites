# analysis/random_forest.py
import logging

import pandas as pd
import numpy as np
from typing import Literal
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate


from glider_sites_app.analysis.data_preparation import prepare_training_data
from glider_sites_app.analysis.model_loader import save_results


SEED = 431
k_folds = 5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def train_flight_predictor(site_name: str, 
                                 type: Literal['classifier', 'regressor'], 
                                 test_size: float = 0.2,
                                 save: bool = False
    ):
    """Train Random Forest to predict daily flight count"""
    
    # Prepare data
    df = await prepare_training_data(site_name)
    
    if len(df) < 50:
        logger.error(f"Insufficient data: only {len(df)} days available")
        return None
    
    # Features and target
    features = [
        'avg_wind_speed', 
        'avg_wind_alignment', 
        'max_wind_gust',
        'min_wind_speed',
        #'avg_cloud_cover',
        'total_sunshine',
        'total_precipitation',
        #'max_boundary_layer_height', 
        'max_lapse_rate'
        ]
    X = df[features]
    y = np.log1p(df['flight_count']) if type=='regressor' else df['flight_count'] > 0
   
    # Initialize model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=SEED,
        n_jobs=-1
    ) if type=='regressor' else RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=SEED,
        n_jobs=-1                
    )
    
    # K-fold cross validation
    if type == 'classifier':
        cv_scores = cross_val_score(model, X, y, cv=k_folds, scoring='f1', n_jobs=-1)
        logger.info(f"{k_folds}-fold CV F1 scores: {cv_scores}")
        logger.info(f"Mean CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    else:
        cv_scores = cross_val_score(model, X, y, cv=k_folds, scoring='neg_root_mean_squared_error', n_jobs=-1)
        logger.info(f"{k_folds}-fold CV RMSE scores: {-cv_scores}")
        logger.info(f"Mean CV RMSE: {-cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Train final model on full dataset
    logger.info(f"Training final model on full dataset ({len(X)} days)")
    model.fit(X, y)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"Feature importance:\n{importance}")
    
    result = {
        'site_name': site_name, 
        'model': model,
        'features': features,
        'feature_importance': importance
    }
    
    if save:
        save_results(site_name, type, result)


    return result


if __name__ == '__main__':
    import asyncio
    do_save=True
    asyncio.run(train_flight_predictor('Königszinne', 'classifier', save=do_save))
    asyncio.run(train_flight_predictor('Rammelsberg NW', 'classifier', save=do_save))
    asyncio.run(train_flight_predictor('Börry', 'classifier', save=do_save))
    asyncio.run(train_flight_predictor('Porta', 'classifier', save=do_save))
    asyncio.run(train_flight_predictor('Brunsberg', 'classifier', save=do_save))