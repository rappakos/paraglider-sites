# analysis/random_forest.py
import logging
import pandas as pd
import numpy as np
from typing import Literal
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, f1_score

from glider_sites_app.analysis.data_preparation import prepare_training_data


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def train_flight_predictor(site_name: str, 
                                 type: Literal['classifier', 'regressor'], 
                                 test_size: float = 0.2
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
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    logger.info(f"Training set: {len(X_train)} days")
    logger.info(f"Test set: {len(X_test)} days")
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ) if type=='regressor' else RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1                
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    if type == 'classifier':
        # Classification metrics
        train_f1 = f1_score(y_train, y_pred_train)
        test_f1 = f1_score(y_test, y_pred_test)
        
        logger.info(f"Train F1: {train_f1:.3f}")
        logger.info(f"Test F1: {test_f1:.3f}")
        #logger.info(f"Classification Report:\n{classification_report(y_test, y_pred_test)}")
    else:
        # Regression metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        logger.info(f"Train RMSE: {train_rmse:.2f}, R²: {train_r2:.3f}")
        logger.info(f"Test RMSE: {test_rmse:.2f}, R²: {test_r2:.3f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"Feature importance:\n{importance}")
    
    result = {
        'model': model,
        'features': features,
        'feature_importance': importance,
        'training_data': df
    }
    
    # Add type-specific metrics
    if type == 'classifier':
        result.update({
            'train_f1': train_f1,
            'test_f1': test_f1
        })
    else:
        result.update({
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        })
    
    return result


if __name__ == '__main__':
    import asyncio
    
    #asyncio.run(train_flight_predictor('Königszinne', 'classifier'))
    asyncio.run(train_flight_predictor('Rammelsberg NW', 'classifier'))
    #asyncio.run(train_flight_predictor('Börry', 'classifier'))
    #asyncio.run(train_flight_predictor('Porta', 'classifier'))