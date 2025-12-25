# analysis/random_forest.py
import logging
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

from glider_sites_app.repositories.weather_repository import load_weather_data
from glider_sites_app.repositories.flights_repository import load_flight_counts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def prepare_training_data(site_name: str) -> pd.DataFrame:
    """Merge flight and weather data"""
    logger.info(f"Loading data for {site_name}")
    
    # Load data
    weather_df = await load_weather_data(site_name)
    flights_df = await load_flight_counts(site_name)
    
    logger.info(f"Weather data: {len(weather_df)} days")
    logger.info(f"Flight data: {len(flights_df)} days")
    
    # Merge on date - LEFT JOIN to keep all weather days
    merged_df = weather_df.merge(
        flights_df[['date', 'flight_count']], 
        on='date', 
        how='left'
    )
    
    # Fill days with no flights as 0
    merged_df['flight_count'] = merged_df['flight_count'].fillna(0)
    
    # Filter out invalid data
    merged_df = merged_df[
        (merged_df['avg_wind_speed'].notna()) &
        (merged_df['total_sunshine'].notna())
    ]
    
    logger.info(f"Merged data: {len(merged_df)} days with complete data")
    logger.info(f"Flight days: {(merged_df['flight_count'] > 0).sum()}")
    
    return merged_df


async def train_flight_predictor(site_name: str, test_size: float = 0.2):
    """Train Random Forest to predict daily flight count"""
    
    # Prepare data
    df = await prepare_training_data(site_name)
    
    if len(df) < 50:
        logger.error(f"Insufficient data: only {len(df)} days available")
        return None
    
    # Features and target
    features = ['avg_wind_speed', 'avg_wind_direction', 'total_sunshine', 'total_precipitation']
    X = df[features]
    y = df['flight_count']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    logger.info(f"Training set: {len(X_train)} days")
    logger.info(f"Test set: {len(X_test)} days")
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
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
    
    return {
        'model': model,
        'features': features,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'feature_importance': importance,
        'training_data': df
    }


if __name__ == '__main__':
    import asyncio
    
    result = asyncio.run(train_flight_predictor('Rammelsberg NW'))
    
    if result:
        print("\n=== Model Performance ===")
        print(f"Train R²: {result['train_r2']:.3f}")
        print(f"Test R²: {result['test_r2']:.3f}")
        print(f"Test RMSE: {result['test_rmse']:.2f} flights")
        print("\n=== Feature Importance ===")
        print(result['feature_importance'])