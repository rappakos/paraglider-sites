# analysis/random_forest.py
import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

from glider_sites_app.repositories.weather_repository import load_weather_data
from glider_sites_app.repositories.flights_repository import load_flight_counts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def aggregate_weather(raw_weather_df: pd.DataFrame, main_direction: int) -> pd.DataFrame:
    """
    Aggregate hourly weather data to daily metrics
    
    Args:
        raw_weather_df: DataFrame with hourly weather data including time, wind_speed_10m, 
                       wind_direction_10m, wind_gusts_10m, sunshine_duration, precipitation
        main_direction: Primary launch direction in degrees (0-360)
    
    Returns:
        DataFrame with daily aggregated weather metrics
    """
    if raw_weather_df.empty:
        return pd.DataFrame()
    
    # Calculate wind direction alignment with main direction
    # cos(main_direction - wind_direction) gives 1 when aligned, -1 when opposite
    wind_dir_diff = np.radians(main_direction - raw_weather_df['wind_direction_10m'])
    raw_weather_df['wind_alignment'] = np.cos(wind_dir_diff)
    
    # Aggregate by date
    daily_weather = raw_weather_df.groupby('date').agg({
        'wind_speed_10m': 'mean',           # AVG wind strength
        'wind_gusts_10m': 'max',            # MAX wind gust
        'wind_alignment': 'mean',           # AVG wind alignment with main direction
        'precipitation': 'sum',             # SUM precipitation
        'sunshine_duration': 'sum'          # SUM sunshine
    }).reset_index()
    
    # Rename columns for clarity
    daily_weather.columns = [
        'date',
        'avg_wind_speed',
        'max_wind_gust',
        'avg_wind_alignment',
        'total_precipitation',
        'total_sunshine'
    ]
    
    logger.info(f"Aggregated {len(raw_weather_df)} hourly records to {len(daily_weather)} daily records")
    
    return daily_weather

async def prepare_training_data(site_name: str, main_direction: int) -> pd.DataFrame:
    """Merge flight and weather data"""
    logger.info(f"Loading data for {site_name}")
    
    # Load data
    raw_weather_df = await load_weather_data(site_name)
    flights_df = await load_flight_counts(site_name)

    weather_df = aggregate_weather(raw_weather_df,main_direction)
    
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
    
    # Add day-of-week features
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df['day_of_week'] = merged_df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    merged_df['is_weekend'] = (merged_df['day_of_week'] >= 5).astype(int)  # 1 for Sat/Sun, 0 for weekdays

    # Keep only weekends OR weekdays with flights
    filter = (merged_df['is_weekend'] == 1) | (merged_df['flight_count'] > 0)
    merged_df = merged_df[filter]   

    
    # Filter out invalid data
    merged_df = merged_df[
        (merged_df['avg_wind_speed'].notna()) &
        (merged_df['total_sunshine'].notna())
    ]
    
    logger.info(f"Merged data: {len(merged_df)} days with complete data")
    logger.info(f"Flight days: {(merged_df['flight_count'] > 0).sum()}")
    logger.info(f"Weekend days: {merged_df['is_weekend'].sum()}, Weekdays: {(1-merged_df['is_weekend']).sum()}")
    
    return merged_df


async def train_flight_predictor(site_name: str, main_direction: int, test_size: float = 0.15):
    """Train Random Forest to predict daily flight count"""
    
    # Prepare data
    df = await prepare_training_data(site_name,main_direction)
    
    if len(df) < 50:
        logger.error(f"Insufficient data: only {len(df)} days available")
        return None
    
    # Features and target
    features = ['avg_wind_speed', 'avg_wind_alignment', 'max_wind_gust','total_sunshine', 'total_precipitation']
    X = df[features]
    y = np.log1p(df['flight_count'])  # ln(1 + flight_count) 
    
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
    
    asyncio.run(train_flight_predictor('Rammelsberg NW', 315))
