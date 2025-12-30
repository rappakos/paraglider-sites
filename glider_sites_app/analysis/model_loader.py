import joblib
import pathlib
import logging
from typing import Literal

logger = logging.getLogger(__name__)

ModelType = Literal['classifier', 'regressor', 'bayesian']

def get_model_path(site_name: str, type: ModelType) -> pathlib.Path:
    """Get the file path for the pre-trained model of the given site"""
    model_dir = pathlib.Path(__file__).parent / "models"
    
    # Determine the suffix based on model type
    if type in ['classifier', 'regressor']:
        suffix = f"{type}_rf_model.joblib"
    elif type == 'bayesian':
        suffix = "bayesian_model.joblib"
    else:
        raise ValueError(f"Invalid model type: {type}. Must be 'classifier', 'regressor', or 'bayesian'")
    
    model_path = model_dir / f"{site_name.replace(' ', '_')}_{suffix}"
    return model_path

def load_site_model(site_name: str, type: ModelType):
    """Load a pre-trained model for the given site
    
    Args:
        site_name: Name of the site
        type: Model type - 'classifier', 'regressor', or 'bayesian'
    """
    model_path = get_model_path(site_name, type)
    
    if not model_path.exists():
        return None
    
    model = joblib.load(model_path)
    return model

def save_results(site_name: str, type: ModelType, results: dict):
    """Save training results to a file
    
    Args:
        site_name: Name of the site
        type: Model type - 'classifier', 'regressor', or 'bayesian'
        results: Model data dictionary
    """
    results_path = get_model_path(site_name, type)
    results_path.parent.mkdir(exist_ok=True)
    joblib.dump(results, results_path)
    logger.info(f"Model saved to {results_path}")


def save_bayesian_model(model, site_name: str, features: list = None):
    """Save a trained Bayesian Network model
    
    Args:
        model: Trained Bayesian Network
        site_name: Name of the site
        features: List of feature names used (optional)
    """
    # Save the model and metadata
    model_data = {
        'model': model,
        'features': features or [
            'avg_wind_speed', 'max_wind_gust', 'avg_wind_alignment',
            'max_lapse_rate', 'max_boundary_layer_height', 'wind_speed_850hPa',
            'is_workingday', 'best_score'
        ],
        'type': 'bayesian_network'
    }
    
    return save_results(site_name, 'bayesian', model_data)


def load_bayesian_model(site_name: str):
    """Load a trained Bayesian Network model
    
    Args:
        site_name: Name of the site
        
    Returns:
        Dictionary with model and metadata, or None if not found
    """
    return load_site_model(site_name, 'bayesian')