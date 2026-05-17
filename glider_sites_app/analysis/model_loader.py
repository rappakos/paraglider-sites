from datetime import datetime, timezone
import joblib
import pathlib
import logging
from typing import Literal

logger = logging.getLogger(__name__)

ModelType = Literal['classifier', 'regressor', 'bayesian', 'prior_counts']

def get_model_path(site_name: str, type: ModelType) -> pathlib.Path:
    """Get the file path for the pre-trained model of the given site"""
    model_dir = pathlib.Path(__file__).parent / "models"
    
    # Determine the suffix based on model type
    if type in ['classifier', 'regressor']:
        suffix = f"{type}_rf_model.joblib"
    elif type == 'bayesian':
        suffix = "bayesian_model.joblib"
    elif type == 'prior_counts':
        suffix = "prior_counts.joblib"
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


def save_site_prior_counts(site_name: str, prior_counts: dict):
    """Save prior counts for a site
    
    Args:
        site_name: Name of the site
        prior_counts: Dictionary of prior counts
    """
    counts_data = {
        'version': datetime.now(timezone.utc).isoformat(),
        'prior_counts': prior_counts
    }
    return save_results(site_name, 'prior_counts', counts_data)

def load_site_prior_counts(site_name: str):
    """Load prior counts for a site
    
    Args:
        site_name: Name of the site
    Returns:

        Dictionary of prior counts, or None if not found
    """
    res = load_site_model(site_name, 'prior_counts')
    return res['prior_counts'] if res is not None else None

def load_bayesian_model(site_name: str):
    """Load a trained Bayesian Network model
    
    Args:
        site_name: Name of the site
        
    Returns:
        Dictionary with model and metadata, or None if not found
    """
    return load_site_model(site_name, 'bayesian')


# --- SVM ---

def get_svm_model_path(site_name: str) -> pathlib.Path:
    """Get the file path for the SVM model of the given site"""
    model_dir = pathlib.Path(__file__).parent / "models"
    return model_dir / f"{site_name.replace(' ', '_')}_svm_model.joblib"


def save_svm_results(site_name: str, results: dict):
    """Save SVM training results to a file

    Args:
        site_name: Name of the site
        results: Model data dictionary (must include 'model', 'features', 'feature_importance', 'flyable_threshold')
    """
    results_path = get_svm_model_path(site_name)
    results_path.parent.mkdir(exist_ok=True)
    joblib.dump(results, results_path)
    logger.info(f"SVM model saved to {results_path}")


def load_svm_model(site_name: str):
    """Load a pre-trained SVM model for the given site

    Args:
        site_name: Name of the site

    Returns:
        Dictionary with keys 'model', 'features', 'feature_importance', 'flyable_threshold',
        or None if no saved model exists.
    """
    model_path = get_svm_model_path(site_name)
    if not model_path.exists():
        return None
    return joblib.load(model_path)


# --- Ensemble ---

def get_ensemble_model_path(site_name: str) -> pathlib.Path:
    """Get the file path for the ensemble model of the given site"""
    model_dir = pathlib.Path(__file__).parent / "models"
    return model_dir / f"{site_name.replace(' ', '_')}_ensemble_model.joblib"


def save_ensemble_results(site_name: str, results: dict):
    """Save ensemble training results to a file

    Args:
        site_name: Name of the site
        results: Model data dictionary (must include 'rf_model', 'svm_model', 'alpha',
                 'sigmoid_scale', 'flyable_threshold', 'feature_importance_rf', 'feature_importance_svm')
    """
    results_path = get_ensemble_model_path(site_name)
    results_path.parent.mkdir(exist_ok=True)
    joblib.dump(results, results_path)
    logger.info(f"Ensemble model saved to {results_path}")


def load_ensemble_model(site_name: str):
    """Load a pre-trained ensemble model for the given site

    Args:
        site_name: Name of the site

    Returns:
        Dictionary with keys 'rf_model', 'svm_model', 'alpha', 'sigmoid_scale',
        'flyable_threshold', 'feature_importance_rf', 'feature_importance_svm',
        or None if no saved model exists.
    """
    model_path = get_ensemble_model_path(site_name)
    if not model_path.exists():
        return None
    return joblib.load(model_path)