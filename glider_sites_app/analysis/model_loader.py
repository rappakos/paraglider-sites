import joblib
import pathlib

def get_model_path(site_name: str, type: str) -> pathlib.Path:
    """Get the file path for the pre-trained model of the given site"""
    model_dir = pathlib.Path(__file__).parent / "models"
    model_path = model_dir / f"{site_name.replace(' ', '_')}_{type}_rf_model.joblib"
    return model_path

def load_site_model(site_name: str, type: str):
    """Load a pre-trained Random Forest model for the given site"""
    model_path = get_model_path(site_name, type)
    
    if not model_path.exists():
        return None
    
    model = joblib.load(model_path)
    return model

def save_resuls(site_name: str, type: str, results: dict):
    """Save training results to a file"""
    results_path = get_model_path(site_name, type)
    joblib.dump(results, results_path)