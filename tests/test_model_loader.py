"""Tests for `glider_sites_app.analysis.model_loader`.

Run from the repo root with this project's venv: `python -m pytest`.

No real `.joblib` artifact is ever read or written here: `joblib` is replaced by a
recording fake, and every path helper is redirected into `tmp_path`, so the trained
models under `glider_sites_app/analysis/models` stay untouched.
"""
import pathlib
from datetime import datetime, timedelta

import pytest

from glider_sites_app.analysis import model_loader

# Captured before the redirect fixture patches them, for the one test that asserts
# where artifacts really live. Computing a path touches no disk.
_REAL_HELPERS = {
    name: getattr(model_loader, name)
    for name in ("get_model_path", "get_svm_model_path", "get_ensemble_model_path")
}

DEFAULT_BAYESIAN_FEATURES = [
    'avg_wind_speed', 'max_wind_gust', 'avg_wind_alignment',
    'max_lapse_rate', 'max_boundary_layer_height', 'wind_speed_850hPa',
    'is_workingday', 'best_score'
]

# Late-bound so the redirect fixture applies; `get_model_path` needs a type argument.
ALL_PATH_HELPERS = [
    pytest.param(lambda site: model_loader.get_model_path(site, 'classifier'), id="rf"),
    pytest.param(lambda site: model_loader.get_svm_model_path(site), id="svm"),
    pytest.param(lambda site: model_loader.get_ensemble_model_path(site), id="ensemble"),
]

ALL_LOADERS = [
    pytest.param(lambda site: model_loader.load_site_model(site, 'classifier'), id="classifier"),
    pytest.param(lambda site: model_loader.load_site_model(site, 'regressor'), id="regressor"),
    pytest.param(lambda site: model_loader.load_bayesian_model(site), id="bayesian"),
    pytest.param(lambda site: model_loader.load_site_prior_counts(site), id="prior_counts"),
    pytest.param(lambda site: model_loader.load_svm_model(site), id="svm"),
    pytest.param(lambda site: model_loader.load_ensemble_model(site), id="ensemble"),
]


class FakeJoblib:
    """Stand-in for `joblib`: records what was dumped, serialises nothing."""

    def __init__(self):
        self.dumped = []
        self.loaded = []
        self.load_returns = None

    def dump(self, obj, path):
        self.dumped.append((obj, pathlib.Path(path)))
        pathlib.Path(path).touch()

    def load(self, path):
        self.loaded.append(pathlib.Path(path))
        return self.load_returns

    @property
    def last_payload(self):
        return self.dumped[-1][0]

    @property
    def last_path(self):
        return self.dumped[-1][1]


@pytest.fixture(autouse=True)
def models_dir(tmp_path, monkeypatch):
    """Point every path helper at `tmp_path`, keeping the real file-naming logic.

    Autouse so no test can write into the repo's models directory by accident.
    """
    target = tmp_path / "models"

    def redirect(original):
        def wrapper(*args, **kwargs):
            return target / original(*args, **kwargs).name
        return wrapper

    for name, original in _REAL_HELPERS.items():
        monkeypatch.setattr(model_loader, name, redirect(original))
    return target


@pytest.fixture(autouse=True)
def fake_joblib(monkeypatch):
    """Autouse so a real `.joblib` model can never be loaded or written."""
    fake = FakeJoblib()
    monkeypatch.setattr(model_loader, "joblib", fake)
    return fake


# --- paths ---

@pytest.mark.parametrize("model_type, expected_name", [
    ('classifier', "Wasserkuppe_classifier_rf_model.joblib"),
    ('regressor', "Wasserkuppe_regressor_rf_model.joblib"),
    ('bayesian', "Wasserkuppe_bayesian_model.joblib"),
    ('prior_counts', "Wasserkuppe_prior_counts.joblib"),
])
def test_each_model_type_maps_to_its_own_file_name(model_type, expected_name):
    assert model_loader.get_model_path("Wasserkuppe", model_type).name == expected_name


def test_svm_and_ensemble_artifacts_have_their_own_file_names():
    assert model_loader.get_svm_model_path("Wasserkuppe").name == "Wasserkuppe_svm_model.joblib"
    assert model_loader.get_ensemble_model_path("Wasserkuppe").name == "Wasserkuppe_ensemble_model.joblib"


def test_unknown_model_type_is_rejected():
    with pytest.raises(ValueError, match="Invalid model type"):
        model_loader.get_model_path("Wasserkuppe", 'transformer')


@pytest.mark.parametrize("path_for", ALL_PATH_HELPERS)
def test_spaces_in_a_site_name_become_underscores(path_for):
    assert path_for("Hohe Wand Nord").name.startswith("Hohe_Wand_Nord_")


@pytest.mark.parametrize("site_name", ["Börry", "Königszinne"])
@pytest.mark.parametrize("path_for", ALL_PATH_HELPERS)
def test_non_ascii_site_names_are_kept_verbatim(path_for, site_name):
    assert path_for(site_name).name.startswith(f"{site_name}_")


def test_artifacts_live_in_the_models_folder_of_the_analysis_package():
    expected = pathlib.Path(model_loader.__file__).parent / "models"

    assert _REAL_HELPERS["get_model_path"]("Wasserkuppe", 'bayesian').parent == expected
    assert _REAL_HELPERS["get_svm_model_path"]("Wasserkuppe").parent == expected
    assert _REAL_HELPERS["get_ensemble_model_path"]("Wasserkuppe").parent == expected


# --- missing artifacts are a normal case ---

@pytest.mark.parametrize("load", ALL_LOADERS)
def test_a_missing_artifact_loads_as_none(load, fake_joblib):
    assert load("Börry") is None
    assert fake_joblib.loaded == [], "an absent file must not be handed to joblib"


def test_a_saved_classifier_does_not_answer_a_regressor_load(fake_joblib):
    model_loader.save_results("Börry", 'classifier', {"model": "<rf>"})
    fake_joblib.load_returns = fake_joblib.last_payload

    assert model_loader.load_site_model("Börry", 'regressor') is None
    assert model_loader.load_site_model("Börry", 'classifier') == {"model": "<rf>"}


def test_a_saved_svm_does_not_answer_an_ensemble_load(fake_joblib):
    model_loader.save_svm_results("Börry", {"model": "<svc>"})

    assert model_loader.load_ensemble_model("Börry") is None


# --- saving ---

def test_saving_creates_the_models_directory_on_first_write(models_dir, fake_joblib):
    assert not models_dir.exists()

    model_loader.save_results("Hohe Wand", 'classifier', {"model": "<rf>"})

    assert models_dir.is_dir()
    assert fake_joblib.last_payload == {"model": "<rf>"}
    assert fake_joblib.last_path == models_dir / "Hohe_Wand_classifier_rf_model.joblib"


def test_a_saved_bayesian_model_carries_its_network_and_type(models_dir, fake_joblib):
    model_loader.save_bayesian_model("<network>", "Hohe Wand")

    assert fake_joblib.last_payload["model"] == "<network>"
    assert fake_joblib.last_payload["type"] == 'bayesian_network'
    assert fake_joblib.last_path == models_dir / "Hohe_Wand_bayesian_model.joblib"


def test_bayesian_features_fall_back_to_the_defaults_when_not_given(fake_joblib):
    model_loader.save_bayesian_model("<network>", "Börry")

    assert fake_joblib.last_payload["features"] == DEFAULT_BAYESIAN_FEATURES


def test_bayesian_features_are_stored_verbatim_when_given(fake_joblib):
    model_loader.save_bayesian_model("<network>", "Börry", features=["avg_wind_speed", "best_score"])

    assert fake_joblib.last_payload["features"] == ["avg_wind_speed", "best_score"]


def test_an_empty_bayesian_feature_list_falls_back_to_the_defaults(fake_joblib):
    model_loader.save_bayesian_model("<network>", "Börry", features=[])

    assert fake_joblib.last_payload["features"] == DEFAULT_BAYESIAN_FEATURES


def test_saved_prior_counts_are_stamped_with_a_utc_version(models_dir, fake_joblib):
    model_loader.save_site_prior_counts("Königszinne", {"flyable": 12, "not_flyable": 30})

    assert fake_joblib.last_payload["prior_counts"] == {"flyable": 12, "not_flyable": 30}
    assert datetime.fromisoformat(fake_joblib.last_payload["version"]).utcoffset() == timedelta(0)
    assert fake_joblib.last_path == models_dir / "Königszinne_prior_counts.joblib"


# --- loading back ---

def test_prior_counts_are_unwrapped_from_their_envelope_on_load(models_dir, fake_joblib):
    model_loader.save_site_prior_counts("Königszinne", {"flyable": 3, "not_flyable": 7})
    fake_joblib.load_returns = fake_joblib.last_payload

    assert model_loader.load_site_prior_counts("Königszinne") == {"flyable": 3, "not_flyable": 7}
    assert fake_joblib.loaded == [models_dir / "Königszinne_prior_counts.joblib"]


def test_a_bayesian_model_is_read_back_with_its_metadata(fake_joblib):
    model_loader.save_bayesian_model("<network>", "Börry", features=["avg_wind_speed"])
    fake_joblib.load_returns = fake_joblib.last_payload

    loaded = model_loader.load_bayesian_model("Börry")

    assert loaded["model"] == "<network>"
    assert loaded["features"] == ["avg_wind_speed"]
    assert loaded["type"] == 'bayesian_network'


def test_an_svm_payload_is_read_back_unchanged(models_dir, fake_joblib):
    results = {
        "model": "<svc>",
        "features": ["avg_wind_speed"],
        "feature_importance": {"avg_wind_speed": 1.0},
        "flyable_threshold": 0.5,
    }
    model_loader.save_svm_results("Hohe Wand", results)
    fake_joblib.load_returns = fake_joblib.last_payload

    assert fake_joblib.last_path == models_dir / "Hohe_Wand_svm_model.joblib"
    assert model_loader.load_svm_model("Hohe Wand") == results


def test_an_ensemble_payload_is_read_back_unchanged(models_dir, fake_joblib):
    results = {
        "rf_model": "<rf>",
        "svm_model": "<svc>",
        "alpha": 0.6,
        "sigmoid_scale": 1.0,
        "flyable_threshold": 0.5,
        "feature_importance_rf": {"avg_wind_speed": 1.0},
        "feature_importance_svm": {"avg_wind_speed": 0.8},
    }
    model_loader.save_ensemble_results("Börry", results)
    fake_joblib.load_returns = fake_joblib.last_payload

    assert fake_joblib.last_path == models_dir / "Börry_ensemble_model.joblib"
    assert model_loader.load_ensemble_model("Börry") == results
