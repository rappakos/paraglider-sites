# Plan: SVM Regressor + Ensemble

## Phase 1 — New `glider_sites_app/analysis/svm.py`

- [ ] Module constants: `SEED = 431`, `k_folds = 5`, `FLYABLE_THRESHOLD = 0.0`
  - Configurable float on log-scale; `y_pred > FLYABLE_THRESHOLD` → flyable
  - Intended to be tuned globally (same value across all sites) once models exist
- [ ] `async train_svm_regressor(site_name, save=False)` function
  - Call `prepare_training_data(site_name)` — same as RF
  - Guard: `len(df) < 50` → log error, return `None`
  - Same 7 features as RF: `avg_wind_speed`, `avg_wind_alignment`, `max_wind_gust`, `min_wind_speed`, `total_sunshine`, `total_precipitation`, `max_lapse_rate`
  - `y = np.log1p(df['flight_count'])`
  - Build `Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='rbf', C=10.0, epsilon=0.1, gamma='scale'))])`
    - Scaler inside Pipeline to prevent data leakage in CV
  - CV RMSE: `cross_val_score(..., scoring='neg_root_mean_squared_error')`
  - CV F1: manual `KFold` loop applying `FLYABLE_THRESHOLD` to fold predictions
    - `f1_score(y_true > FLYABLE_THRESHOLD, y_pred > FLYABLE_THRESHOLD)`
  - Fit full pipeline on all data
  - Permutation importance: `permutation_importance(pipe, X, y, n_repeats=10, random_state=SEED)` → DataFrame sorted by mean, desc
  - Return dict: `{site_name, model (pipeline), features, feature_importance, flyable_threshold}`
  - If `save=True`: call `save_svm_results()`
- [ ] `__main__` block mirroring RF's block (Dielmissen, `do_save=True`)

## Phase 2 — Update `glider_sites_app/analysis/model_loader.py`

- [ ] `get_svm_model_path(site_name)` → `models/{site_name_clean}_svm_model.joblib`
- [ ] `save_svm_results(site_name, results)` → `joblib.dump(results, path)`
- [ ] `load_svm_model(site_name)` → returns full dict or `None` if file missing
- Note: no changes to existing RF functions

## Phase 3 — Documentation

- [ ] `README.md`: add `### SVM regressor` CLI section under `## Analysis`
- [ ] `BACKGROUND.md`: add `### SVM Regressor (SVR)` model section covering:
  - RBF kernel, mandatory StandardScaler (inside Pipeline), permutation importance
  - Same 7 features as RF, log1p target, configurable flyability threshold

## Phase 4 — Verification

- [ ] `python -m glider_sites_app.analysis.svm` runs for Dielmissen, logs CV RMSE + F1
- [ ] Confirm `models/Dielmissen_svm_model.joblib` created
- [ ] `load_svm_model('Dielmissen')` returns dict with keys `model`, `features`, `feature_importance`, `flyable_threshold`
- [ ] `results['model'].predict(X[:5])` returns float log-scale predictions

---

## Phase 5 — Ensemble (later)

> To be designed after seeing grey-zone frequency per site.

- [ ] Profile RF confidence distribution per site — how often does `rf_prob ∈ [0.35, 0.65]`?
- [ ] Decide ensemble strategy (options below) and implement `ensemble.py`
- [ ] Update `site_service.py` to use ensemble output instead of raw RF
- [ ] Update BN `RF_Flyability_Confidence` node to accept ensemble confidence
- [ ] Update docs

**Candidate strategies:**
1. **Grey-zone override**: when `rf_prob ∈ [0.35, 0.65]`, use `svm_log_count > FLYABLE_THRESHOLD` to break the tie
2. **Stacked meta-learner**: logistic regression trained on `[rf_prob, svm_log_count]` per site
3. **Weighted average**: blend `rf_prob` and `sigmoid(svm_log_count - FLYABLE_THRESHOLD)` with tuned weights

---

## Further Considerations

- **Global threshold fitting**: once models for all sites exist, sweep `FLYABLE_THRESHOLD ∈ [-0.5, 1.5]` to maximise mean F1 across all sites
- **Hyperparameter tuning**: optional `GridSearchCV` over `C ∈ {1, 10, 100}` and `epsilon ∈ {0.05, 0.1, 0.2}` as a future `tune=True` flag
- **SVC variant**: SVR only for now; SVC (classifier) could be added later if direct probability estimates are useful alongside the RF classifier
