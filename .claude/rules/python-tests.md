---
paths: ["**/test_*.py", "**/*_test.py", "**/tests/**/*.py"]
---

# Test conventions

pytest, run from the repo root with this project's venv: `python -m pytest`.
Async tests use `pytest-asyncio` (`@pytest.mark.asyncio`); add it to
`requirements.txt` when the first async test lands.

## Never touch the real database

`glider_sites.db` is ~40 MB of collected data and is not reproducible from the repo.
Tests build their own SQLite file in a fixture:

```python
@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    # apply glider_sites_app/init_db.sql to db_path, then:
    monkeypatch.setattr("glider_sites_app.db.DB_NAME", str(db_path))
    return str(db_path)
```

No test may write into the repo root.

## No network, no training

- Stub every loader that reaches out: `dhv_loader`, `xcontest_loader`,
  `openmeteo_loader`, `era5_reanalysis` (cdsapi). A test that needs the network is a
  broken test.
- Never load or train `.joblib` models in a test. Test the code *around*
  `analysis/model_loader.py`, and treat a **missing model artifact** as a first-class
  case that must be asserted, not an error to skip past.

## Cases worth covering

- Non-ASCII site names (`Börry`, `Königszinne`) — these are real data, not edge cases.
- Empty DataFrame results from a repository read.
- Missing site: `sites_repository.get_main_direction` raises `ValueError`.
- Idempotency: running a save twice does not duplicate rows
  (`INSERT OR IGNORE` / `INSERT OR REPLACE`).

Name tests after the behaviour asserted, not the function called.
