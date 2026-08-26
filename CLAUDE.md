# paraglider-sites — guidance for Claude Code

FastAPI + aiosqlite app that joins paragliding flight logs (DHV-XC, XContest) with
weather data (Open-Meteo, ERA5) and serves per-site flyability predictions from
Random Forest, RF+SVR ensemble and Bayesian-network models. Jinja2 templates for the
UI, OpenAPI docs at `/docs`.

Workspace-wide rules (venv isolation, styling, secrets, SQLite hygiene) live in the
workspace `AGENTS.md` — do not restate them here.

## Running things

- **Always run from the repo root.** `db.py` resolves `./glider_sites.db` and
  `./glider_sites_app/init_db.sql` relative to the current working directory.
- **Everything is a module:** `python -m glider_sites_app.<path>` — never
  `python glider_sites_app/analysis/svm.py`.
- Dev server: `uvicorn app:app --reload --host localhost --port 3979`
  (port comes from `config.DefaultConfig.PORT`).
- The venv in this project is `venv/` (`.\venv\Scripts\Activate.ps1` on Windows).
  The `.sh` job scripts intentionally source `.venv/bin/activate` because they run on
  a Linux host — do not "align" one with the other.

## Layering — keep it

`routes.py` → `services/` → `repositories/` → `db.py`

- `routes.py`: HTTP handlers and Jinja2 rendering. No SQL, no direct DB access.
- `services/`: domain logic and orchestration (`site_service`, `flight_service`,
  `weather_service`).
- `repositories/`: **all SQL lives here** — this is currently true, keep it that way.
- `schemas.py`: pydantic request/response models.

New query? Add it to a repository and call it from a service.

## Database

- The live database is **`glider_sites.db`** at the repo root.
  The README's `glider_sites_app.db` is stale — ignore it.
- Schema lives in `glider_sites_app/init_db.sql` and is applied idempotently by
  `setup_db()` on app startup (`executescript`). Schema changes belong in that file.
- Two access patterns coexist in `repositories/`: reads run SQLAlchemy in a worker
  thread (`asyncio.to_thread`), writes use `aiosqlite` directly with an explicit
  commit. The details live in `.claude/rules/sql-repositories.md`, which loads
  automatically when you edit that folder. Keep connection scopes short so parallel
  jobs don't hit SQLite lock errors.
- The DB is ~40 MB, gitignored, and not reproducible from the repo alone.
  Back it up before any schema-breaking change.

## Trained models

- Artifacts: `glider_sites_app/analysis/models/<Site>_<kind>.joblib`
  (`bayesian_model`, `classifier_rf_model`, `ensemble_model`, `svm_model`,
  `prior_counts`), one set per site, gitignored.
- Regenerate with `glider_sites_app/jobs/retrain_model.sh <SiteName>`
  (RF classifier → ensemble → Bayesian network). Never hand-craft or fabricate a
  `.joblib` file.
- **Site names contain non-ASCII characters** (`Börry`, `Königszinne`) — quote paths
  and preserve UTF-8 when globbing or shelling out.
- Not every site has every model kind. Treat a missing artifact as a normal case
  (see `analysis/model_loader.py`), not an error to paper over.

## Fresh-clone expectations

`.env`, `venv/`, `*.db`, `*.joblib` and the ERA5 `*.zip` files are all gitignored, so
a fresh clone has no data and no models. Bootstrap order: create venv → 
`pip install -r requirements.txt` → `python -m glider_sites_app.db` → load data
(`tools.flights.dhv_loader`, `tools.weather.openmeteo_loader`) → retrain models.

## Verification

Tests live in `tests/` and run with `python -m pytest` from the repo root using this
project's venv. Coverage is currently limited to `analysis.model_loader`.

Two constraints on the suite — keep them:

- It must never touch the live DB or a real `.joblib` artifact. The autouse fixtures
  in `tests/test_model_loader.py` redirect every path helper into `tmp_path` and swap
  in a recording fake for `joblib`; new tests belong inside that arrangement.
- Keep heavy analysis imports out of it. Importing `analysis.bayes_network` alone
  costs ~27 s (pgmpy), which would dwarf the current ~2 s run.

The suite covers little so far, so also verify a change by:

1. running the affected module (`python -m glider_sites_app....`),
2. starting uvicorn and exercising the affected route or page,
3. for analysis changes, re-running the relevant `analysis.<module>` and comparing
   output against the previous run.

Do not report a change as done on the basis of reading the code alone.
