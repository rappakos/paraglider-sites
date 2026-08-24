---
paths: ["glider_sites_app/repositories/**"]
---

# Data access conventions (repositories layer)

All SQL in this project lives here. Two patterns are already established — pick by
direction of data flow, do not invent a third.

## Reads that return a DataFrame

Sync body in a private `_name_sync()`, exposed as an async public function:

```python
def _load_x_sync(site_name: str) -> DataFrame:
    engine = create_engine(f'sqlite:///{DB_NAME}', **_ENGINE_KWARGS)
    try:
        with engine.connect() as db:
            return read_sql_query(text("SELECT ... WHERE site_name = :site_name"),
                                  db, params={'site_name': site_name})
    finally:
        engine.dispose()

async def load_x(site_name: str) -> DataFrame:
    return await asyncio.to_thread(_load_x_sync, site_name)
```

- `engine.dispose()` always in `finally`.
- Never call the sync engine path directly from an `async def` — that blocks the event
  loop. `asyncio.to_thread` is what keeps it off the loop.

## Writes

Native async with aiosqlite, explicit commit:

```python
async def save_x(df: DataFrame):
    sql = "INSERT OR IGNORE INTO table (a, b) VALUES (?, ?)"
    async with aiosqlite.connect(DB_NAME, timeout=AIOSQLITE_TIMEOUT) as db:
        await db.executemany(sql, df.values.tolist())
        await db.commit()
```

- Prefer `INSERT OR IGNORE` / `INSERT OR REPLACE` so re-running a loader is idempotent.
- Never forget `await db.commit()`.

## Parameters and interpolation

- **Values are always placeholders**: `:name` for SQLAlchemy `text()`, `?` for aiosqlite.
  Never f-string a value into SQL.
- Column or table **identifiers** may be interpolated only from module-level constants
  (e.g. `HOURLY_PARAMS`, `columns_to_save` built from them) — never from request input
  or DataFrame content.

## Boundaries

- Import `DB_NAME` and `AIOSQLITE_TIMEOUT` from `..db`; never hardcode the DB path or a
  timeout literal.
- Keep the connection scope inside the function — short scopes are what prevent SQLite
  lock errors when jobs run in parallel.
- Repositories return data (DataFrame, scalar) or raise (`ValueError` for a missing row).
  No HTTP concerns, no business rules, no template data — those belong in `services/`.
