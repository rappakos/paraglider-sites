---
description: Review pending changes against this repo's conventions
allowed-tools: Bash(git diff:*), Bash(git status:*), Read, Grep, Glob
---

Review the current uncommitted changes (`git diff` + `git status`; if the diff is
empty, review the latest commit via `git show` instead).

Report ONLY:
- bugs and correctness issues (wrong SQL, broken async patterns, cwd-sensitive paths)
- violations of this repo's layering: SQL outside `repositories/`, DB access from
  `routes.py`, blocking `sqlite3` calls on the event loop
- security issues (unparameterized SQL values, secrets in code)

Do NOT report: style preferences, formatting, missing type hints, or anything a
linter would catch. For each finding: file, line, one-sentence issue, suggested fix,
severity (high/medium/low). If there are no findings, say so explicitly — do not
invent low-value findings to seem thorough.

$ARGUMENTS