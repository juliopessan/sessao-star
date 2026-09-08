# Story 001 — Adaptive interview learning

## Goal

Make STAR Session learn from practice without mixing Portuguese and English data, ask a targeted follow-up when an answer is incomplete, and show measurable progress across saved sessions.

## Acceptance criteria

- [x] Backend behavior is covered by automated tests for SQLite persistence, language-specific ML, fallback question generation, follow-up decisions, and insights payloads.
- [x] Portuguese and English answers train and load separate predictive model artifacts.
- [x] A real interview answer can trigger at most one targeted follow-up question; a complete answer advances immediately.
- [x] Follow-up answers are merged into the saved transcript and final report.
- [x] The report view shows overall STAR coverage bars and a recent-session evolution timeline.
- [x] Existing local databases migrate safely by adding the language columns when needed.
- [x] The test database and generated model artifacts are isolated from the user's local data.

## File list

- `backend/db.py` — language-aware schema, history, coverage aggregates, and migration.
- `backend/ml.py` — language-specific model artifacts and training/prediction APIs.
- `backend/main.py` — adaptive follow-up endpoint and language-aware insights.
- `frontend/index.html` — follow-up interview state and visual progress history.
- `backend/tests/` — automated backend tests.
- `docs/stories/001-adaptive-interview-learning.md` — this story.

## Verification

- `pytest -q backend/tests`
- Manual browser smoke test for a short answer, a follow-up answer, and the history chart.
