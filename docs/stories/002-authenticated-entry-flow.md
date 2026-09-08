# Story 002 — Authenticated entry flow

## Goal

Give STAR Session a persuasive public front door, a real sign-in/account-creation flow, and a private route into the existing interview workspace.

## Acceptance criteria

- [x] `/` serves a bilingual landing page with a clear value proposition and calls to action.
- [x] `/login` supports sign-in and account creation in Portuguese and English.
- [x] Passwords are stored as scrypt-derived hashes, never as plain text.
- [x] Successful authentication creates an expiring, httpOnly session cookie.
- [x] `/app` redirects unauthenticated visitors to `/login` and serves the existing workspace to authenticated users.
- [x] Interview, report, insights, voice, resume extraction, and persistence endpoints require authentication.
- [x] User-owned history and predictive model artifacts are isolated by account and language.
- [x] Existing local SQLite data migrates safely and is claimed by the first account created.
- [x] Automated tests cover account lifecycle, route protection, and password storage.

## File list

- `backend/auth.py` — password hashing and cookie-backed session management.
- `backend/db.py` — users, auth sessions, user ownership, and scoped history queries.
- `backend/ml.py` — user- and language-specific predictive model artifacts.
- `backend/main.py` — auth API, protected endpoints, and public/private routes.
- `frontend/landing.html` — persuasive public landing page.
- `frontend/login.html` — bilingual login and registration experience.
- `frontend/index.html` — authenticated workspace with account controls.
- `backend/tests/test_auth.py` — auth lifecycle and protection tests.
- `backend/tests/test_api.py` — protected API coverage updated for authenticated clients.

## Verification

- `pytest -q -p no:cacheprovider backend/tests`
- JavaScript syntax validation for all three pages.
- Browser smoke test for landing page, unauthenticated `/app` redirect, login screen, and language switch.
