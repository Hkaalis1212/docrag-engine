# telematics-ingestion.md — Ingestion API and Provider Adapters

Source: `api.py` and `tg_telematics_adapters.py` module docstrings,
ThrottleGuard codebase.

## api.py — Telematics Ingestion API

FastAPI layer that handles Motive's OAuth flow and real-time webhook events.
This is the first adapter in an ingestion architecture built to be
provider-agnostic.

**How it fits in:** Motive pushes fault/engine/vehicle events to this
service via webhook. The server normalizes them into ThrottleGuard's
canonical J1939 row shape — the same fields the scoring engine and Streamlit
dashboard already use. Token persistence lives in Supabase
(`tg_motive_tokens`) so Railway restarts don't force fleet operators to
re-authorize.

**Deployment:** Runs as a separate Railway service in the same project.

- Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`

Required env vars (shared with the main service where noted):

- `DATABASE_URL` — same Supabase connection string as the main app
- `MOTIVE_CLIENT_ID` — from the Motive developer portal
- `MOTIVE_CLIENT_SECRET` — from the Motive developer portal
- `MOTIVE_REDIRECT_URI` — `https://<this-service-domain>.up.railway.app/callback`

**Routes:**

| Route | Purpose |
|---|---|
| `GET /health` | Railway health check |
| `GET /authorize` | Starts the Motive OAuth flow (redirects to Motive's consent screen) |
| `GET /callback` | Motive OAuth redirect — exchanges code for token, stored in Supabase |
| `POST /webhook` | Real-time fault/engine/vehicle events from Motive (HMAC-verified) |
| `GET /token` | Internal — confirms a token is live without exposing credentials |

## tg_telematics_adapters.py — Shared Normalization Layer

ThrottleGuard's scoring engine (`scoring_engine.score_row` /
`dpf_expert_system`) takes one canonical row shape — the J1939-derived
fields documented in the main engineering overview's "CSV Input Columns"
section (see `claude-md-overview.md`).

Every telematics provider (Motive, Samsara, Geotab, ...) is ultimately
reporting the same SAE J1939 parameters (SPN/FMI) off the same trucks — so
the SPN → canonical-field mapping is shared domain knowledge, not
Motive-specific plumbing. Adding a new provider means mapping its wire
format onto the existing SPN tables and writing one
`normalize_<provider>_*()` function — not re-deriving which SPN means what.

### Adapter convention

A provider adapter is a function:

```
normalize_<provider>_<event-or-batch>(payload: dict) -> dict | None
```

that converts that provider's payload into a (possibly partial) canonical
row.

- Returns `None` when the payload carries nothing ThrottleGuard tracks.
- Event-driven providers (like Motive webhooks) can usually only report one
  changed signal at a time, so most canonical fields will be missing —
  callers merge partial rows into whatever's already known about that
  vehicle before scoring.
- Keys prefixed with `_` are metadata for logging/merging, not scoring
  engine inputs.

Adding Samsara or Geotab support means writing new adapter functions in this
module plus new routes in `api.py` — the scoring engine itself never
changes, since it only ever sees the canonical row shape regardless of which
provider produced it.
