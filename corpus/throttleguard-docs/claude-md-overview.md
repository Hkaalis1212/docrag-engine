# ThrottleGuard — Claude Code Context

## Product
ThrottleGuard is a DPF + SCR predictive maintenance SaaS for commercial diesel fleets, built by AHC Developers.
The founder has ~20 years diesel technician experience (Detroit, Volvo/Mack, Cummins/PACCAR).
Code must be practical, well-commented, and explainable to someone who knows trucks but not ML.

## Business Model
- **Standalone SaaS** — per-fleet subscription via Stripe, priced per truck (see `PRICING_TIERS` in tg_subscription.py — single source of truth, check there before quoting a figure): Starter $39/truck/mo (1–10 trucks), Growth $29/truck/mo (11–50), Fleet $19/truck/mo (51–250), Enterprise (250+) custom quote, no automated checkout. No annual plan exists in the code — don't invent one.
- **14-day free trial** — no credit card required, full access
- **Optional bundle** with Fleet Optimizer (TruckFleetOptimizer) — gated by env var
- ThrottleGuard MUST work fully standalone. Never hard-depend on Fleet Optimizer.

## Scoring Engine
**NOT XGBoost. NOT ML.** ThrottleGuard v2 uses a rule-based expert system.

- **18 rules** covering the full aftertreatment system (DPF + SCR)
- **3 engine families** with different thresholds: DETROIT, VOLVO_MACK, CUMMINS_PACCAR
- Score 0–100 → priority: CRITICAL (≥60) / HIGH (≥35) / MEDIUM (≥15) / LOW (<15)
- Every flag shows the exact rule that fired and a plain-English action
- NOx conversion rules gate on peak_regen_temp_f ≥ 800°F (sensor not valid below this)

### Rule summary
| # | System | Trigger | Pts |
|---|---|---|---|
| 1 | DPF | Outlet temp <1000°F during regen — incomplete burn (not always DPF — could be mechanical) | 60 |
| 2 | DPF | Peak temp above family limit — thermal shock | 50 |
| 3 | DPF | Sensor delta fault (inlet/outlet spread >100°F once either reaches 950°F) | 70 |
| 4 | DPF | Regen count >2 in 7 days OR driver reports frequent regen | 30 |
| 5 | DPF | Mileage >300k since cleaning AND oil consumption >0.5 qt/1000mi | 25 |
| 6 | DPF | Turbo boost <20 PSI OR EGR flow fault | 25 |
| 7 | DPF | Avg trip <15 mi AND idle >35% — short haul duty cycle | 15 |
| 8 | DPF | DEF contamination >50 ppm OR DEF doser fault | 15 |
| 9 | DPF | Water in fuel OR fuel filter changed <45 days | 10 |
| 10 | DPF | Backpressure >4.0 in.H2O | 10 |
| 11 | SCR | NOx conversion <50% — EPA derate risk | 40 |
| 12 | SCR | NOx conversion 50–70% — catalyst degrading | 20 |
| 13 | SCR | SCR inlet temp <400°F — below catalyst light-off | 15 |
| 14 | SCR | DEF concentration critically wrong (<20% or >40%) | 25 |
| 14 | SCR | DEF concentration moderately wrong (outside 31–34%) | 10 |
| 15 | SCR | NH3 slip detected | 10 |
| 16 | BOTH | Compound DPF+SCR failure (+20 Detroit 1-Box, +15 others) | 15–20 |
| 17 | BOTH | SCR inlet vs. DPF outlet spread >50°F — sensor fault | 15 |
| 18 | DPF | Outlet temp >1160°F during regen — worth checking | 15 |

## CSV Input Columns

### Required (6)
vehicle_id, dpf_outlet_temp_active_regen_f, dpf_outlet_temp_peak_f,
dpf_inlet_temp_f, regen_count_7d, back_pressure_inh2o

### Optional DPF (11)
engine_family, driver_reported_frequent_regen, mileage_since_last_dpf_cleaning,
oil_consumption_qt_per_1000mi, turbo_boost_psi, egr_flow_fault,
avg_trip_distance_mi, idle_time_pct, def_quality_ppm, def_doser_fault,
water_in_fuel_detected, fuel_filter_change_frequency_days

### Optional SCR (5)
nox_conversion_pct, scr_inlet_temp_f, def_concentration_pct,
nh3_slip_detected, regen_active

## Engine Thresholds (throttleguard_engine_thresholds.py)
- REGEN_OUTLET_CRITICAL_F = 1000 (outlet temp below this during regen = incomplete burn, not always DPF; revised 2026-10-06 from 960)
- REGEN_ACTIVE_OUTLET_WATCH_HIGH_F = 1160 (outlet temp above this during regen = worth checking; Rule 18, added 2026-10-06, universal across families)
- DIFF_PRESSURE_CRITICAL_PSI = 4.0 (backpressure limit, in.H2O)
- REGEN_HIGH_CRITICAL_F: DETROIT=1250, VOLVO_MACK=1250, CUMMINS_PACCAR=1200
- NOX_CONVERSION_CRITICAL_PCT = 50, NOX_CONVERSION_WARNING_PCT = 70
- SCR_INLET_TEMP_MIN_F = 400
- DEF_QUALITY_SPEC_PCT = 32.5, DEF_QUALITY_MIN_PCT = 31.0, DEF_QUALITY_MAX_PCT = 34.0
- DEF_QUALITY_CRITICAL_PCT = 20.0
- ONE_BOX_FAMILIES = {"DETROIT"} — DPF and SCR share single housing
- DPF_SENSOR_DELTA_TEMP_FLOOR_F = 950, DPF_SENSOR_DELTA_MAX_SPREAD_F = 100 (Rule 3, added 2026-07-02)
- SCR_DPF_OUTLET_MAX_SPREAD_F = 50 (Rule 17, added 2026-07-02)

## Domain Rules (20 years field experience)
- High backpressure can exist with low soot — ash buildup, not just soot clogging
- Uneven EGT distribution across 4 sensors = early ash channeling
- Regen requesting at lower temps over time = DPF degrading, not just dirty
- Short-haul + excessive idle + poor fuel = fastest DPF clogging combination
- NOx sensors invalid below 800°F peak regen temp — ECM suppresses DEF dosing below this
- Detroit 1-Box: DPF and SCR share single housing — thermal event damages both simultaneously
- SCR catalyst requires >400°F inlet temp to activate urea chemistry (light-off threshold)
- DEF spec: ISO 22241 — 32.5% urea ±1.5% (31–34% acceptable, <20% = water contamination)
- Normal active-regen outlet range is 1000–1160°F; a low reading below 1000°F isn't always the DPF/DOC itself — could be a mechanical issue upstream (turbo, injectors, etc.) — action text must say so, not assume DPF clogging outright

## Tech Stack
- **Dashboard**: Streamlit (app.py)
- **Scoring**: dpf_expert_system.py (Dashboard tab) and scoring_engine.py (Fleet Scores tab, and tg_landing.py's lead-capture preview) are two independent implementations of the same 18 rules — see tests/ below. They have drifted before (a real bug, Rule 2 ignoring engine family in dpf_expert_system.py, shipped undetected until test coverage was added in 2026-10) — when changing a rule, change both and check the parallel test files still agree.
- **Database**: Supabase PostgreSQL via psycopg2 (DATABASE_URL env var)
- **Auth**: tg_auth.py — PBKDF2-HMAC-SHA256 (260k iterations) + per-user salt, with a legacy plain-SHA-256 migration path for old hashes. Roles: Admin / Technician / Viewer
- **Subscriptions**: tg_subscription.py — Stripe PaymentIntent, psycopg2 backend
- **Sales/lead tracking**: HubSpot (optional, HUBSPOT_PRIVATE_APP_TOKEN) — see "Sales & Lead Tooling" below
- **Hosting**: Railway — Streamlit (railway.toml) + FastAPI ingestion API (api.py, separate Railway service, start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`) are both deployed. tg_landing.py (lead-capture landing page) is built and ready but is a third service that has to be deployed separately — check whether it's actually live before assuming so; see its module docstring for the start command and env vars.
- **GitHub**: Hkaalis1212

## Database Tables (all prefixed tg_ to avoid collision with Fleet Optimizer)
- tg_users — authentication (created by tg_auth.py)
- tg_predictions — prediction history + outcome tracking (created by outcome_db.py)
- tg_subscriptions — fleet subscription status (created by tg_subscription.py)
- tg_payment_history — Stripe payment records (created by tg_subscription.py)
- tg_motive_tokens — Motive OAuth token persistence (created by tg_motive_auth.py); survives Railway restarts

All tables auto-created on first launch. No migrations needed.

## Key Files
```
ThrottleGuard/
├── CLAUDE.md
├── app.py                          ← Streamlit dashboard + auth/subscription gates
├── api.py                          ← FastAPI ingestion layer — Motive OAuth + webhooks
│                                     (separate Railway service; start: uvicorn api:app)
├── tg_telematics_adapters.py       ← Provider-agnostic normalization: SPN → canonical row
│                                     Motive adapter lives here; add Samsara/Geotab adapters here too
├── tg_motive_auth.py               ← Supabase-backed Motive token storage (tg_motive_tokens table)
├── dpf_expert_system.py            ← Scoring engine for Dashboard tab
├── scoring_engine.py               ← Scoring engine for Fleet Scores tab + tg_landing.py
├── throttleguard_engine_thresholds.py ← All numeric thresholds (single source of truth)
├── tg_auth.py                      ← User auth + login page + user management panel
├── tg_db.py                        ← Shared psycopg2 connection (get_conn())
├── tg_subscription.py              ← Stripe subscription management + PRICING_TIERS (source of truth for pricing)
├── outcome_db.py                   ← Prediction logging + outcome tracking
├── scored_dashboard.py             ← Fleet Scores tab UI
├── tg_demo_data.py                 ← 30-truck demo fleet (5 CRITICAL/8 HIGH/9 MEDIUM/8 LOW)
├── throttleguard_passive_regen.py  ← Passive-regen health scoring (ECM never flags passive regen
│                                     failure, this detects it from temp/idle/fuel patterns instead).
│                                     Wired into both scoring paths via layer_onto_scored_results() —
│                                     advisory only: can move MEDIUM/LOW tiers, never CRITICAL/HIGH
├── tg_logo.py                      ← Logo renderer
├── tg_tutorial.py                  ← In-app tutorial steps
├── tg_landing.py                   ← Public lead-capture landing page (no login). CSV upload → score
│                                     preview → "Request Access" into HubSpot. NOT self-serve checkout —
│                                     see Subscription Gate's single-tenant note for why. Deploy as a
│                                     3rd Railway service (see its own docstring); check if it's live.
├── tg_hubspot_sync.py              ← push_trial_start() / push_landing_lead() — optional HubSpot
│                                     sync, no-ops silently if HUBSPOT_PRIVATE_APP_TOKEN unset
├── tg_trial_followup_cron.py       ← Daily GitHub Action (.github/workflows/tg-trial-followup.yml):
│                                     reads HubSpot contacts with trial_start_date, drafts the due
│                                     nurture-sequence email as a HubSpot Task (never auto-sends)
├── .claude/skills/tg-promo-*/      ← Sales/marketing Claude Code skills — see "Sales & Lead Tooling"
├── tests/                          ← pytest suite for both scoring engines (run: pytest tests/ -v)
├── .env                            ← Local secrets (never commit)
├── .env.example                    ← Documents all env vars
├── .gitignore
├── railway.toml                    ← Railway deploy config (Streamlit service only). healthcheckPath
│                                     hits Streamlit's /_stcore/script-health-check (an internal,
│                                     experimental Streamlit option enabled via
│                                     --server.scriptHealthCheckEnabled in startCommand) — it actually
│                                     runs app.py end-to-end, so it returns 503 if the DB is
│                                     unreachable. Plain "/" always returns 200 from Streamlit's static
│                                     shell before any Python runs, so it can't gate a deploy on a dead
│                                     DB — don't revert to it. If a future Streamlit upgrade drops this
│                                     option, healthcheckPath silently degrades back to that behavior.
├── requirements.txt
└── requirements-dev.txt            ← requirements.txt + pytest
```

**Built but not currently wired into the live app** — real code, not junk, but
neither imported by app.py/api.py nor covered by tests. Confirm current status
before relying on them: throttleguard_scr_data_generator.py (synthetic DEF/SCR
training data generator). Unlike the files in "Dead Code" below, this was
never found to be broken or superseded — just never integrated.
(throttleguard_passive_regen.py was in this category until 2026-10, when it
was wired into both scoring paths — see Key Files above and "Passive Regen
Integration" below.)

## Passive Regen Integration
`throttleguard_passive_regen.py`'s `layer_onto_scored_results(original_df,
results_df, priority_col)` is called in two places:
- `app.py`'s `run_expert_system()` (real CSV uploads, dpf_expert_system.py path)
- `tg_demo_data.py`'s `get_demo_scored()` (demo fleet, scoring_engine.py path)

It adds `passive_regen_score` / `passive_regen_failure` / `passive_failure_type`
/ `passive_recommendation` / `adjusted_priority` columns, then both call sites
overwrite the main `priority`/`priority_label` column with `adjusted_priority`
and preserve the original under `priority_raw`/`priority_label_raw` — so every
existing renderer (KPI counts, sorting, color-coding, dispatch blocklist) picks
up the adjustment for free, while the raw rule-engine tier stays visible for
audit and is what gets logged to `tg_predictions` for calibration tracking
(logging happens from the pre-adjustment `results` list, before this layer
runs — adjusted tiers are never what `outcome_db.py` records).

**Safety invariant**: CRITICAL and HIGH are never moved by this layer — it can
only downgrade MEDIUM→LOW (passive score ≥0.80) or escalate LOW→MEDIUM
(passive score ≤0.25). `engine_family` is an optional CSV column and often
blank; `layer_onto_scored_results()` defaults it to CUMMINS_PACCAR (most
conservative) rather than crashing, since `PASSIVE_REGEN_FLOOR_F` indexes
engine family directly and has no `DEFAULT`/`""` key.

The demo fleet's 30 trucks are active-regen snapshots (`regen_active: 1`),
so the module's exhaust-temp and EGT-delta components stay neutral for all of
them by design — passive regen health genuinely can't be read from a mid-regen
reading. The remaining idle/regen-frequency/fuel-quality components can't
swing any demo truck's score past the 0.80/0.25 thresholds on their own, so
the documented 5 CRITICAL/8 HIGH/9 MEDIUM/8 LOW split is unaffected.

## Telematics Ingestion Architecture
api.py is the first adapter in a provider-agnostic ingestion layer.
- tg_telematics_adapters.py owns the SPN → canonical-field mapping (shared J1939 domain knowledge)
- Each provider gets normalize_<provider>_<event>() functions in that module
- api.py routes webhook events to the right normalizer; output is always the canonical row shape
- Adding Samsara or Geotab webhooks = new adapter functions in tg_telematics_adapters.py + new routes in api.py

## Testing
`pytest tests/ -v` (needs `pip install -r requirements-dev.txt`, or just
`pip install pytest pandas` — the suite only exercises dpf_expert_system.py,
scoring_engine.py, and throttleguard_passive_regen.py, none of which need the
rest of requirements.txt). 126 tests: 105 structured in parallel across
tests/test_dpf_expert_system.py and tests/test_scoring_engine.py — one fire/no-fire
pair per rule per engine, plus gating, priority, and compound-bonus coverage —
and 21 in tests/test_passive_regen.py covering apply_passive_regen_modifier()'s
CRITICAL/HIGH-never-move invariant and layer_onto_scored_results()'s integration
(blank/missing engine_family, non-standard priority values, end-to-end escalation).
A GitHub Actions workflow (.github/workflows/tests.yml) runs this on every push/PR,
but may show as stuck "queued" rather than passing or failing if the repo's free-plan
Actions minutes are exhausted — that's a billing/quota state, not a code problem;
verify by running the suite locally before concluding anything is actually broken.

## Dead Code (do not restore)
- data_processing.py — v1 feature preprocessing (XGBoost pipeline, removed)
- train_model.py — v1 model training (XGBoost, removed)
- throttleguard_billing.py, tg_stripe_webhook.py — a second, never-wired-in Stripe billing path (separate `tg_billing` table, keyed by Stripe customer ID). Never imported by app.py or api.py — the live app bills through tg_subscription.py's PaymentIntent flow (tg_subscriptions table, keyed by fleet_id). Removed 2026-10 rather than finished, to stop two billing systems from silently drifting.
- throttleguard_samsara_poller.py — a Samsara telematics integration, never wired into api.py (only Motive is live, via tg_telematics_adapters.py). Removed 2026-10. Re-add as a proper tg_telematics_adapters.py-style adapter if/when Samsara support is actually built, not by resurrecting this file.

## Environment Variables
| Variable | Required | Service | Description |
|---|---|---|---|
| DATABASE_URL | Yes | Both | Supabase PostgreSQL connection string |
| TG_ADMIN_PASSWORD | Yes | Streamlit | Default admin password (set before first deploy) |
| STRIPE_SECRET_KEY | Yes | Streamlit | Stripe secret key (sk_test_... or sk_live_...) |
| STRIPE_PUBLISHABLE_KEY | Yes | Streamlit | Stripe publishable key |
| MOTIVE_CLIENT_ID | Yes | API | From Motive developer portal |
| MOTIVE_CLIENT_SECRET | Yes | API | From Motive developer portal |
| MOTIVE_REDIRECT_URI | Yes | API | https://\<api-service-domain\>.up.railway.app/callback |
| MOTIVE_WEBHOOK_SECRET | Yes | API | From Motive portal → Webhooks → your endpoint; enables HMAC signature verification. `/webhook` returns 503 if unset — no unsigned events are accepted |
| MOTIVE_SETUP_KEY | Yes | API | Random secret gating `/authorize?key=...`; without it, `/authorize` refuses to run so the Motive integration can't be hijacked by an unauthenticated caller |
| MOTIVE_SCOPES | No | API | Space-separated OAuth scopes (default: `vehicles.read hours_of_service.read`) |
| THROTTLEGUARD_API_URL | No | Streamlit | Enables Fleet Optimizer integration (optional) |
| THROTTLEGUARD_API_KEY | No | Streamlit | API key for Fleet Optimizer requests |
| HUBSPOT_PRIVATE_APP_TOKEN | No | Streamlit, tg_landing.py, GitHub Action | HubSpot private app token (Settings → Integrations → Private Apps; scopes: crm.objects.contacts.read/write). Enables trial-start and landing-page-lead sync to HubSpot. Set separately in three places if you want all of it live: the Streamlit Railway service, the tg_landing.py Railway service (once deployed), and the `tg-trial-followup.yml` GitHub Actions secret — same token value, three independent settings. Everything no-ops silently without it. |

## Roles & Permissions
| Role | upload | view | outcomes | manage_users | history |
|---|---|---|---|---|---|
| Admin | ✓ | ✓ | ✓ | ✓ | ✓ |
| Technician | ✓ | ✓ | ✓ | | ✓ |
| Viewer | | ✓ | | | ✓ |

## Subscription Gate
Sits between auth and dashboard in app.py.
- fleet_id = "admin" (one subscription per Railway deployment — single-tenant; see note below)
- No active subscription → trial start page or upgrade page
- Trial: 14 days free, full access, no card
- Paid: per-truck tiers (see Business Model above) — fleet size entered at checkout time

**Single-tenant architecture note:** because fleet_id is hardcoded to "admin,"
one Railway deployment serves exactly one paying customer — there's no concept
of "customer #2" in the database. A new customer means a new deployment, set
up manually. Don't build a self-serve signup/checkout flow that assumes
otherwise without accounting for this (see tg_landing.py — it looks like it
could do that and originally tried to, see Dead Code history/git log — it's
lead-capture only now specifically because of this constraint).

## Fleet Optimizer Integration (optional)
- File: TruckFleetOptimizer/throttleguard_integration.py
- Enabled only when THROTTLEGUARD_API_URL env var is set
- Currently calls /api/dpf-status — needs updating to read from tg_predictions in Supabase
- Both apps share the same Supabase project (tg_ prefix prevents table collision)

## Sales & Lead Tooling
Built 2026-10, lives alongside the product code:
- **.claude/skills/tg-promo-*/** — 10 Claude Code skills for sales/marketing work
  (positioning, offer, content angles, lead magnets, prospect research, cold
  outreach, objection handling, follow-up sequences, case studies, demo script).
  Invoke with `/tg-promo-<name>`. Each is grounded in this file's real facts
  (pricing, rule count, engine families) — if you edit pricing or rule counts
  here, check whether any of these skills quote the old numbers.
- **HubSpot** (optional, HUBSPOT_PRIVATE_APP_TOKEN) — stores leads/prospects as
  Contacts, with fleet/duty-cycle detail as Notes and drafted outreach as Tasks
  (never auto-sent — a human always reviews and sends).
- **tg_hubspot_sync.py** — push_trial_start() fires from app.py when a real
  trial starts; push_landing_lead() fires from tg_landing.py's "Request Access."
  Both fail silently without the token — see Environment Variables.
- **tg_trial_followup_cron.py** — daily GitHub Action, drafts the due step of a
  6-stage trial nurture cadence (day 0/2/7/11/14/19) as a HubSpot Task per
  contact. Matches tg-promo-followup's documented cadence — keep them in sync
  if the cadence changes.
- Gmail drafts have also been created ad hoc (via a connected Gmail MCP tool,
  for cold outreach) but nothing in the committed code sends email directly —
  every automated path here stops at "draft for human review," by design.

## Code Style
1. Comments explain WHY not just WHAT
2. No over-engineering — practical over elegant
3. psycopg2 %s placeholders (not ?) — this is PostgreSQL not SQLite
4. All thresholds in throttleguard_engine_thresholds.py — never hardcode in rules
5. Never hardcode secrets — use environment variables
6. tg_ prefix on all DB tables
