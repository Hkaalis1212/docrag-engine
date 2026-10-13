# subscription-billing.md — Subscription Management

Source: `tg_subscription.py` module docstring, ThrottleGuard codebase.

## Model

Subscription is per-fleet (per Railway deployment). Each fleet has one
subscription row tied to their admin account — this is a single-tenant
architecture; `fleet_id` is hardcoded to `"admin"`, so one deployment serves
exactly one paying customer. A new customer means a new deployment, set up
manually.

## Pricing (per truck / month)

| Tier | Price | Truck count |
|---|---|---|
| Trial | $0 | 14 days free, full access, no card required |
| Starter | $39/truck/mo | 1–10 trucks |
| Growth | $29/truck/mo | 11–50 trucks |
| Fleet | $19/truck/mo | 51–250 trucks |
| Enterprise | custom quote | 250+ trucks |

There is no annual plan in the codebase. `PRICING_TIERS` in
`tg_subscription.py` is the single source of truth for these figures — check
there before quoting a number anywhere else (marketing copy, landing page,
sales materials).

## Database tables (Supabase PostgreSQL)

- `tg_subscriptions` — one row per fleet (keyed on `fleet_id` = admin
  username)
- `tg_payment_history` — one row per completed payment

## Environment variables

- `STRIPE_SECRET_KEY` — Stripe secret key (`sk_live_...` or `sk_test_...`)
- `STRIPE_PUBLISHABLE_KEY` — Stripe publishable key (`pk_live_...` or
  `pk_test_...`)

## Relationship to the landing page

The public lead-capture landing page (`tg_landing.py`) is CSV-upload →
score-preview → "Request Access" into HubSpot. It is explicitly **not**
self-serve checkout, because of the single-tenant architecture described
above — a self-serve signup flow would imply multiple concurrent paying
customers per deployment, which the data model doesn't support. An earlier
version of the landing page attempted self-serve checkout before this
constraint was identified; that functionality was removed, not just hidden.
