# passive-regen.md — Passive Regeneration Health Scoring

Source: `throttleguard_passive_regen.py` module docstring, ThrottleGuard
codebase.

## Why this module exists

Passive regen is the silent killer in DPF failures. Unlike forced/active
regen, the ECM never broadcasts a commanded event — no PGN flag, no service
tool alert. The truck is silently failing to burn soot during normal highway
operation, and the first sign most fleets see is a breakdown-level DPF
restriction. OEM software misses this entirely.

Temp ranges are validated against field data from diesel tech community
sources covering DD13/DD15/DD16, D13/MP8, and ISX15/X15 platforms (sourced
April 2026) — field-observed ranges, not OEM spec sheet values.

## How it works

When `regen_active = 0` (no commanded regen in progress), exhaust
temperatures in the passive regen range during highway operation indicate
passive regen is occurring. That range is family-specific:

- **Detroit and Volvo/Mack**: 600–700°F
- **Cummins/PACCAR**: 575–900°F (wider band — Cummins sustains higher EGTs
  under heavy highway load)

Temperatures that stay below **550°F** (a universal failure floor across all
families) mean passive regen is NOT happening — soot builds silently. The
module scores how well a truck's normal operation supports passive regen and
flags when it doesn't.

## Component scorers

Each dimension is scored 0.0–1.0, then composed with weights into a single
passive regen health score:

- **Exhaust temp** (`_score_exhaust_temp`) — is the truck running hot enough
  passively to oxidize soot? Only meaningful when `regen_active = 0`; neutral
  (0.5) during active regen. Below the family failure floor scores 0.0.
- **Idle percentage** — high idle time means no highway temps, so no passive
  regen opportunity. Short-haul + high idle is one leg of the fastest DPF
  clogging combination (the other two legs: excessive idle and poor fuel
  quality — see `scoring-engine.md`'s DPF rule table, Rule 7).
- Additional components cover regen frequency and fuel-quality proxies
  (water-in-fuel, fuel filter change cadence).

## Integration with the main scoring tiers

- High score (0.7–1.0) = highway truck burning clean passively = lower risk
  pressure
- Low score (0.0–0.3) = city/idle truck dependent on forced regens = higher
  risk pressure
- The score feeds `apply_passive_regen_modifier()`, which adjusts MEDIUM/LOW
  risk tiers only
- **Safety invariant: CRITICAL and HIGH priority trucks are never downgraded
  by the passive regen score.** This is enforced in code, not just
  documentation — the modifier function only ever moves MEDIUM↔LOW.
