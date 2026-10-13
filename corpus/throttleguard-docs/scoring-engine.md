# scoring-engine.md — DPF/SCR Expert Scoring System

Source: `dpf_expert_system.py` module docstring and constants, ThrottleGuard
codebase.

## Summary

Rule-based expert system for DPF failure prediction. Encodes 20 years of
diesel technician knowledge as weighted rules. Not a machine-learning model —
every flag traces back to a specific, named rule with a fixed point value, so
a technician can see exactly why a truck scored the way it did.

```python
from dpf_expert_system import calculate_expert_score

result = calculate_expert_score({
    'vehicle_id': '247',
    'dpf_outlet_temp_active_regen_f': 890,
    'dpf_outlet_temp_peak_f': 1050,
    'dpf_inlet_temp_f': 940,
    'regen_count_7d': 4,
    'back_pressure_inh2o': 3.2,
    'driver_reported_frequent_regen': True,
})
```

## Required input fields

- `vehicle_id`
- `dpf_outlet_temp_active_regen_f` — critical: below 1000°F = incomplete
  burn/clogging risk; above 1160°F = worth checking
- `dpf_outlet_temp_peak_f` — critical: above ~1190°F (family-dependent) =
  thermal shock risk
- `dpf_inlet_temp_f` — used in the sensor-fault delta check
- `regen_count_7d` — regen frequency over the trailing 7 days
- `back_pressure_inh2o` — exhaust backpressure, inches of H2O

Missing optional fields are treated as "not triggered," not as errors.

## Priority thresholds

Score is capped at 100, then mapped to a priority tier:

| Priority | Score floor |
|---|---|
| CRITICAL | 60 |
| HIGH | 35 |
| MEDIUM | 15 |
| LOW | 0 |

## Failure-mode priority order

When multiple failure modes are triggered on the same truck, the
highest-priority one wins for display purposes (though all triggered rules
are still listed):

1. `SENSOR_FAULT` — data integrity issue; readings can't be trusted
2. `THERMAL_SHOCK` — structural damage imminent
3. `NOX_BREAKTHROUGH` — EPA violation + derate risk (ranked above clogging
   specifically because of the regulatory consequence)
4. `CLOGGING` — DPF blocked
5. `SCR_CATALYST` — SCR catalyst degraded
6. `DEF_QUALITY` — DEF concentration out of spec
7. `ASH_LOAD` — maintenance overdue
8. `OPERATIONAL` — duty-cycle / minor issues
9. `NONE`

## Engine families

Thresholds vary by engine platform family: `DETROIT`, `VOLVO_MACK`,
`CUMMINS_PACCAR`. All family-specific numeric thresholds live in
`throttleguard_engine_thresholds.py`, not hardcoded in the rule functions —
this is a hard style rule for the codebase, enforced so a threshold never
drifts between the two independent scoring-engine implementations.

## Known implementation detail

There are two independent implementations of the same 18-rule logic:
`dpf_expert_system.py` (used by the main Dashboard tab) and
`scoring_engine.py` (used by the Fleet Scores tab and the public landing
page's lead-capture preview). They have drifted before — a real bug where
Rule 2 ignored engine family in `dpf_expert_system.py` shipped undetected
until parallel test coverage was added. Any rule change must be applied to
both files and checked against both test suites.
