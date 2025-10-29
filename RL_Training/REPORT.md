# Practical Deflection — Offline RL Report (Stub)

This document tracks training runs, configurations, and key findings.

- Dataset: 1G policies aggregated from `data_1G_<policy>` folders (DIBS, ECMP, Prob, Prob_TB, Random, Random_TB, SD, Threshold, Threshold_TB, UniformRandom, Vertigo).
- State: local congestion + short history (k=4), sequence and flow/query context, EMA features.
- Action: binary {FORWARD, DEFLECT}.
- Reward: dense congestion/latency/OOO penalties + small deflection cost; terminal FCT shaping per flow-size bin.
- Algorithms: IQL (primary), CQL (abl), AWR (baseline).
- OPE: FQE with bootstrap CIs.

## Run Log

- [ ] TODO: Fill in after first training runs.

## Key Plots to Produce

- Expectile critic loss, Actor loss, Q magnitude/variance.
- Policy diagnostics: mean P(DEFLECT), KL to behavior by load decile.
- FQE returns and CIs; per behavior-policy slices.
- Ablations: remove queue_util/history; vary w_d.

## Notes / Next Steps

- Extend action space to DEFLECT_to_k when per-egress info is available.
- Add per-switch action masking hooks.
- Improve incast detection using query concurrency across flows at a switch.

