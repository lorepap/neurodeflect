# Practical Deflection — Offline RL Training

This codebase trains local per-switch deflection policies from simulation logs using offline RL (IQL/CQL/AWR) and evaluates them via FQE.

Key ideas:
- Local MDP per switch and flow: state from local congestion + short history.
- Actions: FORWARD vs DEFLECT (binary; can extend to DEFLECT_to_k).
- Dense reward shaping with terminal FCT penalty.
- Strong Offline Policy Evaluation (FQE) and diagnostics.

## Quick Start

1) Build per-switch CSV datasets from OMNeT++ results (already available if you ran the builder):

- Example folders: `Omnet_Sims/dc_simulations/simulations/sims/tmp/data/data_1G_<policy>`

2) Train an algorithm (IQL shown):

```
# EITHER: pass explicit data dirs
python3 /home/ubuntu/practical_deflection/RL_Training/train.py \
  --data-dirs \
    /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/tmp/data/data_1G_dibs \
    /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/tmp/data/data_1G_ecmp \
    /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/tmp/data/data_1G_probabilistic \
  --algo iql --out-dir /home/ubuntu/practical_deflection/RL_Training/runs/iql_baseline --steps 200000 --batch-size 2048

# OR: parametric discovery via base directory
python3 /home/ubuntu/practical_deflection/RL_Training/train.py \
  --data-base /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/tmp/data \
  --algo iql --out-dir /home/ubuntu/practical_deflection/RL_Training/runs/iql_all --steps 200000 --batch-size 2048
```

Flags to note:
- `--data-dirs`: one or more dataset directories (per-switch CSVs inside).
- `--algo`: `iql`, `cql`, or `awr`.
- `--out-dir`: where to save checkpoints, logs, and normalization stats.
- `--dry-run`: parse data and print dataset summary without training.

3) (Optional) Evaluate a trained policy with FQE:

```
python3 /home/ubuntu/practical_deflection/RL_Training/run_fqe.py \
  --data-base /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/tmp/data \
  --ckpt /home/ubuntu/practical_deflection/RL_Training/runs/iql_all/checkpoint_200000.pt \
  --steps 200 --batch-size 4096 --out /home/ubuntu/practical_deflection/RL_Training/runs/iql_all/fqe_eval.json
```

## Data Expectations

Each dataset dir contains CSV files produced by `new_dataset_builder.py`, one per (run, switch module). Required columns per row (packet at switch):

```
run,module,switch_id,timestamp,action,RequesterID,FlowID,seq_num,
queue_util,queues_tot_util,flow_start_time,flow_end_time,
query_id,query_start_time,query_end_time,QCT,FCT,packet_latency,ooo
```

The loader groups rows by `(switch_id, FlowID)`, sorts by `(timestamp, seq_num)`, and builds transitions `(s, a, r, s', done)`.

## Feature Design (default)

- Instant: `queue_util`, `queues_tot_util`, `seq_num_norm`, `flow_age_norm`, `ooo_recent`, `deflect_ema`.
- History stack (k=4): stack `queue_util`, `queues_tot_util`, `deflect_ema`, `ooo_recent` for t-3..t.
- Normalization: per-switch z-score for congestion/latency; others in [0,1].

Reward per step:
```
r = - (w_q*queue_util + w_l*queues_tot_util + w_o*ooo) - w_d*[DEFLECT]
```
Terminal shaping for the last step of a (switch,Flow) episode:
```
r_T += - w_F * FCT_norm
```
where `FCT_norm` is normalized within flow-size bins.

Weights (tunable via CLI): `w_q=1.0, w_l=0.3, w_o=0.2, w_d=0.15, w_F=0.5`.

## Algorithms

- IQL: expectile value net (τ), Q TD(0) to `r + γ V(s')`, actor is advantage-weighted BC with temperature β.
- CQL: Q loss adds conservative term `α*(logsumexp Q - Q(s,a_behavior))`.
- AWR: critic fit + advantage-weighted BC baseline.

## Outputs

`--out-dir` contains:
- `checkpoints/*.pt`: model and normalizer snapshots.
- `logs.csv`: training metrics over steps.
- `normalization.json`: feature normalization parameters.
- `fqe_eval.json`: FQE returns and confidence intervals when produced via `run_fqe.py`.

## Notes

- Python 3.8, torch CPU is sufficient. No venv required.
- Extend to multi-action DEFLECT_to_k by swapping `n_actions=2` to `K+1` and wiring per-port features/masks.
