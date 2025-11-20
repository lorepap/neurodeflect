# NeuroDeflect: In-Network RL for Packet Deflection

This guide orients new contributors (human or AI) to the repository layout, environment setup, OMNeT++ experiment pipeline, and the offline RL training workflow. Follow it end-to-end when bootstrapping on a fresh machine.

---

## 1. Repository Layout (key directories)

```
.
├── Omnet_Sims/                     # OMNeT++ simulations, configs, and helper scripts
│   └── dc_simulations/
│       └── simulations/
│           └── sims/               # Main simulation workspace
│               ├── omnetpp_*.ini   # Traffic and topology configs
│               ├── run_*.sh        # Experiment shells (run_1G_..., run_dataset_creation, etc.)
│               ├── results_*       # Raw OMNeT++ vector dumps (per policy)
│               └── tmp/data/       # Processed datasets (per policy) created by run_dataset_creation.sh
├── RL_Training/                    # Offline RL codebase (dataset loader, algorithms, evaluation)
│   ├── train.py                    # Main training CLI (IQL/CQL/AWR)
│   ├── run_fqe.py                  # Offline policy evaluation (FQE)
│   ├── plots/compare_policies.py   # Visualization of RL vs behavior per policy
│   └── ...                         # data/, algos/, models/, eval/, runs/...
├── Switch_Implementations/         # Switch behavior and deflection mechanisms (P4/eBPF, etc.)
├── RL_INTEGRATION_SUMMARY.md       # Notes linking RL policies to runtime systems
├── PROJECT_GUIDE.md                # (this document)
└── README.md                       # High-level summary of RL training code
```

Key log/output directories:
- `Omnet_Sims/.../results_1G_<policy>` → raw simulation logs (per-run CSVRs)
- `Omnet_Sims/.../tmp/data/data_1G_<policy>` → per-switch CSV datasets (input to RL)
- `RL_Training/runs/<name>` → RL training outputs (logs.csv, checkpoints, FQE results, plots)

---

## 2. Environment Setup

### 2.1 System prerequisites
- Ubuntu 20.04+ (other Linux distros should work with minor adjustments)
- Python ≥ 3.8 (repo uses 3.8.10)
- OMNeT++ 6.x installed and available in PATH (only needed for running new simulations)
- GNU make, g++, cmake (for OMNeT++ buildchain)

### 2.2 Python environment
We do **not** enforce virtualenvs, but using one is recommended:
```bash
python3 -m venv venv
. venv/bin/activate
```

Install Python packages (CPU baseline):
```bash
pip install --upgrade pip
pip install numpy pandas matplotlib torch==1.13.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

Additional packages used by notebooks/scripts (install as needed):
```bash
pip install seaborn scikit-learn tqdm
```

> **GPU note**: this repository currently runs on CPU. If you later move to a GPU build, install CUDA-enabled PyTorch and ensure `torch.cuda.is_available() == True` before using `--device cuda` in training.

### 2.3 OMNeT++ dependencies
If OMNeT++ is not yet installed:
1. Download OMNeT++ (https://omnetpp.org/download/), extract it (e.g., `/opt/omnetpp-6.x`).
2. Run the configure/build sequence inside OMNeT++ directory:
   ```bash
   ./configure
   make
   ```
3. Source the `setenv` script before using OMNeT++ tools:
   ```bash
   source /opt/omnetpp-6.x/setenv
   ```
4. Ensure `opp_run` and `opp_runall` are in PATH (this happens automatically after sourcing).

---

## 3. Running OMNeT++ Simulations

### 3.1 Overview
Simulations live under `Omnet_Sims/dc_simulations/simulations/sims`. We typically generate data for multiple deflection policies (`dibs`, `ecmp`, `probabilistic`, `threshold`, `vertigo`, etc.).

### 3.2 Common scripts
- `run_1G_experiments.sh` / specialized variants (`run_1G_thr_dataset.sh`, etc.): launch experiments for policy collections defined in `.ini` files.
- `run_all_dataset_creation.sh`: convenience wrapper to build per-policy datasets from existing results directories.
- `run_dataset_creation.sh`: transforms raw vector outputs into per-switch CSVs using `new_dataset_builder.py`.

### 3.3 Typical pipeline
1. Ensure OMNeT++ environment is active (`source /opt/omnetpp-6.x/setenv`).
2. Run experiments (example for 1G policies):
   ```bash
   cd Omnet_Sims/dc_simulations/simulations/sims
   ./run_1G_experiments.sh
   ```
   This populates `results_1G_<policy>` directories with raw CSVR exports.
3. After experiments finish, create per-switch datasets for each policy:
   ```bash
   ./run_all_dataset_creation.sh
   ```
   This writes to `tmp/data/data_1G_<policy>`.

> **Note**: `run_all_dataset_creation.sh` currently lists the policies to build. Update the script if you add/remove policies.

### 3.4 Checking outputs
- Raw outputs: `results_1G_<policy>/<run_id>/...` (large, not directly consumed by RL training).
- Processed datasets: `tmp/data/data_1G_<policy>/<run>__<switch>.csv`.
  Each CSV has the schema expected by `RL_Training/data/loader.py`.

### 3.5 RL-enabled relay configuration
- In the `.ini` file, enable RL on the switches of interest:
  - `**.agg[*].bounce_with_rl_policy = true` and/or `**.spine[*].bounce_with_rl_policy = true`
  - Point `rl_model_path` at the scripted checkpoint exported from `RL_Training`.
- Optional knobs that mirror the training feature pipeline (defaults match the code):
  - `rl_history_length` (default 4) – number of past steps stacked in the history block
  - `rl_flow_expected_packets` (default 1000) – expected packets per flow for the sequence-progress feature
  - `rl_flow_age_tau_us` (default 500) – time constant (µs) for the flow-age exponential feature
  - `rl_ema_half_life_us` (default 80) – half-life (µs) used for deflect/OOO exponential moving averages
  They are optional; omit them to fall back to the defaults above.

---

## 4. Offline RL Training Workflow

### 4.1 Data ingestion
`RL_Training/train.py` automatically discovers datasets. You can either specify explicit directories or rely on the policy list:
```bash
# Explicit
python3 /home/ubuntu/practical_deflection/RL_Training/train.py \
  --data-dirs /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/tmp/data/data_1G_dibs \
              /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/tmp/data/data_1G_ecmp \
              /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/tmp/data/data_1G_probabilistic \
  --algo iql --out-dir /home/ubuntu/practical_deflection/RL_Training/runs/iql_baseline --steps 200000 --batch-size 2048

# Parametric (default list: dibs, ecmp, probabilistic, probabilistic_tb, random, random_tb, sd, threshold, threshold_tb, vertigo)
python3 /home/ubuntu/practical_deflection/RL_Training/train.py \
  --data-base /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/tmp/data \
  --algo iql --out-dir /home/ubuntu/practical_deflection/RL_Training/runs/iql_all --steps 200000 --batch-size 2048
```

Key CLI flags:
- `--algo`: `iql` (default), `cql`, `awr`
- `--steps`: gradient updates (e.g., 50000)
- `--batch-size`: larger batch (2–8k) recommended
- `--log-interval`: frequency of logging to `logs.csv`
- `--save-interval`: frequency for saving `checkpoint_<step>.pt`
- `--w-q`, `--w-l`, `--w-o`, `--w-d`, `--w-f`: reward weights
- `--resume`: path to checkpoint to continue training

Outputs in `RL_Training/runs/<name>`:
- `logs.csv`: training metrics per log interval
- `checkpoint_<step>.pt`: serialized actor/value networks + metadata
- `normalization.json`: feature normalization statistics
- `fqe_eval.json`: stored when `run_fqe.py` is executed

Run training in background:
```bash
nohup python3 /home/ubuntu/practical_deflection/RL_Training/train.py \
  --data-base /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/tmp/data \
  --algo iql --out-dir /home/ubuntu/practical_deflection/RL_Training/runs/iql_all --steps 50000 --batch-size 4096 \
  > /home/ubuntu/practical_deflection/RL_Training/runs/iql_all/train.out 2>&1 &
tail -f /home/ubuntu/practical_deflection/RL_Training/runs/iql_all/train.out
```

### 4.2 Offline Policy Evaluation (FQE)
After training, evaluate the learned actor:
```bash
python3 /home/ubuntu/practical_deflection/RL_Training/run_fqe.py \
  --data-base /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/tmp/data \
  --ckpt /home/ubuntu/practical_deflection/RL_Training/runs/iql_all/checkpoint_50000.pt \
  --steps 300 --batch-size 4096 \
  --out /home/ubuntu/practical_deflection/RL_Training/runs/iql_all/fqe_eval_full.json
```

Options:
- `--t-min`, `--t-max`: restrict to time window (e.g., `--t-min 0.2 --t-max 0.3`)
- `--require-fct`: keep only transitions with known FCT (avoids censored flows)
- `--max-transitions`: limit transitions for faster runs (set in environment variables inside `run_fqe.py`)
- `--device`: `cpu` (default) or `cuda` if GPU available

### 4.3 Compare RL vs behavior policies
Use the plot helper on the FQE JSON:
```bash
python3 RL_Training/plots/compare_policies.py \
  --eval-json RL_Training/runs/iql_all/fqe_eval_0.2_0.3_fct.json \
  --out RL_Training/runs/iql_all/policy_comparison_0.2_0.3_fct.png
```

---

## 5. Design Highlights (for reference)

1. **MDP granularity**: episodes are scoped to (switch_id, FlowID); transitions sorted by timestamp/seq_num.
2. **State features**: congestion (queue utilizations), flow/query context, EMAs, 4-step history stack.
3. **Reward**: dense penalties plus terminal FCT shaping when available; weights tunable via CLI.
4. **Algorithms**: IQL (expectile regression + advantage-weighted actor), CQL (pessimistic Q), AWR (baseline).
5. **OPE**: Fitted Q Evaluation with bootstrap confidence intervals and per-policy diagnostics.

---

## 6. Tips for Future Agents

- Before rerunning OMNeT++ experiments, confirm the desired `.ini` files and `run_*.sh` scripts match the paper scenario; experiments can take significant time.
- Dataset creation (`run_all_dataset_creation.sh`) is idempotent; re-run after new simulations to update CSVs.
- Training logs live in `RL_Training/runs/<run_name>/logs.csv`. Use `tail -f` for monitoring.
- When working in headless/batch environments, use `nohup` or `tmux` to keep long training jobs alive.
- For environment reproducibility, document any additional Python packages or OS dependencies you introduce; update this guide if major changes occur.

---

This document should arm the next agent with sufficient context to (1) rebuild the environment, (2) generate new simulation datasets, and (3) train and evaluate offline RL policies within this repository. Update it whenever the pipeline, dependencies, or directory conventions change.
