# DeepSJRP

A Julia implementation of the deep-learning method developed in

> Ata, B., van Eekelen, W.J.E.C., and Zhong, Y. (2026).
> *A Computational Method for Solving the Stochastic Joint Replenishment Problem in High Dimensions.*

The method tackles the discrete-time stochastic joint replenishment problem (SJRP) by approximating it with a continuous-time impulse control problem and solving the latter via a deep BSDE-style scheme with a Lagrangian loss that dualises an almost-sure stochastic inequality (Eq. 17 in the paper). Two neural networks `H` ≈ `V` and `G` ≈ `∇V` are trained on simulated paths of a carefully designed reference process; an implementable inventory policy is then extracted via Algorithm 3.

This repository reproduces every numerical result in the paper:
- 7 two-dimensional test problems (Table 6, Figures 1–2)
- 27 twelve-dimensional test problems (Tables 3–5)
- 9 fifty-dimensional test problems (Table 7)

## Repository layout

```
DeepSJRP/
├── src/                       # solver source code
├── input/                     # 43 problem-instance JSONs (paper Section 6)
├── configs/                   # 43 per-instance hyperparameter JSONs (Appendix B.2)
├── policies/                  # trained neural networks
│   ├── policies2D/            #   pre-trained NN per 2D instance (+ baked grid)
│   ├── policies12D/           #   pre-trained NN per 12D instance
│   └── policies50D/           #   pre-trained NN per 50D instance (both methods)
├── output/
│   ├── 2d_mdp/                # optimal MDP policies for 2D benchmarks
│   ├── benchmark_params/      # optimal (R,S)/(Q,S)/(R,Q,S)/can-order parameters
│   ├── nn_training/           # per-method NN training outputs
│   └── final_results/         # final tables: NN cost ± %SE and relative gaps
├── logs/                      # raw stdout from training / simulation runs
├── run_training.jl            # CLI: train NN for one instance
├── run_simulations.jl         # CLI: benchmark search & final simulations
├── run_mpdsolver2d.jl         # CLI: exact MDP optimal policy (2D only)
├── Project.toml               # Julia package dependencies
└── Manifest.toml              # pinned dependency versions
```

## Instance naming convention

Each instance is named `<dim>D_Case<letters>` where the letters encode the test-problem parameters:

**12D — `Case<Variability><c₀><p>`** (Tables 3–5, 10–12):

| letter | Variability        | c₀  | p   |
|--------|--------------------|-----|-----|
| L      | Poisson            | 20  | 10  |
| M      | NegBin, CV = 0.5   | 100 | 50  |
| H      | NegBin, CV = 1.0   | 200 | 100 |

**2D — `Case<Variability><c₀><p>`** (Table 6, Table 9), base case `MMM` with one-off variants. The c₀ and p grids differ from 12D because the variants are defined around the MMM base:

| letter | Variability        | c₀ | p   |
|--------|--------------------|----|-----|
| L      | Poisson            | 20 | 10  |
| M      | NegBin, CV = 0.5   | 50 | 50  |
| H      | NegBin, CV = 1.0   | 100| 100 |

**50D — `Case<Variability><c₀>`** (Table 7, Table 13):

| letter | Variability        | c₀  |
|--------|--------------------|-----|
| L      | Poisson            | 50  |
| M      | NegBin, CV = 0.5   | 150 |
| H      | NegBin, CV = 1.0   | 250 |



## Requirements

The paper results were produced with the following stack:

**Toolchain**
- Julia 1.11.7
- LLVM 16.0.6

**Julia packages** (declared in `Project.toml`, version-pinned in `Manifest.toml`)
- CUDA.jl 5.11.0, cuDNN.jl, Flux.jl, Zygote.jl, ForwardDiff.jl
- LBFGSB.jl, Distributions.jl, ProgressMeter.jl, TickTock.jl
- JSON.jl, BSON.jl, JLD2.jl
- GPUArrays 11.4.1, GPUCompiler 1.9.1, KernelAbstractions 0.9.41

**GPU stack (optional but recommended)**
- CUDA toolkit: runtime 12.8 (local installation), compiler 12.9, driver 580.126.20 for CUDA 13.2
- CUDA libraries: CUBLAS 12.8.4, CURAND 10.3.9, CUFFT 11.3.3, CUSOLVER 11.7.3, CUSPARSE 12.5.8, CUPTI 2025.1.1 (API 12.8.0), NVML 13.0.0
- CUDA_Driver_jll 13.2.0, CUDA_Compiler_jll 0.4.2, CUDA_Runtime_jll 0.21.0, CUDA_Runtime_Discovery 1.0.0

**Hardware (Appendix B.1)**
- NVIDIA H100 80 GB HBM3 (sm_90)
- AMD EPYC 9334 32-core CPU (64 logical threads), 694 GB RAM
- BLAS threads = 1, Julia threads = 10 used during the reported runs

The code runs on CPU as well; GPU is auto-detected via CUDA.jl. 

## Setup

```bash
git clone https://github.com/wvaneeke/DeepSJRP.git
cd DeepSJRP

# Instantiate dependencies (pinned by Manifest.toml)
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Workflow

There are two ways to use this repository.

### Option A — Replicate published numbers without retraining

The repository ships with every trained neural network and every benchmark-parameter file used in the paper. To regenerate `output/final_results/final_results_<instance>.json` for an instance, run only the final-simulation step:

```bash
julia run_simulations.jl <instance_name> --mode final
```

For example (with `--nn-method` set to match the "Opt. Method" column of Tables 9–13, where Eq. 22 corresponds to `solve_inf` and Eq. 25 to `find_stationary`):

```bash
julia run_simulations.jl 12D_CaseLLL --mode final --nn-method find_stationary
julia run_simulations.jl 50D_CaseHH  --mode final --nn-method find_stationary
julia run_simulations.jl 2D_CaseMMM  --mode final --nn-iterations 1000 --nn-method solve_inf
```

`--mode final` reads `configs/hyperparams_<instance>.json` (for hyperparameters), loads the trained NN weights from `policies/policies<dim>D/`, loads the benchmark policy parameters from `output/benchmark_params/`, then simulates each policy under the original discrete-time SJRP for 10,000 weekly periods. By default it uses 100,000 sample paths per benchmark policy and 100 sample paths for the NN policy. The NN sample count can be overridden via `--nn-iterations <int>`; the paper uses 1,000 for all two-dimensional instances and 100 for the twelve- and fifty-dimensional instances.

The output JSON contains the NN cost ± standard error and the relative gap of each benchmark policy versus the NN, matching the entries in Tables 3–7.

### Option B — Full pipeline from scratch

Each entry-point script documents its arguments and options in a docstring at the top of the file and via `--help`:

```bash
julia run_mpdsolver2d.jl --help     # exact MDP solver (2D)
julia run_simulations.jl --help     # benchmark search & final simulations
julia run_training.jl --help        # NN training
```

To rerun the entire experimental pipeline for a single instance:

```bash
# (2D only) Compute the optimal MDP policy that serves as the
# best-possible benchmark in Table 6. Output goes to output/2d_mdp/.
julia run_mpdsolver2d.jl 0.5 50.0 50.0 2D_CaseMMM

# 1. Preliminary benchmark search. Finds optimal (s,S), (R,S), (Q,S),
#    (R,Q,S), and can-order parameters via 10K paths × 10K periods per
#    candidate, writes them to output/benchmark_params/, and updates
#    the reference_process section of the config with the (Q,S) and
#    can-order results so they can be reused as reference policies
#    during training. CAUTION: this overwrites the per-instance config.
julia run_simulations.jl 2D_CaseMMM --mode preliminary

# 2. Train the two neural networks H_θ and G_ϑ via Algorithm 2.
#    Honors the schedules in configs/hyperparams_<instance>.json.
#    Writes per-method results to output/nn_training/ and the trained
#    weights to policies/policies<dim>D/.
julia run_training.jl 2D_CaseMMM

# 3. High-precision final simulations (as in Option A).
julia run_simulations.jl 2D_CaseMMM --mode final
```


## Notes on configuration

- **`benchmark_source`** in each config is either `"QS"` or `"can_order"`, indicating which benchmark policy supplied the order-up-to vector `S` used as the mean of the reference process (Eq. 16). For the high-variance 2D test problem (`2D_CaseHMM`), the paper uses the MDP-optimal `S`; the config encodes this under the `can_order` slot.
- **`annual_cv`** in the input JSONs is ignored by the solver when `distribution = "Poisson"`. The value `0.2` is a placeholder; the true per-item CVs of the Poisson model are determined by the demand rates and range from 0.158 to 0.224 (cf. Section 6).
- For `12D_CaseHLL`, the paper uses ψ = 2 in Eq. (16). This is implemented by doubling the `S_input` vector in `configs/hyperparams_12D_CaseHLL.json` (with optimisation bounds correspondingly adjusted to `[0.75, 1.25]` relative to the doubled vector).
- **Standard errors on relative gaps** are computed via the delta method. Common random numbers cannot be applied because the benchmark and NN simulations use different numbers of sample paths. The same random seed is used on both sides, so the shared portion of the paths does induce a small positive covariance between the two cost estimators; ignoring it makes the reported SEs slightly conservative.

## Citation

If you use this code, please cite:

```
@unpublished{AtaVanEekelenZhong2026,
  author = {Ata, Bari\c{s} and van Eekelen, Wouter J.E.C. and Zhong, Yuan},
  title  = {A Computational Method for Solving the Stochastic Joint
            Replenishment Problem in High Dimensions},
  year   = {2026},
  note   = {Working paper, Booth School of Business, University of Chicago}
}
```
