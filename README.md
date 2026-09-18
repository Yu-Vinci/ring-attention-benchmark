# Ring Attention Multi-GPU Benchmark

Standalone C/CUDA benchmark package for ring communication and Ring Attention experiments, with a shared Makefile and OpenMPI/Slurm launch scripts.

This repository focuses on running benchmark variants. The related [Multi-GPU-Flashattention](https://github.com/Yu-Vinci/Multi-GPU-Flashattention) repository contains workload modeling and scheduling research. This package does not include the newer local C++ revision or its results.

## Benchmark variants

Three communication families are exposed through five communication variants and seven attention variants:

| Family | Communication variants | Additional attention variant |
| --- | --- | --- |
| Staged MPI | `staged`, `staged_Isendrecv` | `staged_Isendrecv_overlap` |
| CUDA-aware MPI | `cuda_aware`, `cuda_aware_Isendrecv` | `cuda_aware_Isendrecv_overlap` |
| NCCL | `nccl` | — |

[Communication-only sources](src/loop/) isolate ring data exchange. [Attention sources](src/attention/) combine local attention, exchange and output merging. These are experimental CUDA implementations, not a full-model training benchmark.

## Requirements

- Linux, Bash 4+, Make and a host C/C++ compiler.
- NVIDIA GPUs and a CUDA toolkit.
- MPI; CUDA-aware support for the `cuda_aware` variants.
- NCCL for the NCCL variant.
- Python 3 for output comparisons.

Use rank counts and GPU placement supported by your allocation. A configuration listing 2/4/8 ranks is a proposed sweep, not evidence that all those configurations were measured.

## Quick start

```bash
git clone https://github.com/Yu-Vinci/ring-attention-benchmark.git
cd ring-attention-benchmark
cp config.example.env benchmark.env
# Edit CUDA_HOME, MPI_HOME, NCCL_HOME and the benchmark matrix for your machine.
make print-config
```

From a normal shell inside your GPU allocation, prepare one named run:

```bash
export RUN_LABEL="manual_$(date +%Y%m%d_%H%M%S)"
export RESULTS_ROOT="$PWD/results"
./run_suite.sh
```

`run_suite.sh` builds the selected targets, collects environment information and writes launch examples. It does not launch MPI jobs. Keeping the run label fixed before preparation ensures that subsequent logs use the directory that was created.

Run an attention comparison before collecting timings:

```bash
env NP=2 SIZE=262144 WARMUP=1 ITERS=1 LAUNCHER=mpirun \
  ./check_attention_correctness.sh
```

Then launch the benchmark wrappers from the same shell:

```bash
env NP=2 LAUNCHER=mpirun ./benchmark_comm.sh \
  > "$RESULTS_ROOT/$RUN_LABEL/comm_np2.log" 2>&1
env NP=2 LAUNCHER=mpirun ./benchmark_attention.sh \
  > "$RESULTS_ROOT/$RUN_LABEL/attention_np2.log" 2>&1
```

The wrappers call `mpirun` or `srun` themselves. Do not wrap these shell scripts in another MPI launcher. To set sizes directly, use `SIZES`, for example `SIZES="262144 524288"`; check the printed sizes and backend list before collecting results.

## Cluster launch options

- [launch_mpirun.example.sh](launch_mpirun.example.sh) provides an OpenMPI sweep. Export a fixed `RUN_LABEL` and `RESULTS_ROOT` before invoking it so preparation and log paths agree.
- [batch_slurm.example.sh](batch_slurm.example.sh) provides a Slurm submission example. Adapt resource requests and installation paths to your cluster before submitting.
- Inside an existing Slurm GPU allocation, use `LAUNCHER=srun` with the wrappers. Set `SRUN_MPI_TYPE` only to a plugin supported by that cluster.

For manual launches, the preparation step writes exact command examples to `results/<run_label>/launch_examples.txt`.

## Configuration and build

[config.example.env](config.example.env) documents the machine paths and sweep settings. The Makefile reads `benchmark.env` by default.

| Setting | Purpose |
| --- | --- |
| `CONFIG` | Configuration file; default `benchmark.env` |
| `NP` / `NP_LIST` | One wrapper launch / outer sweep rank counts |
| `BACKENDS` / `ATTENTION_BACKENDS` | Communication / attention variants |
| `SIZES` | Sizes passed directly to a benchmark wrapper |
| `COMM_SIZES` / `ATTENTION_SIZES` | Preparation and sweep configuration; verify the wrappers print the intended sizes |
| `WARMUP` / `ITERS` | Warmup / timed iterations |
| `MPIRUN`, `HOSTFILE`, `MPIRUN_EXTRA_ARGS` | OpenMPI launch settings |
| `SRUN`, `SRUN_MPI_TYPE`, `SRUN_EXTRA_ARGS` | Slurm launch settings |
| `RUN_LABEL` / `RESULTS_ROOT` | Run identity and output location |

Useful build targets:

```bash
make print-config
make all
make comm
make attention
make check-attention-correctness
make clean
```

## What the comparison checks

[check_attention_correctness.sh](check_attention_correctness.sh) compares `staged_Isendrecv` with its overlap variant and `cuda_aware_Isendrecv` with its overlap variant. It saves per-rank output dumps and reports absolute and relative differences.

This is a pairwise regression check, not an independent full-attention reference check. Agreement between two implementations does not prove that both are correct. The related research repository has a separate CPU-reference workflow.

## Results and reporting

Preparation writes `manifest.txt`, `build.log`, `env_local.txt`, `run_context.env` and `launch_examples.txt`. The launch commands save `comm_np<N>.log` and `attention_np<N>.log`.

When reporting performance, record the source revision, hardware/topology, problem shape, precision, backend, warmup and iteration counts. Keep communication-only timings separate from end-to-end attention timings. No new performance results are claimed by this documentation update.

Gang Yu (@Yu-Vinci)
