# MPI + OpenMP K-Means Pipeline

A high-performance C++17 pipeline that distributes a binary dataset across MPI ranks, computes column-level statistics, and runs K-Means clustering — all accelerated with OpenMP thread-level parallelism.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Pipeline Phases](#pipeline-phases)
- [Data Models](#data-models)
- [Parallelization Strategy](#parallelization-strategy)
- [Prerequisites](#prerequisites)
- [Build & Run](#build--run)
- [Documentation](#documentation)

---

## Overview

This project implements a **three-phase HPC pipeline** for numerical data analysis:

1. **I/O & Scatter** — Rank 0 reads a binary dataset and distributes rows across MPI ranks via `MPI_Scatterv`.
2. **Column Statistics** — Computes per-column min, max, mean, and variance using OpenMP reductions + `MPI_Allreduce`.
3. **K-Means Clustering** — Iterative K-Means with data redistribution (`MPI_Alltoallv`) until convergence or a maximum iteration limit.

Rank 0 prints a summary with statistics, final centroids, and wall-clock timings for each phase.

## Project Structure

```
01_OpenMP/
├── CMakeLists.txt              # Build configuration (CMake 3.15+, C++17)
├── include/
│   ├── io_utils.h              # Binary file I/O interface
│   ├── kmeans.h                # K-Means initialization & iteration interface
│   ├── stats.h                 # Column statistics interface
│   └── models/
│       ├── dataset.h           # Flattened 2D dataset (row-major)
│       ├── centroids.h         # Cluster centroids + point counts
│       └── column_stats.h      # Per-column min/max/mean/variance
├── src/
│   ├── main.cpp                # Entry point — orchestrates the 3-phase pipeline
│   ├── io_utils.cpp            # Reads binary dataset files
│   ├── kmeans.cpp              # K-Means core: init, assign, redistribute, update
│   └── stats.cpp               # Column statistics with OpenMP array reductions
├── tools/
│   └── file_generator.cpp      # Utility to generate random binary datasets
├── data/                       # Binary dataset files (gitignored)
└── docs/
    ├── approach_comparison.md  # Comparison of alternative design approaches
    └── design_justification.md # Rationale behind key design choices
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         main.cpp (Rank 0)                           │
│   MPI_Init → Read Binary → MPI_Bcast dims → MPI_Scatterv data      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        ┌──────────┐    ┌──────────┐     ┌──────────┐
        │  Rank 0  │    │  Rank 1  │ ... │  Rank N  │
        └────┬─────┘    └────┬─────┘     └────┬─────┘
             │               │                │
             ▼               ▼                ▼
     ┌───────────────────────────────────────────────┐
     │  Phase 2: ComputeLocalStats (OpenMP parallel) │
     │  → MPI_Allreduce (sum, min, max, count)       │
     └───────────────────────┬───────────────────────┘
                             │
                             ▼
     ┌───────────────────────────────────────────────┐
     │  Phase 3: RunKMeans                           │
     │  ┌─ Assign (OpenMP) ────────────────────────┐ │
     │  │  → Convergence check (MPI_Allreduce)     │ │
     │  │  → Redistribute data (MPI_Alltoallv)     │ │
     │  │  → Update centroids (OpenMP + Allreduce) │ │
     │  └────────── loop until converged ──────────┘ │
     └───────────────────────────────────────────────┘
```

## Pipeline Phases

### Phase 1 — I/O & Scatter

- **Rank 0** reads a binary file with format: `[uint32 n_rows][uint32 n_cols][float data[n_rows × n_cols]]`.
- Dataset dimensions are broadcast via `MPI_Bcast`.
- Rows are distributed with `MPI_Scatterv`; the first `remainder` ranks receive one extra row to handle uneven splits.

### Phase 2 — Column Statistics

- Each rank computes local column statistics using **OpenMP array reductions** (`reduction(+:...)`, `reduction(min:...)`, `reduction(max:...)`).
- Partial results are globally combined via four `MPI_Allreduce` calls (sum, sum-of-squares, min, max).
- Variance uses the computational formula `Var = E[X²] − (E[X])²`, clamped to zero for numerical safety.

### Phase 3 — K-Means Clustering

Each iteration performs four steps:

| Step | Operation | Parallelism |
|------|-----------|-------------|
| **Assignment** | Assign each row to its nearest centroid (squared Euclidean distance) | OpenMP `parallel for` |
| **Convergence** | Check if < 5% of rows changed cluster | `MPI_Allreduce` |
| **Redistribution** | Migrate rows to the rank that owns their cluster | `MPI_Alltoallv` |
| **Update** | Recompute centroids as cluster means | OpenMP + `MPI_Allreduce` |

**Centroid initialization** partitions the global row space evenly among K clusters and averages each partition. This avoids random seed dependencies and ensures reproducibility.

## Data Models

All data structures use **flattened 1D vectors in row-major order** for optimal cache locality and direct compatibility with MPI bulk transfers.

| Structure | Purpose | Key Fields |
|-----------|---------|------------|
| `Dataset` | 2D data matrix | `n_rows`, `n_cols`, `data` (vector\<float\>) |
| `Centroids` | K cluster centers | `num_clusters`, `num_cols`, `data`, `counts` |
| `Column_stats` | Per-column statistics | `min`, `max`, `mean`, `variance` |

## Parallelization Strategy

### OpenMP (intra-node)

- **Stats**: Native OpenMP array reductions for sum, sum-of-squares, min, and max.
- **K-Means assignment**: `omp parallel for` with thread-private displacement counters merged via `omp atomic`.
- **K-Means centroid update**: Thread-private sum/count buffers merged via `omp critical` sections.

### MPI (inter-node)

| Collective | Usage |
|------------|-------|
| `MPI_Bcast` | Broadcast dataset dimensions |
| `MPI_Scatterv` | Distribute rows (handles uneven splits) |
| `MPI_Allreduce` | Aggregate statistics, check convergence, combine centroid sums |
| `MPI_Alltoall` | Exchange row migration counts |
| `MPI_Alltoallv` | Redistribute feature data and cluster assignments |

## Prerequisites

- **CMake** ≥ 3.15
- **C++17** compatible compiler (GCC ≥ 7, Clang ≥ 5)
- **OpenMPI** (or any MPI implementation)
- **OpenMP** support (typically bundled with GCC/Clang)
- *(Optional)* `clang-format` for the `make format` target

## Build & Run

```bash
# Configure and build
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# Generate a test dataset (e.g. 100,000 rows × 8 columns)
./generate_dataset 100000 8 ../data/dataset.bin

# Run with 4 MPI processes
mpirun -np 4 ./CAP_Practica_1
```

### Build Targets

| Target | Description |
|--------|-------------|
| `CAP_Practica_1` | Main MPI + OpenMP executable |
| `generate_dataset` | Binary dataset generator utility |
| `format` | Apply Google C++ Style Guide via `clang-format` |

### Compiler Flags

The build uses aggressive optimization flags for HPC workloads:

- `-O3` — Maximum safe optimization level
- `-march=native` — CPU-specific instruction set (AVX, SSE, etc.)
- `-DNDEBUG` — Disable assertions for production performance
- `-Wall -Wextra` — Full compiler warnings

## Documentation

For in-depth technical documentation, see the `docs/` directory:

- [**approach_comparison.md**](docs/approach_comparison.md) — Comparison of alternative design approaches with performance trade-off analysis.
- [**design_justification.md**](docs/design_justification.md) — Detailed rationale behind each architectural and implementation decision.
