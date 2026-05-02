# 🛠️ Solver Implementation Guide

This directory contains detailed, line-by-line documentation for each solver implementation in the Heat Diffusion Simulation project. Each document explores the logic, parallelization strategy, and specific optimizations used to achieve high performance.

## 🧭 Documentation Roadmap

### [1. Sequential Solver (`seq`)](sequential.md)
*The Baseline*. Understand the core 5-point stencil math, double buffering, and CPU-level optimizations.

### [2. OpenMP Solver (`omp`)](openmp.md)
*Shared Memory Parallelism*. Learn how to use multi-core CPUs effectively by optimizing parallel regions and work distribution.

### [3. MPI Blocking Solver (`mpi_blocking`)](mpi_blocking.md)
*Distributed Memory Baseline*. Explore domain decomposition and the fundamentals of halo exchange using synchronous communication.

### [4. MPI Non-blocking Solver (`mpi_nonblocking`)](mpi_nonblocking.md)
*Advanced Distributed Computing*. Discover latency hiding techniques that overlap network communication with CPU computation.

### [5. CUDA Solver (`cuda`)](cuda.md)
*GPU Acceleration*. Dive into GPU kernels, shared memory tiles, and managing the Host-to-Device memory bottleneck.

### [6. Hybrid Solver (`hybrid`)](hybrid.md)
*Multi-Level Parallelism*. See how OpenMP and CUDA work together to orchestrate multi-GPU execution.

---

## 🚀 Quick Comparison

| Solver | Parallelism Type | Best For | Main Constraint |
| :--- | :--- | :--- | :--- |
| **Sequential** | None | Small grids, Validation | CPU Single-core speed |
| **OpenMP** | Thread-level (CPU) | Medium grids, Workstations | Memory Bandwidth |
| **MPI** | Process-level (Multi-node) | Massive grids, Clusters | Network Latency |
| **CUDA** | Thread-level (GPU) | Large grids, Single GPU | PCIe Transfer Overhead |
| **Hybrid** | Multi-GPU / Multi-node | Maximum Scale | Synchronization overhead |
