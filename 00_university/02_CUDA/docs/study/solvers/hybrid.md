# Hybrid Solver (`hybrid`)

The hybrid solver represents the peak of HPC integration, using **OpenMP** to manage multiple **NVIDIA GPUs** simultaneously.

## Source File: `solver_hybrid.cu`

### 1. The Multi-GPU Strategy

```cpp
67: #pragma omp parallel num_threads(num_gpus)
68:   {
69:     int tid = omp_get_thread_num();
76:     cudaSetDevice(tid);
```

- **Line 67**: We use OpenMP to spawn one CPU thread for **every detected GPU** in the system.
- **Line 76**: `cudaSetDevice(tid)`. Each OpenMP thread binds itself to a specific physical GPU. Thread 0 controls GPU 0, Thread 1 controls GPU 1, and so on.

---

### 2. Domain Partitioning

```cpp
79:     int num_compute_rows = global_n - 2;
82:     int local_rows = base_rows + (tid < remainder ? 1 : 0);
```

- **Line 79-82**: We partition the internal compute rows among the GPUs. Each GPU receives a slice of the global grid.

---

### 3. Multi-GPU Halo Exchange

Since GPUs (usually) cannot communicate directly with each other's memory easily, we use the **Host RAM** as a "staging area" for halo exchange.

```cpp
121:         cudaMemcpy(&h_t_new[global_start_row * cols], &d_t_new[1 * cols],
122:                    cols * sizeof(double), cudaMemcpyDeviceToHost);
```

- **Line 121**: Every GPU copies its top/bottom compute rows back to the CPU's `h_t_new` buffer.

```cpp
132: #pragma omp barrier
```

- **Line 132**: **Critical Synchronization**. We must wait for all OpenMP threads (all GPUs) to finish their transfers to Host RAM. Without this, a GPU might read its neighbor's halo before the neighbor has finished writing it.

```cpp
136:         cudaMemcpy(&d_t_new[0], &h_t_new[(global_start_row - 1) * cols],
137:                    cols * sizeof(double), cudaMemcpyHostToDevice);
```

- **Line 136**: Once the barrier is passed, each GPU pulls its neighbor's data from Host RAM into its own local halo rows.

---

## Performance Characteristics
- **Scale**: Designed for multi-GPU workstations or HPC nodes.
- **Complexity**: High overhead due to the constant Device <-> Host data movement at every iteration. This solver is only beneficial for extremely large grids that do not fit on a single GPU.
