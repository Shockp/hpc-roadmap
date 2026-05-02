# CUDA Solver (`cuda`)

The CUDA solver offloads the simulation to an NVIDIA GPU, utilizing thousands of threads to compute the stencil in parallel.

## Source File: `solver_cuda.cu`

### 1. The GPU Kernel (`StencilKernel`)

```cpp
17: __global__ void StencilKernel(const double* __restrict__ d_t_old,
18:                               double* __restrict__ d_t_new, int n) {
20:   int col = blockIdx.x * blockDim.x + threadIdx.x;
21:   int row = blockIdx.y * blockDim.y + threadIdx.y;
```

- **Line 17**: `__global__` indicates this function runs on the GPU and is callable from the CPU. `__restrict__` tells the compiler that the pointers do not overlap, enabling more aggressive optimizations.
- **Lines 20-21**: Calculation of the global thread coordinate. Each thread is mapped to one cell $(row, col)$ in the heat grid.

#### Shared Memory Optimization

```cpp
26:   __shared__ double tile[kBlockDimY + 2][kBlockDimX + 2];
```

- **Line 26**: We allocate **Shared Memory**. This is ultra-fast on-chip memory (L1-speed) shared by all threads in a block. We add `+2` to accommodate the halos for the block.

```cpp
40:   if (row < n && col < n) {
42:     tile[s_row][s_col] = d_t_old[row * n + col];
45:     if (tx == 0 && col > 0) {
46:       tile[s_row][0] = d_t_old[row * n + (col - 1)];  // Left halo
47:     }
        // ... load other halos ...
60:     __syncthreads();
```

- **Line 42**: Every thread loads its "primary" cell from global VRAM into the fast shared memory tile.
- **Line 45-56**: **Collaborative Loading**. Threads on the edges of the 16x16 block take on the extra work of loading the halo cells from global memory.
- **Line 60**: `__syncthreads()`. A barrier that ensures the entire tile (halos included) is populated before computation begins.

---

### 2. Host Orchestration (`Run`)

```cpp
88:   cudaMalloc(&d_t_old, bytes);
93:   cudaMemcpy(d_t_old, host_grid.t_old_ptr(), bytes, cudaMemcpyHostToDevice);
```

- **Line 88**: Allocates memory in the GPU's global VRAM.
- **Line 93**: **Host-to-Device Copy**. Transfers the initial state from CPU RAM to GPU VRAM over the PCIe bus. This is slow, so we only do it once.

```cpp
108:   for (int t = 0; t < iterations; ++t) {
110:     StencilKernel<<<blocks, threads>>>(d_t_old, d_t_new, n);
114:     cudaDeviceSynchronize();
118:     double* temp = d_t_old; d_t_old = d_t_new; d_t_new = temp;
121:   }
```

- **Line 110**: Kernel launch syntax `<<<blocks, threads>>>`.
- **Line 114**: We synchronize the device at every step. This is necessary because we are swapping pointers in the next line, and we must be sure the GPU is finished writing.
- **Line 118**: Pointer swap on the GPU pointers.

---

## Performance Characteristics
- **Massive Parallelism**: Capable of processing millions of cells simultaneously.
- **PCIe Bottleneck**: The initial and final memory transfers are the most expensive parts. For short simulations, the overhead of copying data can exceed the computation time.
