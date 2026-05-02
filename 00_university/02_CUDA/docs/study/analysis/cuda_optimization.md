# 🚀 CUDA Optimization Deep-Dive

This document explores how to maximize GPU performance by analyzing different CUDA implementation strategies for the heat diffusion simulation.

---

## 1. Memory Hierarchy: Global vs. Shared Memory

### Implementation A: Naive Global Memory Access
In a naive implementation, every thread simply reads its 4 neighbors directly from the GPU's Global Memory (VRAM).
```cpp
__global__ void NaiveKernel(double* old, double* next, int n) {
    // ... index calculation ...
    next[idx] = 0.25 * (old[top] + old[bottom] + old[left] + old[right]);
}
```
- **The Problem**: Each cell is read from VRAM **4 times** (once by each neighbor). Global memory access is slow compared to on-chip caches.
- **Bottleneck**: The kernel becomes **Memory Bound** quickly, reaching the VRAM bandwidth limit.

### Implementation B: Shared Memory Tiling (Current Solution)
We load a "tile" of data into a small, ultra-fast `__shared__` memory array.
- **Efficiency**: Each data point is read from VRAM **only once**.
- **Impact**: Drastically reduces VRAM traffic, allowing the GPU to spend more time on math and less time waiting for data.

---

## 2. Occupancy & Thread Block Sizing

GPU performance depends on **Occupancy**—the ratio of active warps to the maximum possible warps on a Streaming Multiprocessor (SM).

### Scenario: 16x16 vs. 32x32 Blocks
- **Current (16x16)**: 256 threads per block.
- **Alternative (32x32)**: 1024 threads per block.

**Why 16x16 is often better for Stencils:**
1.  **Shared Memory Limits**: A 32x32 tile requires much more shared memory ($34 \times 34 \times 8$ bytes $\approx 9.2$ KB). Since an SM has a fixed amount of shared memory (e.g., 96KB), larger blocks can limit the number of blocks that can run concurrently.
2.  **Register Pressure**: Complex kernels use more registers. If a 32x32 block uses too many registers, the GPU can't fit enough warps to hide memory latency.

---

## 3. Minimizing PCIe Overhead (Host-Device Transfers)

The PCIe bus is the narrowest bottleneck in GPU computing.

### Strategy: "Stay on the Device" (Current Solution)
Notice how our loop happens **inside** the `Run` method:
```cpp
cudaMemcpy(..., HostToDevice); // ONCE
for (int t = 0; t < iterations; ++t) {
    Kernel<<<...>>>(...);
    cudaDeviceSynchronize();
    swap_pointers(); // On the Device
}
cudaMemcpy(..., DeviceToHost); // ONCE
```
- **Efficiency**: We keep the data in the GPU's fast VRAM for the entire simulation. We only pay the PCIe transfer cost at the very beginning and very end. 

---

## 4. Advanced: Asynchronous Streams & Overlapping

If we had multiple GPUs or needed to process data larger than VRAM, we would use **CUDA Streams**.

```cpp
cudaStream_t stream1, stream2;
cudaMemcpyAsync(..., stream1); // Copy chunk 1
Kernel<<<..., stream1>>>(...); // Start calculating chunk 1
cudaMemcpyAsync(..., stream2); // Start copying chunk 2 while chunk 1 is calculating
```
- **Optimization**: This allows the **Network (PCIe)** and the **Compute (GPU)** to work at the same time, effectively "hiding" the copy time.

---

## 5. Kernel-Level "Micro-Optimizations"

### Pointer Aliasing (`__restrict__`)
Our kernel uses `const double* __restrict__ d_t_old`.
- **Why**: This tells the compiler that the data in `d_t_old` will not be changed via `d_t_new`. This allows the GPU to use **Read-Only Data Caches** (LDG instructions), which are separate from the standard L1 cache and provide higher bandwidth.

### Grid Stride Loops
For very large grids or varying hardware, using a **Grid Stride Loop** inside the kernel ensures portability:
```cpp
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    // compute...
}
```
- **Benefit**: Allows the same kernel to run efficiently regardless of the physical number of SMs on the specific GPU (e.g., a laptop GPU vs. an H100).

---

## 🚀 CUDA Optimization Checklist

1.  [x] **Use Shared Memory** to minimize Global Memory redundant reads.
2.  [x] **Minimize Memcpy** (Keep the simulation loop on the device).
3.  [x] **Use `__restrict__`** to enable Read-Only cache optimizations.
4.  [ ] **Profile with Nsight Compute**: Check if the kernel is "Latency Bound" or "Memory Bound".
5.  [ ] **Implement Async Streams** if overlapping with Host tasks is required.
