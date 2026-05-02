# 🚀 Hybrid Optimization Deep-Dive (OpenMP + CUDA)

This document analyzes the challenges and optimization strategies for multi-GPU systems, comparing the current "Host-Staged" approach with high-performance industry alternatives.

---

## 1. Halo Exchange: Host RAM vs. Peer-to-Peer (P2P)

Multi-GPU systems must exchange boundary rows (halos) at every iteration.

### Scenario A: Host-Staged Exchange (Current Solution)
Data moves: `GPU 0` → `Host RAM` → `GPU 1`.
```cpp
// Thread 0 (GPU 0)
cudaMemcpy(host_buffer, d_t_new, ..., DeviceToHost);
#pragma omp barrier
// Thread 1 (GPU 1)
cudaMemcpy(d_t_new, host_buffer, ..., HostToDevice);
```
- **Limitation**: Every transfer must cross the slow PCIe bus twice. The CPU is used as a middleman, adding significant latency.
- **Bottleneck**: The simulation speed is limited by the **PCIe bandwidth** and the **OpenMP synchronization** (`#pragma omp barrier`).

### Scenario B: Peer-to-Peer (P2P) via NVLink
Data moves: `GPU 0` → `Direct Link` → `GPU 1`.
```cpp
// Enable Peer Access once at startup
cudaDeviceEnablePeerAccess(neighbor_gpu_id, 0);

// Perform direct transfer
cudaMemcpyPeer(d_neighbor_halo, neighbor_gpu_id, d_my_boundary, my_gpu_id, size);
```
- **Efficiency**: Data travels over **NVLink** (up to 900 GB/s) instead of PCIe (up to 32 GB/s).
- **Impact**: Removes the CPU from the communication path, reducing latency by **10x–20x**.

---

## 2. Memory Speed: Pageable vs. Pinned Memory

When copying data between Host and Device, the type of Host memory used matters significantly.

| Memory Type | Description | Transfer Speed |
| :--- | :--- | :--- |
| **Pageable** | Standard `new` or `malloc`. Can be moved by the OS to disk. | Slower (requires internal staging). |
| **Pinned** | `cudaHostAlloc`. Locked in physical RAM. | **2x Faster** (Direct DMA transfer). |

- **Current Implementation**: Uses standard `Grid` memory (pageable).
- **Optimization**: Using **Pinned Memory** allows the GPU's DMA (Direct Memory Access) engine to copy data at full PCIe speed without CPU intervention.

---

## 3. Orchestration: Synchronous vs. Overlapped

### Scenario A: Strict Synchronization (Current Solution)
```cpp
for (t < iterations) {
    Kernel<<<...>>>();
    cudaDeviceSynchronize();
    #pragma omp barrier // Wait for all GPUs
    ExchangeHalos();
}
```
- **The Problem**: The GPUs sit idle while the halos are being copied. All progress stops until the slowest GPU finishes its transfer.

### Scenario B: Overlapping with Streams
We split the grid into **Interior** and **Boundaries**.
1.  **Start Boundary Calculation** on GPU.
2.  **Initiate Asynchronous Halo Transfer** while the GPU simultaneously calculates the **Interior**.
3.  **Efficiency**: By the time the large interior is done, the halo transfer is already finished. This hides the communication cost entirely.

---

## 4. Multi-GPU Load Balancing

If a system has different GPUs (e.g., an RTX 4090 and an RTX 3060), the 4090 will finish its work much faster.

- **Current Strategy**: Static partitioning (each GPU gets an equal number of rows).
- **Optimization**: **Dynamic Domain Decomposition**. Assign more rows to the faster GPU so they both reach the synchronization barrier at the same time.

---

## 🚀 Hybrid Optimization Checklist

1.  [x] **Device Binding**: Correct use of `cudaSetDevice(tid)` to map threads to specific GPUs.
2.  [ ] **Use Pinned Memory**: Replace `std::vector` with `cudaHostAlloc` for the Host buffers.
3.  [ ] **Enable Peer-to-Peer**: Check if `cudaDeviceCanAccessPeer` is true and enable direct transfers.
4.  [ ] **Overlapping**: Use CUDA Streams to hide halo exchange latency.
5.  [ ] **NVLink Detection**: Check if the system supports NVLink for high-speed interconnects.
