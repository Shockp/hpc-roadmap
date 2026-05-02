# ⚡ OpenMP Optimization Deep-Dive

This document explores various strategies to optimize OpenMP parallel regions, comparing different scenarios and their impact on the Heat Diffusion simulation performance.

---

## 1. Parallel Region Orchestration

### Scenario A: Parallelization Inside the Time Loop
```cpp
for (int t = 0; t < iterations; ++t) {
    #pragma omp parallel for
    for (int i = 1; i < n - 1; ++i) { ... }
    grid.SwapBuffers();
}
```
- **Behavior**: Threads are created (forked) and destroyed (joined) at **every single iteration**.
- **Impact**: **High Overhead**. If you have 1000 iterations, you pay the thread creation cost 1000 times. For small grids, the overhead can be larger than the actual computation.

### Scenario B: Parallelization Outside the Time Loop (Current Solution)
```cpp
#pragma omp parallel
{
    for (int t = 0; t < iterations; ++t) {
        #pragma omp for
        for (int i = 1; i < n - 1; ++i) { ... }
        #pragma omp single
        { grid.SwapBuffers(); }
    }
}
```
- **Behavior**: Threads are created **once** at the start.
- **Efficiency**: **Optimal**. The overhead is negligible, making it efficient even for very small workloads.

---

## 2. Scheduling Strategies

OpenMP allows you to control how iterations are mapped to threads via `schedule()`.

| Strategy | Description | Best For... | Overhead |
| :--- | :--- | :--- | :--- |
| **`static`** | Pre-assigns fixed chunks to threads. | **Uniform workloads** (like our stencil). | Lowest |
| **`dynamic`** | Threads request new chunks when finished. | **Irregular workloads** (e.g., Mandelbrot set). | High |
| **`guided`** | Starts with large chunks, then decreases. | Balancing load without extreme overhead. | Medium |

### Why `static` is best for Stencil?
In our heat simulation, every cell $(i, j)$ requires exactly 4 additions and 1 multiplication. The work is **perfectly uniform**. Using `dynamic` would introduce unnecessary "manager" overhead to track which thread gets which row.

---

## 3. Loop Collapsing

When we have nested loops, we usually only parallelize the outer one.

### Single Loop Parallelization (Current)
```cpp
#pragma omp for
for (int i = 1; i < n - 1; ++i) {
    for (int j = 1; j < n - 1; ++j) { ... }
}
```
- **Limitation**: If you have 128 cores but only 64 rows ($N=64$), 64 cores will sit idle because there aren't enough "chunks" to distribute.

### Collapsed Loops
```cpp
#pragma omp for collapse(2)
for (int i = 1; i < n - 1; ++i) {
    for (int j = 1; j < n - 1; ++j) { ... }
}
```
- **Efficiency**: OpenMP merges the two loops into one large virtual loop of $(N-2)^2$ iterations.
- **Benefit**: Greatly increases the number of work units, ensuring **all cores stay busy** even on small grids or massive supercomputers.

---

## 4. SIMD Vectorization

Modern CPUs can perform operations on multiple data points at once using SIMD (Single Instruction, Multiple Data) like AVX-512.

```cpp
#pragma omp for simd
for (int i = 1; i < n - 1; ++i) {
    for (int j = 1; j < n - 1; ++j) { ... }
}
```
- **Impact**: Encourages the compiler to use vector registers. For our heat equation, this can potentially provide a **4x to 8x speedup** on top of the multi-threading speedup, as it processes multiple `double` values in a single clock cycle.

---

## 5. False Sharing (The Performance Killer)

False sharing occurs when different threads update different variables that happen to reside on the **same CPU cache line** (usually 64 bytes).

- **In our Solver**: Because each thread handles a different **row**, and rows are large ($N \times 8$ bytes), threads are working on memory far apart from each other.
- **Efficiency**: Our row-based decomposition naturally **prevents false sharing**. If we had used column-based decomposition, we might have seen severe performance degradation as multiple threads fought for the same cache lines.

---

## 🚀 Summary Checklist for OpenMP Peak Performance

1.  [x] **Parallelize once** (Minimize Fork/Join overhead).
2.  [x] **Use `static` scheduling** for uniform stencil work.
3.  [ ] **Use `collapse(2)`** if running on very high core counts.
4.  [ ] **Enable `simd`** to leverage hardware vector registers.
5.  [ ] **Check Thread Affinity**: Use `OMP_PROC_BIND=true` to prevent the OS from moving threads between CPU cores, which destroys cache locality.
