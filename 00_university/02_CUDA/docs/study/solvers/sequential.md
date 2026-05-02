# Sequential Solver (`seq`)

The sequential solver provides the baseline implementation of the 2D heat diffusion simulation. It executes on a single CPU thread and serves as the reference for correctness and speedup calculations.

## Source File: `solver_seq.cc`

### 1. The Main Loop (`Run`)

The `Run` method orchestrates the time-stepping process.

```cpp
6: ProfilerResult SolverSeq::Run(Grid& grid, int iterations) {
7:   ProfilerResult res;
8:   for (int t = 0; t < iterations; ++t) {
9:     auto start_compute = std::chrono::high_resolution_clock::now();
10:    ComputeNextState(grid);
11:    auto end_compute = std::chrono::high_resolution_clock::now();
12:    res.compute_time +=
13:        std::chrono::duration<double>(end_compute - start_compute).count();
14: 
16:    grid.SwapBuffers();
17:   }
18:   return res;
19: }
```

- **Lines 8-17**: The simulation proceeds through discrete time steps.
- **Line 10**: Calls `ComputeNextState` to calculate the temperature grid for the next time step $t+1$.
- **Line 16**: **Double Buffering**. After the entire grid is computed, we swap the "old" and "new" buffers. This is an $O(1)$ pointer swap, avoiding expensive memory copies.

---

### 2. The Stencil Kernel (`ComputeNextState`)

This is where the actual physics simulation happens.

```cpp
21: void SolverSeq::ComputeNextState(Grid& grid) {
22:   const int n = grid.rows();
26:   const double* t_old = grid.t_old_ptr();
27:   double* t_new = grid.t_new_ptr();
31:   constexpr double kInvFour = 0.25;
```

- **Lines 26-27**: We extract raw pointers to the underlying data. This bypasses the overhead of `std::vector` bounds checking and function call stack management inside the performance-critical loop.
- **Line 31**: **Optimization**. Floating-point multiplication (`* 0.25`) is historically faster than division (`/ 4.0`) on many CPU architectures.

#### The Triple Nested Loop

```cpp
36:   for (int i = 1; i < n - 1; ++i) {
37:     for (int j = 1; j < n - 1; ++j) {
39:       const int center = grid.Index(i, j);
40:       const int top = grid.Index(i - 1, j);
41:       const int bottom = grid.Index(i + 1, j);
42:       const int left = grid.Index(i, j - 1);
43:       const int right = grid.Index(i, j + 1);
44: 
46:       t_new[center] =
47:           (t_old[top] + t_old[bottom] + t_old[left] + t_old[right]) * kInvFour;
48:     }
49:   }
```

- **Line 36-37**: We iterate from `1` to `n-1`, effectively skipping the boundaries. This preserves the Dirichlet boundary conditions (constant temperature at the edges).
- **Lines 39-43**: Index calculation. While `grid.Index` is used here, in highly optimized builds, these would be simplified to pointer arithmetic.
- **Line 46-47**: **The 5-Point Stencil**. The new temperature is the average of its neighbors. This is the heart of the finite difference method for the heat equation.

---

## Performance Characteristics
- **Complexity**: $O(iterations \times N^2)$
- **Bottleneck**: Memory bandwidth. The CPU spends most of its time waiting for data to arrive from RAM to the L1/L2 caches.
