# OpenMP Solver (`omp`)

The OpenMP solver parallelizes the computation across multiple CPU cores using a shared-memory model.

## Source File: `solver_omp.cc`

### 1. Parallel Region Optimization

```cpp
18: #pragma omp parallel shared(grid, n, iterations)
19:   {
21:     const double* local_t_old = grid.t_old_ptr();
22:     double* local_t_new = grid.t_new_ptr();
```

- **Line 18**: **Crucial Optimization**. We open the `#pragma omp parallel` region **outside** the time-stepping loop. This prevents the "Fork-Join" overhead (creating and destroying threads) from happening hundreds or thousands of times.
- **Lines 21-22**: We capture the buffer pointers once inside the parallel region. Since they are shared, all threads see the same addresses.

---

### 2. The Time Loop & Work Distribution

```cpp
24:     for (int t = 0; t < iterations; ++t) {
28: #pragma omp for schedule(static)
29:       for (int i = 1; i < n - 1; ++i) {
30:         for (int j = 1; j < n - 1; ++j) {
              // ... stencil math ...
41:         }
42:       }
```

- **Line 28**: `#pragma omp for`. This directive divides the rows of the grid among the available threads.
- **`schedule(static)`**: Since the computational work per row is identical (the stencil is uniform), static scheduling provides the lowest overhead. It pre-assigns ranges of `i` to specific threads at the start of the loop.

---

### 3. Synchronization & Double Buffering

```cpp
47: #pragma omp single
48:       {
49:         grid.SwapBuffers();
50:       }
52:       local_t_old = grid.t_old_ptr();
53:       local_t_new = grid.t_new_ptr();
54:     }
```

- **Line 47**: `#pragma omp single`. Only **one** thread executes the `SwapBuffers()` call. This is vital because swapping is a global operation on the `Grid` object.
- **Implicit Barrier**: The `omp single` block has an implicit barrier at the end. This ensures that **no thread** starts computing the next time step until the swap is finished and all threads have reached this point.
- **Lines 52-53**: After the swap, every thread updates its local pointers to point to the correct buffers for the next iteration.

---

## Performance Characteristics
- **Scaling**: Excellent for medium-sized grids ($N < 4096$).
- **Limitation**: Memory contention. As more threads are added, they compete for the same memory bus bandwidth (the "Memory Wall"), eventually leading to diminishing returns.
