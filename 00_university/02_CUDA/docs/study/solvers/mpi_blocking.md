# MPI Blocking Solver (`mpi_blocking`)

The MPI solver distributes the grid across multiple independent processes (potentially on different physical machines). Each process is responsible for a horizontal slice of the grid.

## Source File: `solver_mpi_blocking.cc`

### 1. Domain Decomposition

```cpp
18:   DomainDecomposition domain(global_n, rank, size);
21:   Grid grid(domain.total_rows, global_n, domain.is_top_rank,
22:             domain.is_bottom_rank);
```

- **Line 18**: The `DomainDecomposition` helper calculates which rows of the global $N \times N$ grid this specific rank is responsible for.
- **Line 21**: Each process allocates a local `Grid` that contains its assigned rows **plus two extra rows** for halos (top and bottom ghost cells).

---

### 2. Halo Exchange (Communication)

Because the stencil at the edge of a local sub-grid requires data from a neighbor process, we must exchange "Halo" rows at every time step.

```cpp
44:     double* recv_top = &t_old[0 * cols];
45:     double* send_top = &t_old[1 * cols];
46:     double* send_bottom = &t_old[domain.local_rows * cols];
47:     double* recv_bottom = &t_old[(domain.local_rows + 1) * cols];
```

- **Lines 44-47**: We identify the memory addresses for sending and receiving. 
  - `recv_top` is the ghost row (Row 0).
  - `send_top` is our first real compute row (Row 1).

```cpp
51:     if (!domain.is_top_rank) {
52:       MPI_Sendrecv(send_top, cols, MPI_DOUBLE, rank - 1, 0, recv_top, cols,
53:                    MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
54:     }
```

- **Line 52**: `MPI_Sendrecv`. This is a blocking call that performs a simultaneous send and receive. It is safer than separate `MPI_Send` and `MPI_Recv` calls because it avoids potential deadlocks.
- **Tags**: We use tags `0` and `1` to distinguish between "upward" and "downward" data movement.

---

### 3. Computation & Buffer Swapping

```cpp
71:     for (int i = 1; i <= domain.local_rows; ++i) {
72:       for (int j = 1; j < cols - 1; ++j) {
            // ... stencil math ...
81:       }
82:     }
89:     grid.SwapBuffers();
```

- **Line 71**: The loop iterates from row `1` to `domain.local_rows`. Note that `i=0` and `i=local_rows+1` are the halos we just received.
- **Line 89**: Just like the sequential version, each process swaps its *local* buffers.

---

## Performance Characteristics
- **Network Bound**: Performance is heavily dependent on the interconnect speed (e.g., Ethernet vs InfiniBand).
- **Latency**: In this blocking version, the CPU sits idle while waiting for the `MPI_Sendrecv` to complete, which is inefficient.
