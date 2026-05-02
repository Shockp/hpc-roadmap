# MPI Non-blocking Solver (`mpi_nonblocking`)

This is an advanced version of the MPI solver that implements **Latency Hiding** by overlapping network communication with CPU computation.

## Source File: `solver_mpi_nonblocking.cc`

### 1. Initiating Asynchronous Communication

```cpp
50:     MPI_Request requests[4];
57:       MPI_Irecv(recv_top, cols, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD,
58:                 &requests[req_count++]);
59:       MPI_Isend(send_top, cols, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD,
60:                 &requests[req_count++]);
```

- **Line 50**: We use `MPI_Request` handles to track the status of background network operations.
- **Line 57-59**: `MPI_Irecv` and `MPI_Isend`. The `I` stands for **Immediate**. These functions return instantly, allowing the CPU to continue working while the network hardware (NIC) handles the data transfer.

---

### 2. Overlapping Computation (The "Inner Grid")

While the halos are being sent/received, we calculate the rows that **do not** depend on those halos.

```cpp
78:     for (int i = 2; i < domain.local_rows; ++i) {
80:         // ... compute inner rows ...
90:     }
```

- **Line 78**: Notice the loop starts at `i=2` and ends at `domain.local_rows - 1`. 
- **The Trick**: Row `1` needs the Top Halo, and Row `local_rows` needs the Bottom Halo. However, all rows in between are independent of external data. We compute them now to "hide" the communication time.

---

### 3. Synchronization & Boundary Completion

```cpp
100:       MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
```

- **Line 100**: Before we can compute the "Outer" rows (1 and local_rows), we **must** ensure the halos have actually arrived. `MPI_Waitall` blocks until all the `Isend`/`Irecv` operations are finished.

```cpp
110:       int i = 1; // Compute top boundary row
128:       int i = domain.local_rows; // Compute bottom boundary row
```

- **Lines 110 & 128**: We compute the remaining rows now that the data is guaranteed to be in memory.

---

## Performance Characteristics
- **Efficiency**: Significantly better than the blocking version.
- **Scaling**: Scaling improves because the "cost" of communication is partially absorbed by the time spent calculating the inner grid.
