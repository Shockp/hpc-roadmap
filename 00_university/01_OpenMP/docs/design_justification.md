# Design Justification — Assignment 1: HPC Pipeline with K-Means

## 1. Introduction and Objectives

The objective of this assignment is to design a system capable of processing large volumes of numerical data in the shortest possible time. The program executes three phases on a binary dataset:

1. **I/O and distribution** — Reading the dataset and distributing it among MPI processes.
2. **Column-wise statistics** — Parallel computation of minimum, maximum, mean, and variance.
3. **K-Means** — Iterative clustering with data redistribution across nodes.

To maximize performance, a **hybrid parallelization** strategy is employed: **OpenMPI** for coarse-grained tasks across nodes and **OpenMP** for fine-grained computation within each node. The following sections justify each design decision by explaining *why* it is faster than the alternatives.

---

## 2. Data Structure Design: Data-Oriented Design

### 2.1. Decision: Flat Contiguous Vectors (SoA) Instead of Classes with Pointers (AoS)

The `Dataset` and `Centroids` structures store their data in a single contiguous `std::vector<float>` in row-major order, instead of using individual objects like `class Point { float x, y, z; }`.

**Why is AoS slower?**

With an Array of Structures, each point would be an independent object potentially scattered in memory. When iterating over N points to compute distances or statistics, the CPU needs to jump from one object to another:

```
AoS in memory (pointers to individual objects):
  Point0 [x0, y0, z0]  →  Point1 [x1, y1, z1]  →  Point2 [x2, y2, z2]  ...
  ↑ potentially non-contiguous, generates cache misses
```

**Why is SoA/contiguous vector faster?**

With a flat contiguous vector, the data is laid out sequentially in RAM. When the CPU reads the first float, the cache line (typically 64 bytes = 16 floats) automatically brings along the next 15 values:

```
Contiguous vector in memory:
  [x0, y0, z0, x1, y1, z1, x2, y2, z2, ...]
  ↑ sequential access → automatic prefetch → ~0 cache misses
```

**Concrete example:** In the K-Means Euclidean distance loop (`kmeans.cpp:159-162`), each iteration accesses `row_ptr[c]` and `centroid_ptr[c]` sequentially. With contiguous data, the CPU's hardware prefetcher anticipates these accesses and preloads the next cache lines before they are needed. With AoS, each access to a new point would be a potential *cache miss*, multiplying memory latency by a factor of 10-100×.

**Additional benefit for MPI:** Contiguous data allows sending entire blocks with `MPI_Scatterv` and `MPI_Alltoallv` without needing to serialize/deserialize complex structures. A single `MPI_Scatterv` on `dataset.data.data()` transmits thousands of rows in one operation, whereas with AoS we would need to pack each object into a flat buffer first.

**Benefit for auto-vectorization (SIMD):** The `-march=native` flag allows the compiler to generate AVX/AVX2 instructions that process 8 floats simultaneously. This only works with contiguous and aligned data; with AoS, the compiler cannot guarantee contiguity and disables vectorization.

### 2.2. Decision: `float` Instead of `double` for Data

The dataset data is stored as `float` (32 bits) instead of `double` (64 bits).

**Why is it faster?**

- **Twice the data per cache line:** A 64-byte line stores 16 floats but only 8 doubles. This doubles memory bandwidth efficiency.
- **Twice the elements per SIMD instruction:** A 256-bit AVX register processes 8 floats versus 4 doubles per instruction.
- **Half the MPI bandwidth:** Transferring N rows of floats requires half the bytes compared to doubles, reducing network latency.

**Where is `double` used?** Exclusively in **intermediate accumulators** (`local_sums`, `global_sums`) to avoid rounding errors when summing millions of small values. This is a deliberate trade-off: numerical precision is gained where it is critical (accumulation) without sacrificing performance in the massive storage and transmission of data.

### 2.3. Decision: Row Pointer Access (`GetRowPtr`)

The `Dataset` and `Centroids` structs expose a `GetRowPtr(row)` method that returns a direct pointer to the beginning of each row.

**Why is it faster than 2D indexing?**

Computing `data[row * n_cols + col]` on each access requires a multiplication per iteration. With `GetRowPtr`, the multiplication is performed only once per row and the inner loop traverses the pointer sequentially:

```cpp
// With GetRowPtr (1 multiplication per row):
const float *row_ptr = local_data.GetRowPtr(r);  // row_ptr = &data[r * n_cols]
for (uint32_t c = 0; c < num_cols; ++c)
    sum += row_ptr[c];  // just a pointer increment

// Without GetRowPtr (1 multiplication per element):
for (uint32_t c = 0; c < num_cols; ++c)
    sum += data[r * n_cols + c];  // multiplication on each iteration
```

Additionally, the direct pointer makes it easier for the compiler to detect the linear access pattern and generate prefetch and vectorization instructions.

---

## 3. I/O: Native Binary Format

### 3.1. Decision: Direct Binary Reading Instead of CSV/Text

The dataset is read with `file.read()` directly into a `std::vector<float>`, without text parsing.

**Why is CSV/text slower?**

1. **Character-by-character parsing:** Reading a CSV requires identifying delimiters (`,`, `\n`), converting strings to float (`std::stof` or `sscanf`), and managing intermediate buffers. Each string→float conversion requires dozens of instructions.
2. **Size on disk:** The number `99.743652` takes 9 bytes as text but only 4 bytes as a binary `float`. A dataset of 10 million rows × 10 columns would occupy ~900 MB in CSV versus ~400 MB in binary.
3. **Reading in a single `read()`:** With binary format, the entire dataset is read in a single system call: `file.read(data.data(), total_elements * sizeof(float))`. No loops, no parsing, no intermediate allocations.

**Magnitude example:** For a dataset of 10M rows × 10 columns:
- **CSV:** ~900 MB on disk, ~5-10 seconds of parsing (CPU-limited).
- **Binary:** ~400 MB on disk, ~0.5-1 second (limited only by disk/SSD).

### 3.2. Decision: `std::optional` for I/O Error Handling

`ReadBinaryFile` returns `std::optional<Dataset>` instead of throwing exceptions or using error codes.

**Why?** C++ exceptions have added cost in stack *unwinding* and can interfere with compiler optimizations on the success path. `std::optional` allows checking for success with no additional cost in the normal case (the compiler optimizes the optional's internal boolean).

---

## 4. Parallelization Strategy

### 4.1. Inter-Node Level: OpenMPI

#### 4.1.1. Initial Distribution: `MPI_Scatterv` with Load Balancing

Data is distributed evenly among MPI processes, with exact balancing when rows are not evenly divisible:

```cpp
int local_rows = total_rows / num_procs;
int remainder = total_rows % num_procs;
if (rank < remainder) local_rows++;  // first 'remainder' ranks receive one extra row
```

**Why not simple `MPI_Scatter`?** `MPI_Scatter` requires all processes to receive exactly the same number of elements. If `total_rows` is not divisible by `num_procs` (the usual case), data would need to be truncated or padded. `MPI_Scatterv` accepts individual counts and displacements for each rank, guaranteeing that **no data is lost and no rank is imbalanced by more than 1 row**.

**Why not point-to-point sends (`MPI_Send/Recv`)?** A loop of `MPI_Send` is sequential: Rank 0 sends to Rank 1, waits, sends to Rank 2, waits... With P processes, distribution time is O(P). `MPI_Scatterv` is a collective operation that the MPI implementation optimizes internally using communication trees (binomial or binary tree), reducing latency to O(log P).

#### 4.1.2. Global Statistics Reduction: `MPI_Allreduce`

Partial statistics from each node are combined with `MPI_Allreduce` instead of `MPI_Reduce` + `MPI_Bcast`.

**Why is it faster than Reduce + Bcast?**

- `MPI_Reduce` + `MPI_Bcast` = 2 collective operations, each with O(log P) latency.
- `MPI_Allreduce` = 1 operation that internally fuses both phases (reduce-scatter + allgather) with half the messages.
- Modern MPI implementations (OpenMPI, MPICH) employ the *recursive doubling* or *ring reduce* algorithm for `Allreduce`, which is asymptotically optimal in bandwidth.

**Why `MPI_Allreduce` and not just `MPI_Reduce` on Rank 0?** Because both the statistics and the centroids are needed on **all** ranks simultaneously for the next iteration. If only Rank 0 had the results, we would need an additional `MPI_Bcast`, doubling the communication.

#### 4.1.3. Data Redistribution in K-Means: `MPI_Alltoallv`

When a point changes clusters and the cluster belongs to a different rank, the point is migrated using `MPI_Alltoallv`.

**Why not point-to-point sends?**

With manual `MPI_Isend`/`MPI_Irecv`, each rank needs P sends + P receives, two nested loops, and synchronization logic with `MPI_Waitall`. This generates:
- P² total messages in the system.
- Complex code prone to deadlocks.
- Inability for the MPI library to optimize the communication pattern.

`MPI_Alltoallv` expresses the exact semantics ("everyone sends to everyone with variable amounts") in a single call, allowing the MPI implementation to:
1. Aggregate messages to the same destination.
2. Use pipelined communication or direct RDMA when available.
3. Avoid network congestion through intelligent scheduling.

**Why redistribute data and not just centroids?** Redistribution guarantees **data locality**: each rank only has the points of its assigned clusters. This has two benefits:
1. **Centroid recomputation is completely local** — no communication is needed to accumulate partial sums per cluster (only the final `MPI_Allreduce` for the global mean).
2. **Better cache balance** — each rank works with a more coherent subset of data in each iteration.

#### 4.1.4. Convergence with `MPI_Allreduce` on Displacements

The convergence check uses `MPI_Allreduce` with `MPI_SUM` on the number of points that changed clusters.

**Why a 5% threshold?** This threshold avoids unnecessary iterations when the algorithm is "practically converged." Without a threshold (exact convergence with 0 changes), K-Means can execute dozens of additional iterations moving 2-3 boundary points between clusters, where each iteration involves a complete data redistribution with `MPI_Alltoallv`.

### 4.2. Intra-Node Level: OpenMP

#### 4.2.1. Statistics: Native Array Reductions (`reduction`)

In `stats.cpp`, the statistics computation uses OpenMP reduction clauses on arrays:

```cpp
#pragma omp parallel for reduction(+: sum_ptr[:num_cols], sum_sq_ptr[:num_cols]) \
    reduction(min: min_ptr[:num_cols]) reduction(max: max_ptr[:num_cols])
```

**Why is it faster than alternatives?**

| Alternative | Problem | Cost |
|---|---|---|
| `#pragma omp critical` | Each thread waits its turn to update → total serialization | O(T × N) where T = threads |
| `#pragma omp atomic` per variable | Only works with scalar operations, not with arrays | Not applicable to min/max |
| Private buffers + manual merge | Works, but requires manual code and a `critical` section at the end | O(T × cols) for merge |
| **Native reduction** | **The OpenMP runtime manages buffers internally, merges with trees** | **O(log T × cols)** |

Native reduction is optimal because:
1. **Eliminates contention**: Each thread operates on its own copy of the array without locks.
2. **Tree merge**: The final combination is performed in O(log T) steps instead of T sequential steps.
3. **Cache locality**: Private copies fit in the L1/L2 cache of each core, avoiding inter-core coherence traffic (*cache bouncing*).

**Example:** With 8 threads on a 10M rows × 10 columns dataset:
- `critical`: 8 threads compete for a lock → ~8× slower than sequential.
- Native reduction: 8 threads work independently, merge in 3 steps (log₂ 8) → ~8× speedup.

#### 4.2.2. K-Means: Per-Thread Private Buffers with `critical` Merge

In `kmeans.cpp`, the accumulation of sums and counts per cluster uses **per-thread private buffers** with a merge in a `critical` section:

```cpp
#pragma omp parallel
{
    std::vector<double>   thread_sums(K * cols, 0.0);
    std::vector<uint32_t> thread_counts(K, 0);

    #pragma omp for nowait
    for (uint32_t r = 0; r < num_rows; ++r) { /* accumulate in thread_sums/counts */ }

    #pragma omp critical
    { /* merge thread_sums → local_sums */ }
}
```

**Why not use `reduction` like in stats?**

The OpenMP `reduction(+:...)` clause works with fixed-size arrays known at compile time or with pointers to simple arrays. However, here the accumulation is into a 2D matrix (K × cols) indexed by `cluster_id`, which varies dynamically. Per-thread private buffers offer total flexibility for this pattern.

**Why not use `atomic`?**

If K = 4 and cols = 10, each loop iteration updates 11 different positions (10 sums + 1 count). With `atomic`:
- Each atomic operation costs 5-20 cycles due to the memory barrier.
- With multiple threads updating the same clusters → *cache line bouncing* between cores (constant invalidations of shared cache lines).
- Total cost: 11 × 5-20 cycles × N rows.

With private buffers:
- Zero contention during the loop — each thread writes to its own memory.
- A single `critical` merge at the end with T steps (T = number of threads), which is negligible compared to the main loop.

**Why `nowait`?**

The `nowait` directive eliminates the implicit barrier at the end of `omp for`. This allows threads that finish early to enter the `critical` section immediately without waiting for the slower ones, reducing idle time.

#### 4.2.3. Displacements with `omp atomic`

In the K-Means assignment step, the displacement count (how many points changed clusters) uses a per-thread private variable + merge with `atomic` scheme:

```cpp
#pragma omp parallel
{
    uint64_t thread_displacement = 0;
    #pragma omp for nowait
    for (...) { if (changed) thread_displacement++; }
    #pragma omp atomic
    local_displacements += thread_displacement;
}
```

**Why is it faster than a direct `atomic` per iteration?**

- Direct `atomic`: N atomic operations (one per row), each with a memory barrier.
- Private accumulator + 1 `atomic`: N local increments (0 synchronization cost) + T atomic operations at the end (T << N).

**Why not `reduction(+:local_displacements)`?**

It would work equally well. The manual pattern was chosen for consistency with the rest of the function (private buffers + merge), and to avoid the implicit barrier of `parallel for reduction`, since `nowait` is needed for the subsequent `critical`.

---

## 5. Compilation Flags

```cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native -DNDEBUG")
```

| Flag | Purpose | Performance Impact |
|---|---|---|
| `-O3` | Maximum optimization level: aggressive inlining, loop unrolling, dead code elimination, automatic vectorization. | 5-15× faster than `-O0` |
| `-march=native` | Generates AVX/AVX2/FMA instructions specific to the CPU where it is compiled. | 2-4× in numerical loops via SIMD vectorization |
| `-DNDEBUG` | Disables `assert()` to eliminate checks in production. | Eliminates verification overhead in hot loops |
| `-Wall -Wextra` | Compiler warnings to maintain clean code. | No runtime impact |

**Why not `-Ofast`?** `-Ofast` includes `-ffast-math`, which allows reordering floating-point operations (breaks associativity). This can change the numerical results of reductions and variance. `-O3` is the maximum optimization level that **preserves IEEE 754 semantics**.

---

## 6. Advantages of the Complete Design: End-to-End Data Flow

```
┌───────────────────────────────────────────────────────── RANK 0 ─────────────┐
│  Disk ──── file.read() ──── vector<float> contiguous ──── MPI_Scatterv ──→   │
└──────────────────────────────────────────────────────────────────────────────┘
                          ↓                                    ↓
                ┌─── RANK 0 ───┐  ┌─── RANK 1 ───┐  ┌─── RANK 2 ───┐
                │  OpenMP par. │  │  OpenMP par. │  │  OpenMP par. │
                │  Local stats │  │  Local stats │  │  Local stats │
                └──── ↓ ───────┘  └──── ↓ ───────┘  └──── ↓ ───────┘
                         └─── MPI_Allreduce(SUM/MIN/MAX) ───┘
                                         ↓
                              Global stats on all ranks
                                         ↓
                ┌─── RANK 0 ───┐  ┌─── RANK 1 ───┐  ┌─── RANK 2 ───┐
                │  Assign pts  │  │  Assign pts  │  │  Assign pts  │
                │  (OpenMP)    │  │  (OpenMP)    │  │  (OpenMP)    │
                └──── ↓ ───────┘  └──── ↓ ───────┘  └──── ↓ ───────┘
                │  Converged?  │ ←── MPI_Allreduce(SUM displacements)
                └──── ↓ ───────┘
                │  Redistrib.  │ ←── MPI_Alltoallv (data + assignments)
                └──── ↓ ───────┘
                │  Update cent │ ←── OpenMP par. + MPI_Allreduce
                └──────────────┘
                     ↺ iterate until convergence
```

Each component is designed so that data flows contiguously, without unnecessary intermediate copies, minimizing memory and network latency in every phase.

---

## 7. Summary: Decision Comparison

| Design Decision | Rejected Alternative | Estimated Improvement Factor |
|---|---|---|
| Contiguous vector (SoA) | Array of Structures (AoS) | 3-10× (cache locality) |
| `float` for data | `double` for everything | 2× (memory/network bandwidth) |
| Native binary for I/O | CSV/text | 5-20× (no parsing) |
| `MPI_Scatterv` | `MPI_Send` loop | O(log P) vs O(P) |
| `MPI_Allreduce` | `Reduce` + `Bcast` | ~2× fewer messages |
| `MPI_Alltoallv` | Manual `Isend`/`Irecv` | Internal MPI optimization + robustness |
| Native OpenMP reduction | `critical` on every iteration | O(log T) vs O(T×N) |
| Private buffers + merge | `atomic` per iteration | 0 contention in hot loop |
| `-O3 -march=native` | Default compilation | 5-15× (SIMD vectorization) |
| Data redistribution | Centroids-only broadcast | Improved cache locality |
