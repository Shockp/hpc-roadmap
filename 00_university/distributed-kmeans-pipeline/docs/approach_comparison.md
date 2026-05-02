# Approach Comparison — Code vs. Alternatives

This document walks through **every code fragment in the project**, compares it with alternative approaches, and explains why the chosen implementation is superior in performance.

---

## 1. Data Structures (`include/models/`)

### 1.1. `Dataset` — Flat Contiguous Vector

#### Our Implementation

```cpp
struct Dataset {
  uint32_t n_rows;
  uint32_t n_cols;
  std::vector<float> data;  // row-major contiguous

  inline const float *GetRowPtr(uint32_t row) const {
    return &data[row * n_cols];
  }
};
```

#### Alternative A: Array of Structures with `std::vector<Point>`

```cpp
struct Point {
  std::vector<float> features;
};

struct DatasetAoS {
  std::vector<Point> points;
};
```

**Why is it worse?**

1. **Double indirection:** `points[i].features[j]` requires two pointers: one to the `Point`, another to the inner `vector<float>`. Each access can be a *cache miss*.
2. **Fragmented memory:** Each `Point` has its own `std::vector`, which allocates its buffer on the heap independently. With 10 million points, there are 10 million allocations scattered across RAM.
3. **Cannot send via MPI:** You cannot do `MPI_Scatterv(points.data(), ...)` because the data is not contiguous. You would need to copy everything to a flat buffer first → double memory + copy time.
4. **No SIMD vectorization:** The compiler cannot generate AVX instructions because it cannot guarantee that `features.data()` of one point is contiguous with the next.

```
Memory with AoS (fragmented):
  Heap: [Point0.features → 0xA000] [Point1.features → 0xB400] [Point2.features → 0xC100]
        scattered across the heap → constant cache misses

Memory with SoA (contiguous):
  Heap: [x0,y0,z0, x1,y1,z1, x2,y2,z2, ...]  ← single block → optimal prefetch
```

#### Alternative B: `float**` (pointer to pointers)

```cpp
float** data = new float*[n_rows];
for (int i = 0; i < n_rows; i++)
    data[i] = new float[n_cols];
```

**Why is it worse?**

- Same fragmentation as AoS: each row is an independent allocation.
- `n_rows` calls to `new` → slow initialization.
- Memory leaks if `delete[]` is not called for each row.
- Pattern impossible for the compiler to optimize.

#### Alternative C: `std::vector<std::vector<float>>`

```cpp
std::vector<std::vector<float>> data(n_rows, std::vector<float>(n_cols));
```

**Why is it worse?**

- Each inner row is a separate `vector` with its own heap allocation → same fragmentation.
- Metadata overhead: each `vector` stores a pointer, size, and capacity (24 extra bytes per row).
- With 10M rows: 240 MB in inner vector metadata alone.

---

### 1.2. `Centroids` — Same Contiguous Philosophy

#### Our Implementation

```cpp
struct Centroids {
  uint32_t num_clusters;
  uint32_t num_cols;
  std::vector<float> data;       // K × cols contiguous
  std::vector<uint32_t> counts;  // size of each cluster
};
```

#### Alternative: `std::map<int, std::vector<float>>`

```cpp
std::map<int, std::vector<float>> centroids;
// centroids[0] = {1.2, 3.4, 5.6};
// centroids[1] = {7.8, 9.0, 1.1};
```

**Why is it worse?**

- `std::map` stores nodes in a red-black tree: each centroid is in a separate heap node.
- O(log K) access per lookup versus O(1) with direct indexing.
- Cannot send via MPI: no contiguous buffer.
- With K = 4 and cols = 10, the difference is small, but the pattern is necessary to scale to K = 1000+.

---

### 1.3. `Column_stats` — Flat Struct

#### Our Implementation

```cpp
struct Column_stats {
  float min = 0.0f;
  float max = 0.0f;
  float mean = 0.0f;
  float variance = 0.0f;
};
```

#### Alternative: `std::unordered_map<std::string, float>`

```cpp
std::unordered_map<std::string, float> stats;
stats["min"] = ...;
stats["max"] = ...;
```

**Why is it worse?**

- Hash map with strings: each lookup requires hash + string comparison.
- ~100× slower than accessing a struct field directly (1 instruction vs. hash + potential collision).
- Massive memory overhead: each string has its own dynamic allocation.

---

## 2. I/O (`io_utils.cpp`)

### 2.1. Direct Binary Reading

#### Our Implementation

```cpp
std::optional<Dataset> ReadBinaryFile(const std::filesystem::path &filepath) {
  Dataset dataset;
  std::ifstream file(filepath, std::ios::binary);

  file.read(reinterpret_cast<char *>(&dataset.n_rows), sizeof(uint32_t));
  file.read(reinterpret_cast<char *>(&dataset.n_cols), sizeof(uint32_t));

  dataset.data.resize(total_elements);
  file.read(reinterpret_cast<char *>(dataset.data.data()),
            total_elements * sizeof(float));
  return dataset;
}
```

#### Alternative A: Line-by-Line CSV Reading

```cpp
Dataset ReadCSV(const std::string &filepath) {
  Dataset dataset;
  std::ifstream file(filepath);
  std::string line;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    float val;
    while (ss >> val) {
      dataset.data.push_back(val);  // ← constant realloc
      if (ss.peek() == ',') ss.ignore();
    }
    dataset.n_rows++;
  }
  return dataset;
}
```

**Why is it worse?**

| Aspect | Binary | CSV |
|---|---|---|
| Read operations | 3 `read()` | N×M `>>` + `getline` |
| Conversions | None | N×M `string → float` |
| `push_back` with realloc | 0 | Hundreds of reallocations |
| Size on disk (10M×10) | ~400 MB | ~900 MB |
| Estimated time | ~0.5 s | ~5-10 s |

The critical point is `push_back`: without knowing the total size in advance, the vector reallocates (doubles capacity) multiple times, copying all data each time. Our version calls `resize(total_elements)` a single time.

#### Alternative B: `mmap` (memory mapping)

```cpp
int fd = open("dataset.bin", O_RDONLY);
float *data = (float*)mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
```

**When would it be better?** With datasets larger than RAM, `mmap` allows on-demand access without loading everything into memory. However:

- **Our case:** The dataset fits in RAM and we need it entirely in a `vector` for MPI. `mmap` would add on-demand *page faults* and is not compatible with `MPI_Scatterv` (which needs a contiguous buffer already resident in memory).
- **Conclusion:** For in-memory datasets like ours, direct `file.read()` is more predictable and faster.

### 2.2. `std::optional` for Error Handling

#### Our Implementation

```cpp
std::optional<Dataset> ReadBinaryFile(...);
// Usage:
auto result = ReadBinaryFile("dataset.bin");
if (result.has_value()) { /* ok */ }
```

#### Alternative A: Exceptions

```cpp
Dataset ReadBinaryFile(...) {
  if (!file.is_open()) throw std::runtime_error("...");
  // ...
  return dataset;
}
```

**Why is it worse in HPC?**

- C++ exceptions have zero cost on the *happy path* with modern compilers, but *unwinding* on error is extremely expensive (100-1000 cycles).
- More importantly: `-fno-exceptions` is a common flag in HPC to reduce binary size. `std::optional` works without exceptions.
- The compiler generates cleanup tables (`.gcc_except_table`) that increase binary size and can affect code cache locality.

#### Alternative B: Error Code (C-style)

```cpp
int ReadBinaryFile(const char *path, Dataset *out_dataset);
// Usage:
Dataset ds;
if (ReadBinaryFile("data.bin", &ds) != 0) { /* error */ }
```

**Why is it worse?**

- The output parameter forces the caller to declare the variable beforehand → cannot use `auto`.
- No semantic clarity: is `0` success or error? Depends on the convention.
- `std::optional` is self-explanatory and the compiler can optimize Return Value Optimization (RVO) just as well as a direct return.

---

## 3. Data Generator (`tools/file_generator.cpp`)

### 3.1. Mersenne Twister with Fixed Seed

#### Our Implementation

```cpp
std::mt19937 rng(42);  // fixed seed
std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
for (uint64_t i = 0; i < total_elements; ++i)
    data[i] = dist(rng);
```

#### Alternative A: C's `rand()`

```cpp
srand(42);
for (uint64_t i = 0; i < total_elements; ++i)
    data[i] = -100.0f + (rand() / (float)RAND_MAX) * 200.0f;
```

**Why is it worse?**

- `rand()` uses a linear congruential generator with a short period (2³¹). With 100M elements, patterns repeat.
- `RAND_MAX` is only 2³¹-1 → only 2.1 billion possible values between -100 and 100. Granular distribution.
- `rand()` is not thread-safe: if we parallelize the generation, there are race conditions.
- `mt19937` has a period of 2¹⁹⁹³⁷-1 and a statistically verified uniform distribution.

#### Alternative B: `/dev/urandom`

```cpp
int fd = open("/dev/urandom", O_RDONLY);
read(fd, data.data(), total_elements * sizeof(float));
```

**Why is it worse?**

- Not reproducible: each run generates different data → impossible to compare benchmarks.
- Values do not follow a uniform distribution within a specific range; they are random bytes reinterpreted as float → includes NaN, infinities, and denormals.
- The fixed seed (`42`) guarantees that all tests and benchmarks are **deterministic**.

### 3.2. Block Binary Writing

#### Our Implementation

```cpp
file.write(reinterpret_cast<const char *>(&num_rows), sizeof(uint32_t));
file.write(reinterpret_cast<const char *>(&num_cols), sizeof(uint32_t));
file.write(reinterpret_cast<const char *>(data.data()),
           total_elements * sizeof(float));
```

#### Alternative: Point-by-Point CSV Writing

```cpp
for (uint32_t r = 0; r < num_rows; r++) {
    for (uint32_t c = 0; c < num_cols; c++) {
        file << data[r * num_cols + c];
        if (c < num_cols - 1) file << ",";
    }
    file << "\n";
}
```

**Why is it worse?**

- N×M `operator<<` operations with `float → string` conversion (each one internally requires `sprintf`).
- The C++ stream flushes its internal buffer frequently with so many small writes.
- Our `file.write()` sends a single contiguous block to the kernel in a single syscall.

---

## 4. Initial Data Distribution (`main.cpp`)

### 4.1. `MPI_Bcast` for Dimensions

#### Our Implementation

```cpp
uint32_t dimensions[2] = {0, 0};
if (rank == 0) {
    dimensions[0] = global_dataset.n_rows;
    dimensions[1] = global_dataset.n_cols;
}
MPI_Bcast(dimensions, 2, MPI_UINT32_T, 0, MPI_COMM_WORLD);
```

#### Alternative: `MPI_Send` Loop from Rank 0

```cpp
if (rank == 0) {
    for (int i = 1; i < num_procs; i++)
        MPI_Send(dimensions, 2, MPI_UINT32_T, i, 0, MPI_COMM_WORLD);
} else {
    MPI_Recv(dimensions, 2, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
```

**Why is it worse?**

- **P-1 sequential messages** from Rank 0: total latency = (P-1) × network_latency.
- `MPI_Bcast` uses a **binomial tree**: Rank 0 sends to 1, then 0→2 and 1→3 simultaneously, etc. Latency = O(log P).
- With P = 64 processes: Loop Send = 63 steps. Bcast = 6 steps. **10× faster**.

### 4.2. `MPI_Scatterv` with Remainder Balancing

#### Our Implementation

```cpp
int local_rows = total_rows / num_procs;
int remainder = total_rows % num_procs;
if (rank < remainder) local_rows++;

// Build sendcounts/displacements arrays for Scatterv
for (int i = 0; i < num_procs; ++i) {
    int rows_for_proc = total_rows / num_procs + (i < remainder ? 1 : 0);
    sendcounts[i] = rows_for_proc * num_cols;
    displacements[i] = current_displacement;
    current_displacement += sendcounts[i];
}
MPI_Scatterv(global_dataset.data.data(), sendcounts.data(), ...);
```

#### Alternative A: `MPI_Scatter` with Truncation

```cpp
int rows_per_proc = total_rows / num_procs;  // discards the remainder
MPI_Scatter(data, rows_per_proc * num_cols, MPI_FLOAT, ...);
```

**Why is it worse?**

- If `total_rows = 1003` and `num_procs = 4`, each process receives 250 rows → **3 rows are lost**.
- In a real dataset, losing data is unacceptable. Furthermore, the imbalance grows with more processes.

#### Alternative B: Padding with Empty Rows

```cpp
// Pad until total_rows is divisible by num_procs
while (total_rows % num_procs != 0) {
    global_dataset.data.insert(global_dataset.data.end(), num_cols, 0.0f);
    total_rows++;
}
MPI_Scatter(...);
```

**Why is it worse?**

- The padding rows (zeros) **corrupt statistics**: the minimum will always be 0, the mean shifts.
- The padding rows **affect K-Means**: centroids become biased toward the origin (0,0,...,0).
- `MPI_Scatterv` avoids both problems by distributing exactly the real data.

### 4.3. Computing `global_offset`

#### Our Implementation

```cpp
uint64_t global_offset = 0;
for (int i = 0; i < rank; ++i)
    global_offset += (total_rows / num_procs) + (i < remainder ? 1 : 0);
```

#### Alternative: Closed-Form Formula

```cpp
uint64_t global_offset = rank * (total_rows / num_procs)
                       + std::min(rank, remainder);
```

**Is the closed-form formula better?** Technically yes — O(1) vs O(P). But with P < 1000 processes, the difference is nanoseconds for an operation that occurs only once. The loop version is more readable and less prone to overflow errors with large integers.

---

## 5. Statistics (`stats.cpp`)

### 5.1. OpenMP Reduction with Arrays

#### Our Implementation

```cpp
double *sum_ptr = local_sum.data();
float *min_ptr = local_min.data();

#pragma omp parallel for \
    reduction(+: sum_ptr[:num_cols], sum_sq_ptr[:num_cols]) \
    reduction(min: min_ptr[:num_cols]) \
    reduction(max: max_ptr[:num_cols])
for (uint32_t r = 0; r < num_rows; ++r) {
    const float *row_ptr = local_data.GetRowPtr(r);
    for (uint32_t c = 0; c < num_cols; ++c) {
        sum_ptr[c] += row_ptr[c];
        sum_sq_ptr[c] += (double)row_ptr[c] * row_ptr[c];
        min_ptr[c] = std::min(min_ptr[c], row_ptr[c]);
        max_ptr[c] = std::max(max_ptr[c], row_ptr[c]);
    }
}
```

#### Alternative A: `#pragma omp critical` on Every Iteration

```cpp
#pragma omp parallel for
for (uint32_t r = 0; r < num_rows; ++r) {
    const float *row_ptr = local_data.GetRowPtr(r);
    #pragma omp critical
    {
        for (uint32_t c = 0; c < num_cols; ++c) {
            sum[c] += row_ptr[c];
            min_val[c] = std::min(min_val[c], row_ptr[c]);
            // ...
        }
    }
}
```

**Why is it worse?**

- `critical` completely serializes the block: only one thread executes it at a time.
- With 8 threads, 7 are always waiting → **worse than sequential** due to lock overhead.
- Performance: `T_critical ≈ T_sequential × (1 + lock_overhead)` → slower than without OpenMP.

#### Alternative B: `#pragma omp atomic` for Each Variable

```cpp
#pragma omp parallel for
for (uint32_t r = 0; r < num_rows; ++r) {
    const float *row_ptr = local_data.GetRowPtr(r);
    for (uint32_t c = 0; c < num_cols; ++c) {
        #pragma omp atomic
        sum[c] += row_ptr[c];
        // atomic does NOT support min/max → not applicable
    }
}
```

**Why is it worse?**

- `atomic` does not support `min`/`max` operations → we would need a `critical` section anyway.
- Even for the sum: N×M atomic operations, each with a memory barrier (5-20 extra cycles).
- *Cache line bouncing*: multiple cores try to write to `sum[0]` simultaneously → the cache line bounces between L1 caches of different cores.

#### Alternative C: Manual Private Buffers (as in kmeans.cpp)

```cpp
#pragma omp parallel
{
    std::vector<double> my_sum(num_cols, 0.0);
    std::vector<float> my_min(num_cols, FLT_MAX);

    #pragma omp for nowait
    for (uint32_t r = 0; r < num_rows; ++r) { /* accumulate in my_sum, my_min */ }

    #pragma omp critical
    {
        for (uint32_t c = 0; c < num_cols; ++c) {
            sum[c] += my_sum[c];
            min_val[c] = std::min(min_val[c], my_min[c]);
        }
    }
}
```

**Is it worse?** It is functional and performs well, but:

- Requires manually writing and maintaining the merge buffers.
- The `critical` section at the end is O(T) — it serializes thread by thread.
- The native OpenMP reduction merges in O(log T) steps using an internal binary tree.
- With T = 16 threads: critical = 16 merge steps, native reduction = 4 steps. **4× faster merge**.

### 5.2. `double` Accumulators for `float` Sums

#### Our Implementation

```cpp
std::vector<double> local_sum(num_cols, 0.0);      // double
std::vector<double> local_sum_sq(num_cols, 0.0);    // double
sum_ptr[c] += val;                                   // float → double implicit
sum_sq_ptr[c] += static_cast<double>(val) * val;     // double × float
```

#### Alternative: Everything in `float`

```cpp
std::vector<float> local_sum(num_cols, 0.0f);
local_sum[c] += val;  // float + float
```

**Why is it worse numerically?**

`float` has ~7 digits of precision. When summing 10 million values:

```
Example: sum 10,000,000 values of ~50.0
Real sum:     500,000,000.0
Float sum:    499,999,872.0   ← error of 128 due to precision loss
Double sum:   500,000,000.0   ← exact up to 15 digits
```

**Catastrophic cancellation** is even worse for variance: `Var = E[X²] - (E[X])²` subtracts two large numbers to obtain a small one. With `float`, the result can be **negative** (mathematically impossible), which is why we include the clamp to zero.

### 5.3. Variance with Computational Formula

#### Our Implementation

```cpp
double mean = global_sum[c] / (double)global_rows;
double mean_of_squares = global_sum_sq[c] / (double)global_rows;
double variance = mean_of_squares - (mean * mean);  // E[X²] - (E[X])²
if (variance < 0.0) variance = 0.0;                  // clamp
```

#### Alternative: Two-Pass Formula

```cpp
// Pass 1: compute mean
double mean = 0;
for (auto x : data) mean += x;
mean /= N;

// Pass 2: compute variance
double var = 0;
for (auto x : data) var += (x - mean) * (x - mean);
var /= N;
```

**Is it better numerically?** Yes, the two-pass formula is more numerically stable. **Is it worse in performance?** Much worse in our distributed context:

- Requires **two complete passes** over the data → double memory bandwidth.
- In MPI, the first pass needs an `MPI_Allreduce` to obtain the global mean **before** the second pass → two rounds of communication.
- Our one-pass formula computes `sum` and `sum_sq` simultaneously in the same loop → **1 pass, 1 Allreduce**.

The clamp `if (variance < 0) variance = 0` compensates for the lower stability at zero cost.

### 5.4. `MPI_Allreduce` to Combine Statistics

#### Our Implementation

```cpp
MPI_Allreduce(local_sum.data(), global_sum.data(), num_cols,
              MPI_DOUBLE, MPI_SUM, comm);
MPI_Allreduce(local_min.data(), global_min.data(), num_cols,
              MPI_FLOAT, MPI_MIN, comm);
```

#### Alternative A: `MPI_Reduce` to Rank 0 + `MPI_Bcast`

```cpp
MPI_Reduce(local_sum.data(), global_sum.data(), num_cols,
           MPI_DOUBLE, MPI_SUM, 0, comm);
MPI_Bcast(global_sum.data(), num_cols, MPI_DOUBLE, 0, comm);
```

**Why is it worse?**

- 2 collective operations = 2 × O(log P) latency.
- `MPI_Allreduce` fuses both phases internally with algorithms like *recursive doubling* or *ring allreduce*.
- MPI implementations (like OpenMPI) detect the message size and automatically choose the most efficient algorithm for `Allreduce`.

#### Alternative B: Manual Reduction with `MPI_Send/Recv` in a Tree

```cpp
// Manually implement a binary tree...
int partner = rank ^ (1 << step);
if (rank < partner) {
    MPI_Recv(buffer, ...);
    for (int c = 0; c < num_cols; c++) sum[c] += buffer[c];
} else {
    MPI_Send(sum, ...);
}
```

**Why is it worse?**

- Reimplementing the reduction tree is error-prone (powers of 2, odd ranks, etc.).
- `MPI_Allreduce` has been optimized for decades with adaptive algorithms that choose the strategy based on message size, number of processes, and network topology.
- Maintaining manual communication code is a debugging nightmare.

---

## 6. K-Means — Initialization (`kmeans.cpp: InitializeCentroids`)

### 6.1. Uniform Partitioning by Global Index

#### Our Implementation

```cpp
uint64_t rows_per_cluster = total_rows / num_clusters;
uint32_t cluster_id = std::min(
    static_cast<uint32_t>(global_index / rows_per_cluster),
    num_clusters - 1);
```

#### Alternative A: K-Means++ (Probabilistic Selection)

```cpp
// Choose first centroid randomly
// For each point, compute distance to the nearest centroid
// Choose next centroid with probability proportional to distance²
```

**Is K-Means++ better?** It produces more representative initial centroids (converges in fewer iterations). **Why don't we use it?**

- K-Means++ is **inherently sequential**: each centroid depends on the previous ones.
- In an MPI environment, each step requires an `MPI_Allreduce` for the chosen centroid + `MPI_Bcast` → K rounds of communication.
- Uniform partitioning is O(1) per point, fully parallel, and requires no communication for assignment (only for the final sums).

#### Alternative B: Random Centroids

```cpp
std::mt19937 rng(42);
for (int k = 0; k < num_clusters; k++)
    for (int c = 0; c < num_cols; c++)
        centroids[k][c] = dist(rng);
```

**Why is it worse?**

- Random centroids can be very far from any real data → first iterations wasted.
- Risk of empty clusters: if a centroid is placed where there is no data, it remains empty indefinitely.
- Uniform partitioning guarantees that **each centroid starts as the mean of real data** → faster convergence.

### 6.2. Per-Thread Private Buffers with `critical` Merge

#### Our Implementation

```cpp
#pragma omp parallel
{
    std::vector<double> thread_sums(K * cols, 0.0);
    std::vector<uint32_t> thread_counts(K, 0);

    #pragma omp for nowait
    for (uint32_t r = 0; r < num_rows; ++r) {
        uint32_t cluster_id = ...;
        thread_counts[cluster_id]++;
        for (uint32_t c = 0; c < num_cols; ++c)
            thread_sums[cluster_id * num_cols + c] += row_ptr[c];
    }

    #pragma omp critical
    {
        for (uint32_t k = 0; k < K; ++k) {
            local_counts[k] += thread_counts[k];
            for (uint32_t c = 0; c < num_cols; ++c)
                local_sums[k * num_cols + c] += thread_sums[k * num_cols + c];
        }
    }
}
```

#### Alternative: `atomic` on Every Accumulation

```cpp
#pragma omp parallel for
for (uint32_t r = 0; r < num_rows; ++r) {
    uint32_t cluster_id = ...;
    #pragma omp atomic
    local_counts[cluster_id]++;
    for (uint32_t c = 0; c < num_cols; ++c) {
        #pragma omp atomic
        local_sums[cluster_id * num_cols + c] += row_ptr[c];
    }
}
```

**Why is it worse?**

For each row, **1 + num_cols** atomic operations are executed. With 10M rows and 10 columns:
- Total atomics: 10M × 11 = **110 million atomic operations**.
- Each atomic: memory barrier + possible cache invalidation = 5-20 cycles.
- Estimated cost: 110M × 10 cycles = **1.1 seconds** (at 1 GHz) in synchronization alone.

With private buffers:
- 0 atomic operations during the loop.
- In the merge: T threads × K clusters × cols = 8 × 4 × 10 = **320 additions** → negligible.

---

## 7. K-Means — Main Loop (`kmeans.cpp: RunKMeans`)

### 7.1. Assignment: Squared Euclidean Distance

#### Our Implementation

```cpp
float dist_sq = 0.0f;
for (uint32_t c = 0; c < num_cols; ++c) {
    float diff = row_ptr[c] - centroid_ptr[c];
    dist_sq += diff * diff;
}
if (dist_sq < min_dist_sq) { best_cluster = k; ... }
```

#### Alternative A: Euclidean Distance with `sqrt`

```cpp
float dist = 0.0f;
for (uint32_t c = 0; c < num_cols; ++c) {
    float diff = row_ptr[c] - centroid_ptr[c];
    dist += diff * diff;
}
dist = std::sqrt(dist);  // ← unnecessary
```

**Why is it worse?**

- `sqrt` is one of the most expensive processor operations (~15-20 cycles versus 1 cycle for multiplication).
- Since we only compare distances (`dist < min_dist`), and `sqrt` is a monotonic function, `dist² < min_dist²` produces the same result without the square root.
- With 10M rows × K = 4 centroids: **40 million `sqrt` calls eliminated**.

#### Alternative B: Manhattan Distance

```cpp
float dist = 0.0f;
for (uint32_t c = 0; c < num_cols; ++c)
    dist += std::abs(row_ptr[c] - centroid_ptr[c]);
```

**Is it faster?** Yes (`abs` is cheaper than `x*x`). **Is it correct?** Not for standard K-Means. K-Means minimizes intra-cluster variance, which is defined with L2 distance (Euclidean). Using Manhattan (L1) converges toward *medians* instead of *means*, which is a different algorithm (K-Medians).

### 7.2. Convergence: 5% Threshold

#### Our Implementation

```cpp
uint64_t global_displacement = 0;
MPI_Allreduce(&local_displacements, &global_displacement, 1,
              MPI_UINT64_T, MPI_SUM, comm);

double displacement_ratio = (double)global_displacement / (double)total_rows;
if (displacement_ratio < 0.05) { converged = true; break; }
```

#### Alternative A: Exact Convergence (0 changes)

```cpp
if (global_displacement == 0) { converged = true; break; }
```

**Why is it worse?**

- K-Means can oscillate with 2-3 boundary points that switch clusters every iteration indefinitely.
- Each extra iteration involves a full `MPI_Alltoallv` (massive data redistribution).
- The 5% threshold stops the algorithm **when 95% of points are already stable**, potentially saving dozens of expensive iterations.

#### Alternative B: Convergence by Centroid Shift

```cpp
double centroid_shift = 0;
for (int k = 0; k < K; k++)
    for (int c = 0; c < cols; c++)
        centroid_shift += pow(new_centroid[k][c] - old_centroid[k][c], 2);
if (centroid_shift < epsilon) break;
```

**Is it better?** It is more finely tunable, but requires storing the previous centroids and computing the difference. The displacement ratio is more intuitive ("% of points that changed") and requires no additional memory.

### 7.3. Data Redistribution with `MPI_Alltoallv`

#### Our Implementation

```cpp
// 1. Partition rows by destination rank
for (uint32_t r = 0; r < current_num_rows; ++r) {
    int owner_rank = get_owner_rank(local_assignments[r]);
    if (owner_rank == rank) {
        next_local_data.insert(...);  // stays here
    } else {
        send_data_buffers[owner_rank].insert(...);  // send
    }
}

// 2. Notify how many points each rank sends to each other
MPI_Alltoall(send_point_counts, 1, MPI_INT, recv_point_counts, 1, MPI_INT, comm);

// 3. Exchange data
MPI_Alltoallv(flat_send_data, send_data_counts, send_data_displacements, MPI_FLOAT,
              flat_recv_data, recv_data_counts, recv_data_displacements, MPI_FLOAT, comm);

// 4. Merge: received data + local data
next_local_data.insert(next_local_data.end(), flat_recv_data.begin(), ...);
```

#### Alternative A: Point-to-Point `MPI_Isend`/`MPI_Irecv`

```cpp
std::vector<MPI_Request> requests;
for (int dest = 0; dest < num_procs; dest++) {
    if (dest != rank && send_counts[dest] > 0) {
        MPI_Isend(send_buf[dest].data(), send_counts[dest], MPI_FLOAT,
                  dest, 0, comm, &requests.back());
    }
}
for (int src = 0; src < num_procs; src++) {
    if (src != rank) {
        MPI_Irecv(recv_buf[src].data(), recv_counts[src], MPI_FLOAT,
                  src, 0, comm, &requests.back());
    }
}
MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
```

**Why is it worse?**

| Aspect | `MPI_Alltoallv` | Manual `Isend`/`Irecv` |
|---|---|---|
| Lines of code | ~10 | ~30+ |
| Determining recv_counts | Automatic `MPI_Alltoall` | Must be exchanged manually |
| Internal optimization | MPI chooses algorithm (pairwise, Bruck, linear) | Direct send only |
| Deadlock risk | None | Possible if Send/Recv order is incorrect |
| Hardware integration | Can use RDMA, NIC offload | Depends on implementation |

#### Alternative B: No Redistribution (only centroid broadcast)

```cpp
// Without MPI_Alltoallv: each rank keeps its original data
// Only share updated centroids
MPI_Allreduce(local_sums, global_sums, K * cols, MPI_DOUBLE, MPI_SUM, comm);
MPI_Allreduce(local_counts, global_counts, K, MPI_UINT32_T, MPI_SUM, comm);
```

**Is it simpler?** Yes. **Is it worse in long-term performance?** It can be:

- Without redistribution, each rank accumulates partial sums for **all** K clusters, even though the majority of its points belong to 1-2 clusters.
- With redistribution, each rank only has points from its assigned clusters → accumulations are more efficient and local.
- However, redistribution has a fixed cost per iteration (`MPI_Alltoallv`). For small K and few iterations, the version without redistribution can be faster.

**Our choice:** Redistribution scales better with large K and massive datasets, which is the HPC use case we designed the system for.

### 7.4. Cluster Ownership: `get_owner_rank`

#### Our Implementation

```cpp
auto get_owner_rank = [&](uint32_t cluster_id) -> int {
    uint32_t base_clusters = num_clusters / num_procs;
    uint32_t remainder = num_clusters % num_procs;
    // Distribute clusters evenly: first 'remainder' ranks get 1 extra
    int rank_owner = 0;
    uint32_t limit = 0;
    for (int i = 0; i < num_procs; ++i) {
        limit += base_clusters + (i < remainder ? 1 : 0);
        if (cluster_id < limit) { rank_owner = i; break; }
    }
    return rank_owner;
};
```

#### Alternative A: Simple Round-Robin

```cpp
int get_owner_rank(uint32_t cluster_id) {
    return cluster_id % num_procs;
}
```

**Is it better?** Simpler, O(1). **Is it worse?** It can create imbalance: if K = 5 and P = 4, Rank 0 has clusters {0, 4} (2 clusters) and Rank 3 has only cluster {3}. Our version distributes 2-1-1-1, which is as balanced as possible.

#### Alternative B: Hash

```cpp
int get_owner_rank(uint32_t cluster_id) {
    return std::hash<uint32_t>{}(cluster_id) % num_procs;
}
```

**Is it worse?** The hash can produce collisions and uneven distribution with small K. Furthermore, it is not deterministic across compilers → non-reproducible results.

---

## 8. Build Configuration (`CMakeLists.txt`)

### 8.1. C++17 Without GNU Extensions

#### Our Implementation

```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
```

#### Alternative: Manual Flags

```cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
```

**Why is it worse?**

- Only works with GCC/Clang. MSVC uses `/std:c++17`.
- `CMAKE_CXX_EXTENSIONS OFF` prevents `gnu++17`, which enables non-portable extensions. With manual `-std=c++17`, CMake could default to `gnu++17`.
- `CMAKE_CXX_STANDARD_REQUIRED ON` fails configuration if the compiler does not support C++17, instead of silently degrading to C++14.

### 8.2. Modern Targets for MPI and OpenMP

#### Our Implementation

```cmake
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
```

#### Alternative: Manual Flags

```cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
include_directories(/usr/lib/openmpi/include)
link_directories(/usr/lib/openmpi/lib)
target_link_libraries(${PROJECT_NAME} mpi)
```

**Why is it worse?**

- Hardcoded paths: only works on a specific Linux distribution.
- `MPI::MPI_CXX` and `OpenMP::OpenMP_CXX` are *imported targets* that CMake configures automatically with the correct includes, linker flags, and compiler flags for **any** system.
- With modern targets, a compiler change (GCC → Intel) or MPI change (OpenMPI → MPICH) does not require modifying the CMakeLists.txt.

---

## 9. General Summary: All Decisions

| Component | Decision | Worse Alternative | Why It Is Worse |
|---|---|---|---|
| `Dataset` | Contiguous `vector<float>` | `vector<Point>` AoS | Cache misses, no SIMD, not MPI-compatible |
| `Dataset` | Contiguous `vector<float>` | `float**` pointers | Fragmentation, memory leaks |
| `Centroids` | Flat `vector<float>` | `map<int, vector>` | O(log K) access, not contiguous |
| `Column_stats` | Flat struct | `unordered_map<string>` | 100× slower per lookup |
| I/O | Direct binary | Line-by-line CSV | 10-20× slower, costly parsing |
| I/O | `std::optional` | Exceptions | Unwinding cost, incompatible with `-fno-exceptions` |
| Generator | Fixed `mt19937` | `rand()` | Short period, not thread-safe, poor distribution |
| Broadcast | `MPI_Bcast` | `MPI_Send` loop | O(log P) vs O(P) |
| Distribution | `MPI_Scatterv` | `MPI_Scatter` truncating | Loses data, imbalance |
| Stats OpenMP | Native array reduction | `critical` every row | Serializes everything: worse than sequential |
| Stats OpenMP | Native array reduction | `atomic` every variable | Does not support min/max, cache bouncing |
| Stats | One-pass formula | Two passes | Double bandwidth + 2 Allreduce vs 1 |
| Stats MPI | `MPI_Allreduce` | `Reduce` + `Bcast` | 2 collectives vs 1 fused |
| K-Means init | Uniform partition | K-Means++ | Inherently sequential in MPI |
| K-Means init | Uniform partition | Random | Centroids far from real data |
| K-Means accum. | Private buffers + critical | `atomic` every sum | 110M atomics vs ~300 additions |
| K-Means dist. | dist² without sqrt | dist with sqrt | 40M sqrt calls eliminated |
| K-Means conv. | 5% threshold | Exact convergence | Possible infinite oscillation |
| Redistribution | `MPI_Alltoallv` | `Isend`/`Irecv` | More code, no internal opt., deadlock risk |
| Owner rank | Balanced distribution | Round-robin | Imbalance when K % P ≠ 0 |
| Build | Modern CMake targets | Manual flags | Not portable, hardcoded paths |
| Build | `-O3 -march=native` | Default (`-O0`) | 5-15× without SIMD vectorization |
