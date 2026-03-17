# Comparativa de Enfoques — Código vs. Alternativas

Este documento recorre **cada fragmento de código del proyecto**, lo compara con enfoques alternativos, y explica por qué la implementación elegida es superior en rendimiento.

---

## 1. Estructuras de Datos (`include/models/`)

### 1.1. `Dataset` — Vector contiguo plano

#### Nuestra implementación

```cpp
struct Dataset {
  uint32_t n_rows;
  uint32_t n_cols;
  std::vector<float> data;  // fila-mayor contiguo

  inline const float *GetRowPtr(uint32_t row) const {
    return &data[row * n_cols];
  }
};
```

#### Alternativa A: Array of Structures con `std::vector<Point>`

```cpp
struct Point {
  std::vector<float> features;
};

struct DatasetAoS {
  std::vector<Point> points;
};
```

**¿Por qué es peor?**

1. **Doble indirección:** `points[i].features[j]` requiere dos punteros: uno al `Point`, otro al `vector<float>` interno. Cada acceso puede ser un *cache miss*.
2. **Memoria fragmentada:** Cada `Point` tiene su propio `std::vector`, que aloja su buffer en el heap de forma independiente. Con 10 millones de puntos, hay 10 millones de allocaciones dispersas en RAM.
3. **Imposible enviar por MPI:** No se puede hacer `MPI_Scatterv(points.data(), ...)` porque los datos no son contiguos. Hay que copiar todo a un buffer plano primero → doble memoria + tiempo de copia.
4. **Sin vectorización SIMD:** El compilador no puede generar instrucciones AVX porque no puede garantizar que `features.data()` de un punto sea contiguo con el del siguiente.

```
Memoria con AoS (fragmentada):
  Heap: [Point0.features → 0xA000] [Point1.features → 0xB400] [Point2.features → 0xC100]
        dispersos por el heap → cache misses constantes

Memoria con SoA (contigua):
  Heap: [x0,y0,z0, x1,y1,z1, x2,y2,z2, ...]  ← un solo bloque → prefetch óptimo
```

#### Alternativa B: `float**` (punteros a punteros)

```cpp
float** data = new float*[n_rows];
for (int i = 0; i < n_rows; i++)
    data[i] = new float[n_cols];
```

**¿Por qué es peor?**

- Misma fragmentación que AoS: cada fila es una allocación independiente.
- `n_rows` llamadas a `new` → lento en la inicialización.
- Memory leaks si no se hace `delete[]` de cada fila.
- Patrón imposible de optimizar para el compilador.

#### Alternativa C: `std::vector<std::vector<float>>`

```cpp
std::vector<std::vector<float>> data(n_rows, std::vector<float>(n_cols));
```

**¿Por qué es peor?**

- Cada fila interior es un `vector` separado con su propia allocación heap → misma fragmentación.
- Overhead de metadatos: cada `vector` almacena puntero, tamaño y capacidad (24 bytes extra por fila).
- Con 10M filas: 240 MB solo en metadatos de vectores internos.

---

### 1.2. `Centroids` — Misma filosofía contigua

#### Nuestra implementación

```cpp
struct Centroids {
  uint32_t num_clusters;
  uint32_t num_cols;
  std::vector<float> data;       // K × cols contiguo
  std::vector<uint32_t> counts;  // tamaño de cada clúster
};
```

#### Alternativa: `std::map<int, std::vector<float>>`

```cpp
std::map<int, std::vector<float>> centroids;
// centroids[0] = {1.2, 3.4, 5.6};
// centroids[1] = {7.8, 9.0, 1.1};
```

**¿Por qué es peor?**

- `std::map` almacena nodos en un árbol rojo-negro: cada centroide está en un nodo del heap separado.
- Acceso O(log K) por búsqueda frente a O(1) con indexado directo.
- Imposible enviar por MPI: no hay buffer contiguo.
- Con K = 4 y cols = 10, la diferencia es pequeña, pero el patrón es necesario para escalar a K = 1000+.

---

### 1.3. `Column_stats` — Struct plano

#### Nuestra implementación

```cpp
struct Column_stats {
  float min = 0.0f;
  float max = 0.0f;
  float mean = 0.0f;
  float variance = 0.0f;
};
```

#### Alternativa: `std::unordered_map<std::string, float>`

```cpp
std::unordered_map<std::string, float> stats;
stats["min"] = ...;
stats["max"] = ...;
```

**¿Por qué es peor?**

- Hash map con strings: cada lookup requiere hash + comparación de cadena.
- ~100× más lento que acceder a un campo de struct directamente (1 instrucción vs. hash + posible colisión).
- Overhead de memoria masivo: cada string tiene allocación dinámica propia.

---

## 2. I/O (`io_utils.cpp`)

### 2.1. Lectura binaria directa

#### Nuestra implementación

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

#### Alternativa A: Lectura de CSV línea a línea

```cpp
Dataset ReadCSV(const std::string &filepath) {
  Dataset dataset;
  std::ifstream file(filepath);
  std::string line;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    float val;
    while (ss >> val) {
      dataset.data.push_back(val);  // ← realloc constante
      if (ss.peek() == ',') ss.ignore();
    }
    dataset.n_rows++;
  }
  return dataset;
}
```

**¿Por qué es peor?**

| Aspecto | Binario | CSV |
|---|---|---|
| Operaciones de lectura | 3 `read()` | N×M `>>` + `getline` |
| Conversiones | Ninguna | N×M `string → float` |
| `push_back` con realloc | 0 | Cientos de reallocaciones |
| Tamaño en disco (10M×10) | ~400 MB | ~900 MB |
| Tiempo estimado | ~0.5 s | ~5-10 s |

El punto crítico es el `push_back`: sin conocer el tamaño total de antemano, el vector se reaoja (duplica capacidad) múltiples veces, copiando todos los datos cada vez. Nuestra versión hace `resize(total_elements)` una única vez.

#### Alternativa B: `mmap` (mapeo de memoria)

```cpp
int fd = open("dataset.bin", O_RDONLY);
float *data = (float*)mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
```

**¿Cuándo sería mejor?** Con datasets mayores que la RAM, `mmap` permite acceso bajo demanda sin cargar todo en memoria. Sin embargo:

- **Nuestro caso:** El dataset cabe en RAM y lo necesitamos completamente en un `vector` para MPI. `mmap` añadiría *page faults* bajo demanda y no es compatible con `MPI_Scatterv` (que necesita un buffer contiguo ya residente en memoria).
- **Conclusión:** Para datasets in-memory como el nuestro, `file.read()` directo es más predecible y rápido.

### 2.2. `std::optional` para errores

#### Nuestra implementación

```cpp
std::optional<Dataset> ReadBinaryFile(...);
// Uso:
auto result = ReadBinaryFile("dataset.bin");
if (result.has_value()) { /* ok */ }
```

#### Alternativa A: Excepciones

```cpp
Dataset ReadBinaryFile(...) {
  if (!file.is_open()) throw std::runtime_error("...");
  // ...
  return dataset;
}
```

**¿Por qué es peor en HPC?**

- Las excepciones C++ tienen coste cero en el *happy path* con compiladores modernos, pero el *unwinding* en caso de error es extremadamente costoso (100-1000 ciclos).
- Más importante: `-fno-exceptions` es un flag común en HPC para reducir el tamaño del binario. `std::optional` funciona sin excepciones.
- El compilador genera tablas de limpieza (`.gcc_except_table`) que aumentan el tamaño del binario y pueden afectar la localidad de caché del código.

#### Alternativa B: Código de error (estilo C)

```cpp
int ReadBinaryFile(const char *path, Dataset *out_dataset);
// Uso:
Dataset ds;
if (ReadBinaryFile("data.bin", &ds) != 0) { /* error */ }
```

**¿Por qué es peor?**

- El parámetro de salida obliga al llamador a declarar la variable antes → no se puede usar con `auto`.
- No hay claridad semántica: ¿`0` es éxito o error? Depende de la convención.
- `std::optional` es autoexplicativo y el compilador puede optimizar el Return Value Optimization (RVO) igual que con un retorno directo.

---

## 3. Generador de Datos (`tools/file_generator.cpp`)

### 3.1. Mersenne Twister con semilla fija

#### Nuestra implementación

```cpp
std::mt19937 rng(42);  // semilla fija
std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
for (uint64_t i = 0; i < total_elements; ++i)
    data[i] = dist(rng);
```

#### Alternativa A: `rand()` de C

```cpp
srand(42);
for (uint64_t i = 0; i < total_elements; ++i)
    data[i] = -100.0f + (rand() / (float)RAND_MAX) * 200.0f;
```

**¿Por qué es peor?**

- `rand()` usa un generador lineal congruencial con período corto (2³¹). Con 100M de elementos, los patrones se repiten.
- `RAND_MAX` es solo 2³¹-1 → solo 2.1 billones de valores posibles entre -100 y 100. Distribución granular.
- `rand()` no es thread-safe: si paralelizamos la generación, hay condiciones de carrera.
- `mt19937` tiene período 2¹⁹⁹³⁷-1 y distribución estadísticamente uniforme comprobada.

#### Alternativa B: `/dev/urandom`

```cpp
int fd = open("/dev/urandom", O_RDONLY);
read(fd, data.data(), total_elements * sizeof(float));
```

**¿Por qué es peor?**

- No reproducible: cada ejecución genera datos diferentes → imposible comparar benchmarks.
- Los valores no siguen una distribución uniforme en un rango específico; son bytes aleatorios reinterpretados como float → incluye NaN, infinitos, y denormales.
- La semilla fija (`42`) garantiza que todos los tests y benchmarks son **determinísticos**.

### 3.2. Escritura binaria en bloque

#### Nuestra implementación

```cpp
file.write(reinterpret_cast<const char *>(&num_rows), sizeof(uint32_t));
file.write(reinterpret_cast<const char *>(&num_cols), sizeof(uint32_t));
file.write(reinterpret_cast<const char *>(data.data()),
           total_elements * sizeof(float));
```

#### Alternativa: Escritura de CSV punto por punto

```cpp
for (uint32_t r = 0; r < num_rows; r++) {
    for (uint32_t c = 0; c < num_cols; c++) {
        file << data[r * num_cols + c];
        if (c < num_cols - 1) file << ",";
    }
    file << "\n";
}
```

**¿Por qué es peor?**

- N×M operaciones de `operator<<` con conversión `float → string` (cada una requiere `sprintf` internamente).
- El stream de C++ flushea el buffer interno frecuentemente con tantas escrituras pequeñas.
- Nuestro `file.write()` envía un solo bloque contiguo al kernel en una única syscall.

---

## 4. Distribución Inicial de Datos (`main.cpp`)

### 4.1. `MPI_Bcast` para dimensiones

#### Nuestra implementación

```cpp
uint32_t dimensions[2] = {0, 0};
if (rank == 0) {
    dimensions[0] = global_dataset.n_rows;
    dimensions[1] = global_dataset.n_cols;
}
MPI_Bcast(dimensions, 2, MPI_UINT32_T, 0, MPI_COMM_WORLD);
```

#### Alternativa: `MPI_Send` en bucle desde Rank 0

```cpp
if (rank == 0) {
    for (int i = 1; i < num_procs; i++)
        MPI_Send(dimensions, 2, MPI_UINT32_T, i, 0, MPI_COMM_WORLD);
} else {
    MPI_Recv(dimensions, 2, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
```

**¿Por qué es peor?**

- **P-1 mensajes secuenciales** desde Rank 0: latencia total = (P-1) × latencia_red.
- `MPI_Bcast` usa un **árbol binomial**: Rank 0 envía a 1, luego 0→2 y 1→3 simultáneamente, etc. Latencia = O(log P).
- Con P = 64 procesos: Send en bucle = 63 pasos. Bcast = 6 pasos. **10× más rápido**.

### 4.2. `MPI_Scatterv` con balance por remainder

#### Nuestra implementación

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

#### Alternativa A: `MPI_Scatter` con truncamiento

```cpp
int rows_per_proc = total_rows / num_procs;  // descarta el resto
MPI_Scatter(data, rows_per_proc * num_cols, MPI_FLOAT, ...);
```

**¿Por qué es peor?**

- Si `total_rows = 1003` y `num_procs = 4`, cada proceso recibe 250 filas → **se pierden 3 filas**.
- En un dataset real, perder datos es inaceptable. Además, el desbalance crece con más procesos.

#### Alternativa B: Padding con filas vacías

```cpp
// Rellenar hasta que total_rows sea divisible por num_procs
while (total_rows % num_procs != 0) {
    global_dataset.data.insert(global_dataset.data.end(), num_cols, 0.0f);
    total_rows++;
}
MPI_Scatter(...);
```

**¿Por qué es peor?**

- Las filas de relleno (ceros) **corrompen las estadísticas**: el mínimo siempre será 0, la media se desplaza.
- Las filas de relleno **afectan K-Medias**: los centroides se sesgan hacia el origen (0,0,...,0).
- `MPI_Scatterv` evita ambos problemas distribuyendo exactamente los datos reales.

### 4.3. Cálculo de `global_offset`

#### Nuestra implementación

```cpp
uint64_t global_offset = 0;
for (int i = 0; i < rank; ++i)
    global_offset += (total_rows / num_procs) + (i < remainder ? 1 : 0);
```

#### Alternativa: Fórmula cerrada

```cpp
uint64_t global_offset = rank * (total_rows / num_procs)
                       + std::min(rank, remainder);
```

**¿Es mejor la fórmula cerrada?** Técnicamente sí — O(1) vs O(P). Pero con P < 1000 procesos, la diferencia es de nanosegundos frente a una operación que ocurre una sola vez. La versión con bucle es más legible y menos propensa a errores de desbordamiento con enteros grandes.

---

## 5. Estadísticas (`stats.cpp`)

### 5.1. Reducción OpenMP con arrays

#### Nuestra implementación

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

#### Alternativa A: `#pragma omp critical` en cada iteración

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

**¿Por qué es peor?**

- `critical` serializa completamente el bloque: solo un hilo lo ejecuta a la vez.
- Con 8 hilos, 7 están siempre esperando → **peor que secuencial** por el overhead del lock.
- Rendimiento: `T_critical ≈ T_secuencial × (1 + overhead_lock)` → más lento que sin OpenMP.

#### Alternativa B: `#pragma omp atomic` por cada variable

```cpp
#pragma omp parallel for
for (uint32_t r = 0; r < num_rows; ++r) {
    const float *row_ptr = local_data.GetRowPtr(r);
    for (uint32_t c = 0; c < num_cols; ++c) {
        #pragma omp atomic
        sum[c] += row_ptr[c];
        // atomic NO soporta min/max → no aplicable
    }
}
```

**¿Por qué es peor?**

- `atomic` no soporta operaciones `min`/`max` → necesitaríamos un `critical` de todas formas.
- Incluso para la suma: N×M operaciones atómicas, cada una con barrera de memoria (5-20 ciclos extra).
- *Cache line bouncing*: múltiples cores intentan escribir en `sum[0]` simultáneamente → la línea de caché rebota entre caches L1 de diferentes cores.

#### Alternativa C: Buffers manuales privados (como en kmeans.cpp)

```cpp
#pragma omp parallel
{
    std::vector<double> my_sum(num_cols, 0.0);
    std::vector<float> my_min(num_cols, FLT_MAX);

    #pragma omp for nowait
    for (uint32_t r = 0; r < num_rows; ++r) { /* acumular en my_sum, my_min */ }

    #pragma omp critical
    {
        for (uint32_t c = 0; c < num_cols; ++c) {
            sum[c] += my_sum[c];
            min_val[c] = std::min(min_val[c], my_min[c]);
        }
    }
}
```

**¿Es peor?** Es funcional y tiene buen rendimiento, pero:

- Requiere escribir y mantener manualmente los buffers de merge.
- La sección `critical` al final es O(T) — se serializa hilo por hilo.
- La reducción nativa OpenMP fusiona en O(log T) pasos usando un árbol binario interno.
- Con T = 16 hilos: critical = 16 pasos de merge, reducción nativa = 4 pasos. **4× más rápido en el merge**.

### 5.2. Acumuladores `double` para sumas de `float`

#### Nuestra implementación

```cpp
std::vector<double> local_sum(num_cols, 0.0);      // double
std::vector<double> local_sum_sq(num_cols, 0.0);    // double
sum_ptr[c] += val;                                   // float → double implícito
sum_sq_ptr[c] += static_cast<double>(val) * val;     // double × float
```

#### Alternativa: Todo en `float`

```cpp
std::vector<float> local_sum(num_cols, 0.0f);
local_sum[c] += val;  // float + float
```

**¿Por qué es peor numéricamente?**

`float` tiene ~7 dígitos de precisión. Al sumar 10 millones de valores:

```
Ejemplo: sumar 10,000,000 valores de ~50.0
Suma real:    500,000,000.0
Suma float:   499,999,872.0   ← error de 128 por pérdida de precisión
Suma double:  500,000,000.0   ← exacto hasta 15 dígitos
```

La **cancelación catastrófica** es aún peor para la varianza: `Var = E[X²] - (E[X])²` resta dos números grandes para obtener uno pequeño. Con `float`, el resultado puede ser **negativo** (imposible matemáticamente), por eso incluimos el clamp a cero.

### 5.3. Varianza con fórmula computacional

#### Nuestra implementación

```cpp
double mean = global_sum[c] / (double)global_rows;
double mean_of_squares = global_sum_sq[c] / (double)global_rows;
double variance = mean_of_squares - (mean * mean);  // E[X²] - (E[X])²
if (variance < 0.0) variance = 0.0;                  // clamp
```

#### Alternativa: Fórmula de dos pasadas

```cpp
// Pasada 1: calcular media
double mean = 0;
for (auto x : data) mean += x;
mean /= N;

// Pasada 2: calcular varianza
double var = 0;
for (auto x : data) var += (x - mean) * (x - mean);
var /= N;
```

**¿Es mejor numéricamente?** Sí, la fórmula de dos pasadas es más estable numéricamente. **¿Es peor en rendimiento?** Mucho peor en nuestro contexto distribuido:

- Requiere **dos pasadas completas** sobre los datos → doble ancho de banda de memoria.
- En MPI, la primera pasada necesita un `MPI_Allreduce` para obtener la media global **antes** de la segunda pasada → dos rondas de comunicación.
- Nuestra fórmula de una pasada calcula `sum` y `sum_sq` simultáneamente en el mismo bucle → **1 pasada, 1 Allreduce**.

El clamp `if (variance < 0) variance = 0` compensa la menor estabilidad con coste cero.

### 5.4. `MPI_Allreduce` para combinar estadísticas

#### Nuestra implementación

```cpp
MPI_Allreduce(local_sum.data(), global_sum.data(), num_cols,
              MPI_DOUBLE, MPI_SUM, comm);
MPI_Allreduce(local_min.data(), global_min.data(), num_cols,
              MPI_FLOAT, MPI_MIN, comm);
```

#### Alternativa A: `MPI_Reduce` a Rank 0 + `MPI_Bcast`

```cpp
MPI_Reduce(local_sum.data(), global_sum.data(), num_cols,
           MPI_DOUBLE, MPI_SUM, 0, comm);
MPI_Bcast(global_sum.data(), num_cols, MPI_DOUBLE, 0, comm);
```

**¿Por qué es peor?**

- 2 operaciones colectivas = 2 × O(log P) de latencia.
- `MPI_Allreduce` fusiona ambas fases internamente con algoritmos como *recursive doubling* o *ring allreduce*.
- Las implementaciones de MPI (como OpenMPI) detectan el tamaño del mensaje y eligen automáticamente el algoritmo más eficiente para `Allreduce`.

#### Alternativa B: Reducción manual con `MPI_Send/Recv` en árbol

```cpp
// Implementar un árbol binario manualmente...
int partner = rank ^ (1 << step);
if (rank < partner) {
    MPI_Recv(buffer, ...);
    for (int c = 0; c < num_cols; c++) sum[c] += buffer[c];
} else {
    MPI_Send(sum, ...);
}
```

**¿Por qué es peor?**

- Reimplementar el árbol de reducción es propenso a errores (potencias de 2, ranks impares, etc.).
- `MPI_Allreduce` ha sido optimizado durante décadas con algoritmos adaptativos que eligen la estrategia según el tamaño del mensaje, el número de procesos, y la topología de red.
- Mantener código manual de comunicación es una pesadilla de depuración.

---

## 6. K-Medias — Inicialización (`kmeans.cpp: InitializeCentroids`)

### 6.1. Particionado uniforme por índice global

#### Nuestra implementación

```cpp
uint64_t rows_per_cluster = total_rows / num_clusters;
uint32_t cluster_id = std::min(
    static_cast<uint32_t>(global_index / rows_per_cluster),
    num_clusters - 1);
```

#### Alternativa A: K-Means++ (selección probabilística)

```cpp
// Elegir primer centroide aleatorio
// Para cada punto, calcular distancia al centroide más cercano
// Elegir siguiente centroide con probabilidad proporcional a distancia²
```

**¿Es mejor K-Means++?** Produce centroides iniciales más representativos (converge en menos iteraciones). **¿Por qué no lo usamos?**

- K-Means++ es **inherentemente secuencial**: cada centroide depende de los anteriores.
- En un entorno MPI, cada paso requiere un `MPI_Allreduce` para el centroide elegido + `MPI_Bcast` → K rondas de comunicación.
- El particionado uniforme es O(1) por punto, completamente paralelo, y no requiere comunicación para la asignación (solo para las sumas finales).

#### Alternativa B: Centroides aleatorios

```cpp
std::mt19937 rng(42);
for (int k = 0; k < num_clusters; k++)
    for (int c = 0; c < num_cols; c++)
        centroids[k][c] = dist(rng);
```

**¿Por qué es peor?**

- Los centroides aleatorios pueden estar muy lejos de cualquier dato real → primeras iteraciones desperdiciadas.
- Riesgo de clústeres vacíos: si un centroide se coloca donde no hay datos, permanece vacío indefinidamente.
- El particionado uniforme garantiza que **cada centroide empieza como la media de datos reales** → convergencia más rápida.

### 6.2. Buffers privados por hilo con merge `critical`

#### Nuestra implementación

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

#### Alternativa: `atomic` en cada acumulación

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

**¿Por qué es peor?**

Por cada fila se ejecutan **1 + num_cols** operaciones atómicas. Con 10M filas y 10 columnas:
- Total de atomics: 10M × 11 = **110 millones de operaciones atómicas**.
- Cada atomic: barrera de memoria + posible invalidación de caché = 5-20 ciclos.
- Coste estimado: 110M × 10 ciclos = **1.1 segundo** (a 1 GHz) solo en sincronización.

Con buffers privados:
- 0 operaciones atómicas durante el bucle.
- En el merge: T hilos × K clusters × cols = 8 × 4 × 10 = **320 sumas** → despreciable.

---

## 7. K-Medias — Bucle Principal (`kmeans.cpp: RunKMeans`)

### 7.1. Asignación: distancia euclidiana cuadrática

#### Nuestra implementación

```cpp
float dist_sq = 0.0f;
for (uint32_t c = 0; c < num_cols; ++c) {
    float diff = row_ptr[c] - centroid_ptr[c];
    dist_sq += diff * diff;
}
if (dist_sq < min_dist_sq) { best_cluster = k; ... }
```

#### Alternativa A: Distancia euclidiana con `sqrt`

```cpp
float dist = 0.0f;
for (uint32_t c = 0; c < num_cols; ++c) {
    float diff = row_ptr[c] - centroid_ptr[c];
    dist += diff * diff;
}
dist = std::sqrt(dist);  // ← innecesario
```

**¿Por qué es peor?**

- `sqrt` es una de las operaciones más costosas del procesador (~15-20 ciclos frente a 1 ciclo de multiplicación).
- Como solo comparamos distancias (`dist < min_dist`), y `sqrt` es una función monótona, `dist² < min_dist²` produce el mismo resultado sin la raíz cuadrada.
- Con 10M filas × K = 4 centroides: **40 millones de `sqrt` eliminados**.

#### Alternativa B: Distancia Manhattan

```cpp
float dist = 0.0f;
for (uint32_t c = 0; c < num_cols; ++c)
    dist += std::abs(row_ptr[c] - centroid_ptr[c]);
```

**¿Es más rápida?** Sí (`abs` es más barato que `x*x`). **¿Es correcta?** No para K-Medias estándar. K-Medias minimiza la varianza intra-clúster, que está definida con distancia L2 (euclidiana). Usar Manhattan (L1) converge hacia *medianas* en lugar de *medias*, lo que es un algoritmo diferente (K-Medians).

### 7.2. Convergencia: umbral del 5%

#### Nuestra implementación

```cpp
uint64_t global_displacement = 0;
MPI_Allreduce(&local_displacements, &global_displacement, 1,
              MPI_UINT64_T, MPI_SUM, comm);

double displacement_ratio = (double)global_displacement / (double)total_rows;
if (displacement_ratio < 0.05) { converged = true; break; }
```

#### Alternativa A: Convergencia exacta (0 cambios)

```cpp
if (global_displacement == 0) { converged = true; break; }
```

**¿Por qué es peor?**

- K-Medias puede oscilar con 2-3 puntos fronterizos que cambian de clúster en cada iteración indefinidamente.
- Cada iteración extra implica un `MPI_Alltoallv` completo (redistribución masiva de datos).
- El umbral del 5% detiene el algoritmo **cuando el 95% de los puntos ya están estables**, ahorrando potencialmente decenas de iteraciones costosas.

#### Alternativa B: Convergencia por cambio de centroides

```cpp
double centroid_shift = 0;
for (int k = 0; k < K; k++)
    for (int c = 0; c < cols; c++)
        centroid_shift += pow(new_centroid[k][c] - old_centroid[k][c], 2);
if (centroid_shift < epsilon) break;
```

**¿Es mejor?** Es más finamente sintonizable, pero requiere almacenar los centroides anteriores y calcular la diferencia. El ratio de desplazamiento es más intuitivo ("% de puntos que cambiaron") y no requiere memoria adicional.

### 7.3. Redistribución de datos con `MPI_Alltoallv`

#### Nuestra implementación

```cpp
// 1. Particionar filas por rank destino
for (uint32_t r = 0; r < current_num_rows; ++r) {
    int owner_rank = get_owner_rank(local_assignments[r]);
    if (owner_rank == rank) {
        next_local_data.insert(...);  // queda aquí
    } else {
        send_data_buffers[owner_rank].insert(...);  // enviar
    }
}

// 2. Notificar cuántos puntos envía cada rank a cada otro
MPI_Alltoall(send_point_counts, 1, MPI_INT, recv_point_counts, 1, MPI_INT, comm);

// 3. Intercambiar datos
MPI_Alltoallv(flat_send_data, send_data_counts, send_data_displacements, MPI_FLOAT,
              flat_recv_data, recv_data_counts, recv_data_displacements, MPI_FLOAT, comm);

// 4. Merge: datos recibidos + datos locales
next_local_data.insert(next_local_data.end(), flat_recv_data.begin(), ...);
```

#### Alternativa A: `MPI_Isend`/`MPI_Irecv` punto a punto

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

**¿Por qué es peor?**

| Aspecto | `MPI_Alltoallv` | `Isend`/`Irecv` manual |
|---|---|---|
| Líneas de código | ~10 | ~30+ |
| Determinación de recv_counts | `MPI_Alltoall` automático | Hay que intercambiarlos manualmente |
| Optimización interna | MPI elige algoritmo (pairwise, Bruck, linear) | Solo envío directo |
| Riesgo de deadlock | Ninguno | Posible si el orden de Send/Recv no es correcto |
| Integración con hardware | Puede usar RDMA, offload a NIC | Depende de la implementación |

#### Alternativa B: No redistribuir (solo broadcast de centroides)

```cpp
// Sin MPI_Alltoallv: cada rank mantiene sus datos originales
// Solo compartir centroides actualizados
MPI_Allreduce(local_sums, global_sums, K * cols, MPI_DOUBLE, MPI_SUM, comm);
MPI_Allreduce(local_counts, global_counts, K, MPI_UINT32_T, MPI_SUM, comm);
```

**¿Es más simple?** Sí. **¿Es peor en rendimiento a largo plazo?** Puede serlo:

- Sin redistribución, cada rank acumula sumas parciales para **todos** los K clústeres, aunque la mayoría de sus puntos pertenezcan a 1-2 clústeres.
- Con redistribución, cada rank solo tiene puntos de sus clústeres asignados → las acumulaciones son más eficientes y locales.
- Sin embargo, la redistribución tiene un coste fijo por iteración (`MPI_Alltoallv`). Para K pequeño y pocas iteraciones, la versión sin redistribución puede ser más rápida.

**Nuestra elección:** La redistribución escala mejor con K grande y datasets masivos, que es el caso de uso HPC para el que diseñamos el sistema.

### 7.4. Propiedad de clústeres: `get_owner_rank`

#### Nuestra implementación

```cpp
auto get_owner_rank = [&](uint32_t cluster_id) -> int {
    uint32_t base_clusters = num_clusters / num_procs;
    uint32_t remainder = num_clusters % num_procs;
    // Distribuir clústeres equitativamente: primeros 'remainder' ranks tienen 1 extra
    int rank_owner = 0;
    uint32_t limit = 0;
    for (int i = 0; i < num_procs; ++i) {
        limit += base_clusters + (i < remainder ? 1 : 0);
        if (cluster_id < limit) { rank_owner = i; break; }
    }
    return rank_owner;
};
```

#### Alternativa A: Round-robin simple

```cpp
int get_owner_rank(uint32_t cluster_id) {
    return cluster_id % num_procs;
}
```

**¿Es mejor?** Más simple, O(1). **¿Es peor?** Puede crear desbalance: si K = 5 y P = 4, Rank 0 tiene clústeres {0, 4} (2 clústeres) y Rank 3 tiene solo clúster {3}. Nuestra versión distribuye 2-1-1-1, que es lo más equitativo posible.

#### Alternativa B: Hash

```cpp
int get_owner_rank(uint32_t cluster_id) {
    return std::hash<uint32_t>{}(cluster_id) % num_procs;
}
```

**¿Es peor?** El hash puede producir colisiones y distribución desigual con K pequeño. Además, no es determinístico entre compiladores → resultados no reproducibles.

---

## 8. Configuración del Build (`CMakeLists.txt`)

### 8.1. C++17 sin extensiones GNU

#### Nuestra implementación

```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
```

#### Alternativa: Flags manuales

```cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
```

**¿Por qué es peor?**

- Solo funciona con GCC/Clang. MSVC usa `/std:c++17`.
- `CMAKE_CXX_EXTENSIONS OFF` evita `gnu++17`, que activa extensiones no portables. Con `-std=c++17` manual, CMake podría usar `gnu++17` por defecto.
- `CMAKE_CXX_STANDARD_REQUIRED ON` falla la configuración si el compilador no soporta C++17, en vez de silenciosamente degradar a C++14.

### 8.2. Targets modernos para MPI y OpenMP

#### Nuestra implementación

```cmake
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
```

#### Alternativa: Flags manuales

```cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
include_directories(/usr/lib/openmpi/include)
link_directories(/usr/lib/openmpi/lib)
target_link_libraries(${PROJECT_NAME} mpi)
```

**¿Por qué es peor?**

- Paths hardcodeados: solo funciona en una distribución Linux específica.
- `MPI::MPI_CXX` y `OpenMP::OpenMP_CXX` son *imported targets* que CMake configura automáticamente con los includes, linker flags, y compiler flags correctos para **cualquier** sistema.
- Con targets modernos, un cambio de compilador (GCC → Intel) o de MPI (OpenMPI → MPICH) no requiere modificar el CMakeLists.txt.

---

## 9. Resumen General: Todas las Decisiones

| Componente | Decisión | Alternativa peor | Por qué es peor |
|---|---|---|---|
| `Dataset` | `vector<float>` contiguo | `vector<Point>` AoS | Cache misses, sin SIMD, no MPI-compatible |
| `Dataset` | `vector<float>` contiguo | `float**` punteros | Fragmentación, memory leaks |
| `Centroids` | `vector<float>` plano | `map<int, vector>` | O(log K) acceso, no contiguo |
| `Column_stats` | Struct plano | `unordered_map<string>` | 100× más lento por lookup |
| I/O | Binario directo | CSV línea a línea | 10-20× más lento, parseo costoso |
| I/O | `std::optional` | Excepciones | Coste de unwinding, incompatible con `-fno-exceptions` |
| Generador | `mt19937` fija | `rand()` | Periodo corto, no thread-safe, mala distribución |
| Broadcast | `MPI_Bcast` | `MPI_Send` en bucle | O(log P) vs O(P) |
| Distribución | `MPI_Scatterv` | `MPI_Scatter` truncando | Pierde datos, desbalanceo |
| Stats OpenMP | Reducción nativa array | `critical` cada fila | Serializa todo: peor que secuencial |
| Stats OpenMP | Reducción nativa array | `atomic` cada variable | No soporta min/max, cache bouncing |
| Stats | Fórmula 1 pasada | 2 pasadas | Doble ancho banda + 2 Allreduce vs 1 |
| Stats MPI | `MPI_Allreduce` | `Reduce` + `Bcast` | 2 colectivas vs 1 fusionada |
| K-Means init | Partición uniforme | K-Means++ | Inherentemente secuencial en MPI |
| K-Means init | Partición uniforme | Aleatorio | Centroides lejos de datos reales |
| K-Means acum. | Buffers privados + critical | `atomic` cada suma | 110M atomics vs ~300 sumas |
| K-Means dist. | dist² sin sqrt | dist con sqrt | 40M sqrt eliminados |
| K-Means conv. | Umbral 5% | Convergencia exacta | Oscilación infinita posible |
| Redistribución | `MPI_Alltoallv` | `Isend`/`Irecv` | Más código, sin opt. interna, riesgo deadlock |
| Owner rank | Balance equitativo | Round-robin | Desbalanceo con K % P ≠ 0 |
| Build | Targets CMake modernos | Flags manuales | No portable, paths hardcodeados |
| Build | `-O3 -march=native` | Default (`-O0`) | 5-15× sin vectorización SIMD |
