# Justificación de Diseño — Práctica 1: Pipeline HPC con K-Medias

## 1. Introducción y Objetivos

El objetivo de esta práctica es diseñar un sistema capaz de procesar grandes volúmenes de datos numéricos en el menor tiempo posible. El programa ejecuta tres fases sobre un dataset binario:

1. **I/O y distribución** — Lectura del dataset y reparto entre procesos MPI.
2. **Estadísticas por columna** — Cálculo paralelo de mínimo, máximo, media y varianza.
3. **K-Medias** — Agrupación iterativa con redistribución de datos entre nodos.

Para maximizar el rendimiento, se emplea una estrategia de **paralelización híbrida**: **OpenMPI** para tareas de grano grueso entre nodos y **OpenMP** para cómputo de grano fino dentro de cada nodo. A continuación se justifica cada decisión de diseño explicando *por qué* es más rápida que las alternativas.

---

## 2. Diseño de Estructuras de Datos: Data-Oriented Design

### 2.1. Decisión: Vectores contiguos planos (SoA) en lugar de clases con punteros (AoS)

Las estructuras `Dataset` y `Centroids` almacenan sus datos en un único `std::vector<float>` contiguo en fila mayor (*row-major*), en lugar de utilizar objetos individuales como `class Point { float x, y, z; }`.

**¿Por qué AoS es más lento?**

Con un Array of Structures, cada punto sería un objeto independiente potencialmente disperso en memoria. Al iterar sobre N puntos para calcular distancias o estadísticas, la CPU necesita saltar de un objeto a otro:

```
AoS en memoria (punteros a objetos individuales):
  Point0 [x0, y0, z0]  →  Point1 [x1, y1, z1]  →  Point2 [x2, y2, z2]  ...
  ↑ posiblemente no contiguo, genera cache misses
```

**¿Por qué SoA/vector contiguo es más rápido?**

Con un vector plano contiguo, los datos están uno tras otro en RAM. Cuando la CPU lee el primer float, la línea de caché (normalmente 64 bytes = 16 floats) trae consigo los siguientes 15 valores automáticamente:

```
Vector contiguo en memoria:
  [x0, y0, z0, x1, y1, z1, x2, y2, z2, ...]
  ↑ acceso secuencial → prefetch automático → ~0 cache misses
```

**Ejemplo concreto:** En el bucle de distancia euclidiana de K-Medias (`kmeans.cpp:159-162`), cada iteración accede a `row_ptr[c]` y `centroid_ptr[c]` de forma secuencial. Con datos contiguos, el hardware prefetcher de la CPU anticipa estos accesos y precarga las siguientes líneas de caché antes de que se necesiten. Con AoS, cada acceso a un nuevo punto sería un *cache miss* potencial, multiplicando la latencia de memoria por un factor de 10-100×.

**Beneficio adicional para MPI:** Los datos contiguos permiten enviar bloques enteros con `MPI_Scatterv` y `MPI_Alltoallv` sin necesidad de serializar/deserializar estructuras complejas. Un solo `MPI_Scatterv` sobre `dataset.data.data()` transmite miles de filas en una operación, mientras que con AoS necesitaríamos empaquetar cada objeto en un buffer plano primero.

**Beneficio para auto-vectorización (SIMD):** El flag `-march=native` permite al compilador generar instrucciones AVX/AVX2 que procesan 8 floats simultáneamente. Esto solo funciona con datos contiguos y alineados; con AoS, el compilador no puede garantizar la contigüidad y desactiva la vectorización.

### 2.2. Decisión: `float` en lugar de `double` para los datos

Los datos del dataset se almacenan como `float` (32 bits) en lugar de `double` (64 bits).

**¿Por qué es más rápido?**

- **El doble de datos por línea de caché:** Una línea de 64 bytes almacena 16 floats pero solo 8 doubles. Esto duplica la eficiencia del ancho de banda de memoria.
- **El doble de elementos por instrucción SIMD:** Un registro AVX de 256 bits procesa 8 floats frente a 4 doubles por instrucción.
- **Mitad de ancho de banda MPI:** Transferir N filas de floats requiere la mitad de bytes que con doubles, reduciendo la latencia de red.

**¿Dónde se usa `double`?** Exclusivamente en los **acumuladores intermedios** (`local_sums`, `global_sums`) para evitar errores de redondeo al sumar millones de valores pequeños. Esto es un compromiso deliberado: se gana precisión numérica donde es crítica (acumulación) sin sacrificar rendimiento en el almacenamiento y transmisión masiva de datos.

### 2.3. Decisión: Acceso por puntero a filas (`GetRowPtr`)

Los structs `Dataset` y `Centroids` exponen un método `GetRowPtr(row)` que devuelve un puntero directo al inicio de cada fila.

**¿Por qué es más rápido que indexación 2D?**

Calcular `data[row * n_cols + col]` en cada acceso requiere una multiplicación por iteración. Con `GetRowPtr`, la multiplicación se realiza una sola vez por fila y el bucle interior recorre el puntero de forma secuencial:

```cpp
// Con GetRowPtr (1 multiplicación por fila):
const float *row_ptr = local_data.GetRowPtr(r);  // row_ptr = &data[r * n_cols]
for (uint32_t c = 0; c < num_cols; ++c)
    sum += row_ptr[c];  // solo un incremento de puntero

// Sin GetRowPtr (1 multiplicación por elemento):
for (uint32_t c = 0; c < num_cols; ++c)
    sum += data[r * n_cols + c];  // multiplicación en cada iteración
```

Además, el puntero directo facilita que el compilador detecte el patrón de acceso lineal y genere instrucciones de prefetch y vectorización.

---

## 3. I/O: Formato Binario Nativo

### 3.1. Decisión: Lectura binaria directa en lugar de CSV/texto

El dataset se lee con `file.read()` directamente a un `std::vector<float>`, sin parseo de texto.

**¿Por qué CSV/texto es más lento?**

1. **Parseo carácter por carácter:** Leer un CSV requiere identificar delimitadores (`,`, `\n`), convertir cadenas a float (`std::stof` o `sscanf`), y gestionar buffers intermedios. Cada conversión string→float requiere decenas de instrucciones.
2. **Tamaño en disco:** El número `99.743652` ocupa 9 bytes como texto pero solo 4 bytes como `float` binario. Un dataset de 10 millones de filas × 10 columnas ocuparía ~900 MB en CSV frente a ~400 MB en binario.
3. **Lectura en un solo `read()`:** Con formato binario, se lee todo el dataset en una única llamada al sistema: `file.read(data.data(), total_elements * sizeof(float))`. No hay bucles, no hay parseo, no hay allocaciones intermedias.

**Ejemplo de magnitud:** Para un dataset de 10M filas × 10 columnas:
- **CSV:** ~900 MB en disco, ~5-10 segundos de parseo (limitado por CPU).
- **Binario:** ~400 MB en disco, ~0.5-1 segundo (limitado solo por disco/SSD).

### 3.2. Decisión: `std::optional` para gestión de errores en I/O

`ReadBinaryFile` retorna `std::optional<Dataset>` en lugar de lanzar excepciones o usar códigos de error.

**¿Por qué?** Las excepciones de C++ tienen un coste añadido en el *unwind* del stack y pueden interferir con las optimizaciones del compilador en la ruta de éxito. `std::optional` permite verificar el éxito sin coste adicional en el caso normal (el compilador optimiza el booleano interno del optional).

---

## 4. Estrategia de Paralelización

### 4.1. Nivel Inter-Nodo: OpenMPI

#### 4.1.1. Reparto inicial: `MPI_Scatterv` con balance de carga

Los datos se reparten equitativamente entre procesos MPI, con un balance exacto cuando las filas no son divisibles:

```cpp
int local_rows = total_rows / num_procs;
int remainder = total_rows % num_procs;
if (rank < remainder) local_rows++;  // primeros 'remainder' ranks reciben una fila extra
```

**¿Por qué no `MPI_Scatter` simple?** `MPI_Scatter` requiere que todos los procesos reciban exactamente el mismo número de elementos. Si `total_rows` no es divisible por `num_procs` (caso habitual), habría que truncar o rellenar datos. `MPI_Scatterv` acepta conteos y desplazamientos individuales para cada rank, garantizando que **ningún dato se pierde y ningún rank queda desbalanceado por más de 1 fila**.

**¿Por qué no envíos punto a punto (`MPI_Send/Recv`)?** Un bucle de `MPI_Send` es secuencial: Rank 0 envía a Rank 1, espera, envía a Rank 2, espera... Con P procesos, el tiempo de distribución es O(P). `MPI_Scatterv` es una operación colectiva que la implementación MPI optimiza internamente utilizando árboles de comunicación (árbol binomial o binario), reduciendo la latencia a O(log P).

#### 4.1.2. Reducción global de estadísticas: `MPI_Allreduce`

Las estadísticas parciales de cada nodo se combinan con `MPI_Allreduce` en lugar de `MPI_Reduce` + `MPI_Bcast`.

**¿Por qué es más rápido que Reduce + Bcast?**

- `MPI_Reduce` + `MPI_Bcast` = 2 operaciones colectivas, cada una con latencia O(log P).
- `MPI_Allreduce` = 1 operación que internamente fusiona ambas fases (reduce-scatter + allgather) con la mitad de mensajes.
- Las implementaciones modernas de MPI (OpenMPI, MPICH) emplean el algoritmo *recursive doubling* o *ring reduce* para `Allreduce`, que es asintóticamente óptimo en ancho de banda.

**¿Por qué `MPI_Allreduce` y no `MPI_Reduce` solo en Rank 0?** Porque tanto las estadísticas como los centroides son necesarios en **todos** los ranks simultáneamente para la siguiente iteración. Si solo Rank 0 tuviera los resultados, necesitaríamos un `MPI_Bcast` adicional, duplicando la comunicación.

#### 4.1.3. Redistribución de datos en K-Medias: `MPI_Alltoallv`

Cuando un punto cambia de clúster y el clúster pertenece a un rank diferente, el punto se migra usando `MPI_Alltoallv`.

**¿Por qué no envíos punto a punto?**

Con `MPI_Isend`/`MPI_Irecv` manuales, cada rank necesita P envíos + P recepciones, dos bucles anidados, y lógica de sincronización con `MPI_Waitall`. Esto genera:
- P² mensajes totales en el sistema.
- Código complejo y propenso a deadlocks.
- Imposibilidad de que la biblioteca MPI optimice el patrón de comunicación.

`MPI_Alltoallv` expresa la semántica exacta ("todos envían a todos con cantidades variables") en una sola llamada, permitiendo que la implementación MPI:
1. Agrupe mensajes hacia el mismo destino.
2. Utilice comunicación en pipeline o RDMA directo cuando está disponible.
3. Evite congestión de red mediante planificación inteligente.

**¿Por qué redistribuir datos y no solo centroides?** La redistribución garantiza **localidad de datos**: cada rank solo tiene los puntos de sus clústeres asignados. Esto tiene dos beneficios:
1. **El recálculo de centroides es completamente local** — no se necesita comunicación para acumular sumas parciales por clúster (solo el `MPI_Allreduce` final para la media global).
2. **Mejor balance de caché** — cada rank trabaja con un subconjunto más coherente de datos en cada iteración.

#### 4.1.4. Convergencia con `MPI_Allreduce` sobre desplazamientos

La verificación de convergencia utiliza `MPI_Allreduce` con `MPI_SUM` sobre el número de puntos que cambiaron de clúster.

**¿Por qué un umbral del 5%?** Este umbral evita iteraciones innecesarias cuando el algoritmo ya está "prácticamente convergido". Sin umbral (convergencia exacta con 0 cambios), K-Medias puede ejecutar decenas de iteraciones adicionales moviendo 2-3 puntos fronterizos entre clústeres, donde cada iteración implica una redistribución completa de datos con `MPI_Alltoallv`.

### 4.2. Nivel Intra-Nodo: OpenMP

#### 4.2.1. Estadísticas: Reducciones nativas de array (`reduction`)

En `stats.cpp`, el cálculo de estadísticas utiliza cláusulas de reducción OpenMP sobre arrays:

```cpp
#pragma omp parallel for reduction(+: sum_ptr[:num_cols], sum_sq_ptr[:num_cols]) \
    reduction(min: min_ptr[:num_cols]) reduction(max: max_ptr[:num_cols])
```

**¿Por qué es más rápido que alternativas?**

| Alternativa | Problema | Coste |
|---|---|---|
| `#pragma omp critical` | Cada hilo espera su turno para actualizar → serialización total | O(T × N) donde T = hilos |
| `#pragma omp atomic` por cada var. | Solo funciona con operaciones escalares, no con arrays | No aplicable a min/max |
| Buffers privados + merge manual | Funciona, pero requiere código manual y una sección `critical` al final | O(T × cols) para merge |
| **Reducción nativa** | **El runtime OpenMP gestiona los buffers internamente, fusiona con árboles** | **O(log T × cols)** |

La reducción nativa es óptima porque:
1. **Elimina la contención**: Cada hilo opera sobre su propia copia del array sin locks.
2. **Merge en árbol**: La combinación final se realiza en O(log T) pasos en lugar de T pasos secuenciales.
3. **Localidad de caché**: Las copias privadas caben en la caché L1/L2 de cada core, evitando tráfico de coherencia entre cores (*cache bouncing*).

**Ejemplo:** Con 8 hilos sobre un dataset de 10M filas × 10 columnas:
- `critical`: 8 hilos compiten por un lock → ~8× más lento que secuencial.
- Reducción nativa: 8 hilos trabajan independientes, merge en 3 pasos (log₂ 8) → ~8× speedup.

#### 4.2.2. K-Medias: Buffers privados por hilo con merge `critical`

En `kmeans.cpp`, la acumulación de sumas y conteos por clúster utiliza **buffers privados por hilo** con merge en una sección `critical`:

```cpp
#pragma omp parallel
{
    std::vector<double>   thread_sums(K * cols, 0.0);
    std::vector<uint32_t> thread_counts(K, 0);

    #pragma omp for nowait
    for (uint32_t r = 0; r < num_rows; ++r) { /* acumular en thread_sums/counts */ }

    #pragma omp critical
    { /* merge thread_sums → local_sums */ }
}
```

**¿Por qué no usar `reduction` como en stats?**

La cláusula `reduction(+:...)` de OpenMP funciona con arrays de tamaño fijo conocidos en tiempo de compilación o con punteros a arrays simples. Sin embargo, aquí la acumulación es en una matriz 2D (K × cols) indexada por `cluster_id`, que varía dinámicamente. Los buffers privados por hilo ofrecen total flexibilidad para este patrón.

**¿Por qué no usar `atomic`?**

Si K = 4 y cols = 10, cada iteración del bucle actualiza 11 posiciones diferentes (10 sumas + 1 conteo). Con `atomic`:
- Cada operación atómica tiene un coste de 5-20 ciclos por la barrera de memoria.
- Con varios hilos actualizando los mismos clústeres → *cache line bouncing* entre cores (invalidaciones constantes de líneas de caché compartidas).
- Coste total: 11 × 5-20 ciclos × N filas.

Con buffers privados:
- Cero contención durante el bucle — cada hilo escribe en su propia memoria.
- Un único merge `critical` al final con T pasos (T = número de hilos), que es despreciable frente al bucle principal.

**¿Por qué `nowait`?**

La directiva `nowait` elimina la barrera implícita al final del `omp for`. Esto permite que los hilos que terminan antes entren directamente a la sección `critical` sin esperar a los más lentos, reduciendo el tiempo idle.

#### 4.2.3. Desplazamientos con `omp atomic`

En el paso de asignación de K-Medias, el conteo de desplazamientos (cuántos puntos cambiaron de clúster) usa un esquema de variable privada por hilo + merge con `atomic`:

```cpp
#pragma omp parallel
{
    uint64_t thread_displacement = 0;
    #pragma omp for nowait
    for (...) { if (cambió) thread_displacement++; }
    #pragma omp atomic
    local_displacements += thread_displacement;
}
```

**¿Por qué es más rápido que un `atomic` directo por iteración?**

- `atomic` directo: N operaciones atómicas (una por fila), cada una con barrera de memoria.
- Acumulador privado + 1 `atomic`: N incrementos locales (coste 0 de sincronización) + T operaciones atómicas al final (T << N).

**¿Por qué no `reduction(+:local_displacements)`?**

Funcionaría igual de bien. El patrón manual se eligió para ser consistente con el resto de la función (buffers privados + merge), y para evitar la barrera implícita del `parallel for reduction`, ya que `nowait` es necesario para el siguiente `critical`.

---

## 5. Flags de Compilación

```cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native -DNDEBUG")
```

| Flag | Propósito | Impacto en rendimiento |
|---|---|---|
| `-O3` | Máximo nivel de optimización: inlining agresivo, desenrollado de bucles, eliminación de código muerto, vectorización automática. | 5-15× más rápido que `-O0` |
| `-march=native` | Genera instrucciones AVX/AVX2/FMA específicas de la CPU donde se compila. | 2-4× en bucles numéricos por vectorización SIMD |
| `-DNDEBUG` | Desactiva `assert()` para eliminar comprobaciones en producción. | Elimina overhead de verificación en hot loops |
| `-Wall -Wextra` | Advertencias de compilación para mantener código limpio. | Sin impacto en runtime |

**¿Por qué no `-Ofast`?** `-Ofast` incluye `-ffast-math`, que permite reordenar operaciones de punto flotante (rompe asociatividad). Esto puede cambiar los resultados numéricos de las reducciones y la varianza. `-O3` es el máximo nivel de optimización que **preserva la semántica IEEE 754**.

---

## 6. Ventajas del Diseño Completo: Flujo de Datos de Extremo a Extremo

```
┌───────────────────────────────────────────────────────── RANK 0 ─────────────┐
│  Disco ──── file.read() ──── vector<float> contiguo ──── MPI_Scatterv ──→   │
└──────────────────────────────────────────────────────────────────────────────┘
                          ↓                                    ↓
                ┌─── RANK 0 ───┐  ┌─── RANK 1 ───┐  ┌─── RANK 2 ───┐
                │  OpenMP par. │  │  OpenMP par. │  │  OpenMP par. │
                │  Stats local │  │  Stats local │  │  Stats local │
                └──── ↓ ───────┘  └──── ↓ ───────┘  └──── ↓ ───────┘
                         └─── MPI_Allreduce(SUM/MIN/MAX) ───┘
                                         ↓
                              Stats globales en todos
                                         ↓
                ┌─── RANK 0 ───┐  ┌─── RANK 1 ───┐  ┌─── RANK 2 ───┐
                │  Asignar pts │  │  Asignar pts │  │  Asignar pts │
                │  (OpenMP)    │  │  (OpenMP)    │  │  (OpenMP)    │
                └──── ↓ ───────┘  └──── ↓ ───────┘  └──── ↓ ───────┘
                │  Convergenc? │ ←── MPI_Allreduce(SUM displacements)
                └──── ↓ ───────┘
                │  Redistrib.  │ ←── MPI_Alltoallv (datos + asignaciones)
                └──── ↓ ───────┘
                │  Update cent │ ←── OpenMP par. + MPI_Allreduce
                └──────────────┘
                     ↺ iterar hasta convergencia
```

Cada componente está diseñado para que los datos fluyan de forma contigua, sin copias intermedias innecesarias, minimizando latencia de memoria y red en cada fase.

---

## 7. Resumen: Comparativa de Decisiones

| Decisión de diseño | Alternativa rechazada | Factor de mejora estimado |
|---|---|---|
| Vector contiguo (SoA) | Array of Structures (AoS) | 3-10× (cache locality) |
| `float` para datos | `double` para todo | 2× (ancho banda mem/red) |
| Binario nativo para I/O | CSV/texto | 5-20× (sin parseo) |
| `MPI_Scatterv` | `MPI_Send` en bucle | O(log P) vs O(P) |
| `MPI_Allreduce` | `Reduce` + `Bcast` | ~2× menos mensajes |
| `MPI_Alltoallv` | `Isend`/`Irecv` manuales | Optimización interna MPI + robustez |
| Reducción nativa OpenMP | `critical` en cada iteración | O(log T) vs O(T×N) |
| Buffers privados + merge | `atomic` por iteración | 0 contención en hot loop |
| `-O3 -march=native` | Compilación por defecto | 5-15× (vectorización SIMD) |
| Redistribución de datos | Centroides-only broadcast | Mejora localidad de caché |
