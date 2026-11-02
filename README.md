# Graph Coloring with CUDA

**GPU Architecture & Programming — Cairo University, Faculty of Engineering (Spring 2024)**  
**Team:** Abdelaziz Salah and Ahmed Hosny  
**Languages:** CUDA C/C++, C, Python · **Parallel programming:** CUDA, OpenMP

This project explores graph coloring on the GPU using an iterative, vertex-based CUDA implementation. We built a sequential backtracking solver as a CPU reference, implemented the parallel coloring and conflict-detection stages in CUDA, and compared the results with a separate OpenMP implementation.

The goal is to assign a color to every vertex of an undirected graph such that **no two connected vertices share a color**. Finding a coloring that uses the *fewest possible* colors is a difficult optimization problem, so the CUDA implementation uses an iterative heuristic rather than an exact minimum-color search.

## How the CUDA implementation works

The CUDA program is in [`parallelGraphColouring.cu`](parallelGraphColouring.cu). Its main steps are:

1. **Read an undirected graph.** The input contains the number of vertices and edges, followed by one vertex pair per edge. Vertex indices are zero-based.
2. **Build a CSR representation.** `readGraphInCSRFormat()` first builds adjacency lists, then flattens them into `neighbours` and `srcPointers`. For an undirected graph, each input edge is stored in both directions.
3. **Copy the graph to GPU memory.** The host allocates device buffers with `cudaMalloc` and transfers the CSR arrays with `cudaMemcpy`.
4. **Assign tentative colors in parallel.** `assignColoursKernel` maps one GPU thread to a vertex. For vertices marked as conflicting, it scans the neighboring colors, records forbidden values, and selects the first available color in the range `1 .. maxDegree + 1`.
5. **Detect conflicts.** `detectConflictsKernel` checks each vertex against its neighbors. When two adjacent vertices have the same color, the vertex with the higher index is marked for recoloring (`neighbour < node`).
6. **Repeat until no conflicts remain.** The host alternates the coloring and conflict-detection kernels until the device reports no remaining conflicts.
7. **Return and check the coloring.** The resulting colors are copied back to the CPU. The program prints the assignments, reports the CUDA elapsed time, and checks adjacent vertices for matching colors.

### CUDA kernels

| Kernel | Work per thread | Role |
| --- | --- | --- |
| `assignColoursKernel` | One vertex | Chooses a color not currently used by the vertex's neighbors; skips vertices that do not need recoloring. |
| `detectConflictsKernel` | One vertex | Checks neighbors for equal colors and marks vertices for another iteration. |

A thread identifies its vertex with:

```cpp
int node = blockIdx.x * blockDim.x + threadIdx.x;
```

The host launches enough blocks to cover the graph, using `MAX_THREADS = 1024` threads per block in the current implementation. Kernel completion is synchronized with `cudaDeviceSynchronize()`. GPU execution time is recorded using CUDA events.

### Why CSR?

An adjacency matrix is simple to use, but allocates space for every possible vertex pair. CSR stores only the adjacency entries and the offsets that delimit each vertex's neighbors:

- `srcPointers[v]` is the starting offset for vertex `v`.
- `srcPointers[v + 1]` is the ending offset.
- `neighbours[srcPointers[v] ... srcPointers[v + 1] - 1]` contains its neighbors.

This is useful for sparse graphs, although the variable number of neighbors per vertex can also lead to irregular memory access and uneven thread workloads on the GPU.

## CPU and OpenMP versions

**Sequential solver — [`sequentialImplementation.c`](sequentialImplementation.c)**

The CPU version stores the graph as an adjacency matrix and uses recursive backtracking. It takes a specified number of colors, checks whether the current vertex can use each one, and backtracks when the assignment cannot be completed. Unlike the CUDA heuristic, this program answers whether a coloring is possible using the given color limit.

**OpenMP version — [`openMP/openMP.cpp`](openMP/openMP.cpp)**

The OpenMP implementation provides another parallel approach on the CPU. It assigns tentative colors, identifies conflicting vertices, and repeats until all vertices have colors. The `openMP/` directory also contains a graph generator and sample edge-list inputs.

## Build and run

### Requirements

- NVIDIA GPU with CUDA support and the CUDA Toolkit (`nvcc`) for the GPU implementation
- GCC/G++ for the CPU and OpenMP implementations
- Python 3 and NumPy for test-case generation

The source uses GNU C++ headers (`bits/stdc++.h`), so a Linux/GCC-based environment is the simplest setup.

### CUDA version

From the repository root:

```bash
nvcc -O2 parallelGraphColouring.cu -o cuda_coloring
./cuda_coloring openMP/5.txt
```

The included [`openMP/5.txt`](openMP/5.txt) describes a five-vertex cycle:

```text
5 5
0 1
0 4
1 2
2 3
3 4
```

The first line is `number_of_vertices number_of_edges`; each following line contains the two zero-based endpoints of an undirected edge. The program prints the chosen color for each vertex, the highest color label it used, elapsed GPU time, and the number of adjacency conflicts found.

The executable also accepts an optional second argument for an output file:

```bash
./cuda_coloring openMP/5.txt colors.txt
```

In the current source, debug output is enabled, so the output file may also contain verification messages after the color list.

### Sequential CPU version

```bash
gcc -O2 sequentialImplementation.c -o sequential_coloring
```

The sequential program expects a *different input format*: one line with the vertex count, an `N × N` adjacency matrix, and a final line specifying the allowed number of colors. The repository includes matrix-format inputs in [`test_cases/`](test_cases/), although randomly generated cases should be checked for self-loops before use.

For a small graph without self-loops, a five-vertex cycle can be specified as:

```text
5
0 1 0 0 1
1 0 1 0 0
0 1 0 1 0
0 0 1 0 1
1 0 0 1 0
3
```

Save this as `cycle5_matrix.txt`, then run:

```bash
./sequential_coloring < cycle5_matrix.txt
```

### OpenMP version

```bash
g++ -O2 -fopenmp openMP/openMP.cpp -o openmp_coloring
OMP_NUM_THREADS=4 ./openmp_coloring < openMP/5.txt
```

The OpenMP program reads the same edge-list layout as the CUDA version.

## Testing and performance analysis

The repository contains:

- [`generate_test_cases.py`](generate_test_cases.py): generates random adjacency-matrix and edge-list inputs using NumPy.
- [`automateTests.py`](automateTests.py): compiles the sequential C implementation and runs the matrix-format inputs in `test_cases/`.
- [`test_cases/`](test_cases/): sample graphs of different sizes, including corresponding edge-list files.
- [`GraphColoringReport.docx`](GraphColoringReport.docx): implementation discussion and course-project timing measurements.

The CUDA program includes a final host-side check that visits each edge and reports whether its two endpoints received the same color.

### Measurements from the course report

The following timings are recorded in the project's final report (milliseconds):

| Vertices | Sequential backtracking | CUDA | OpenMP |
| ---: | ---: | ---: | ---: |
| 10 | 10.00 | 5.00 | 3.69 |
| 100 | 33.00 | 5.73 | 4.53 |
| 200 | 60.00 | 5.86 | 4.28 |
| 500 | 302.00 | 5.49 | 4.75 |
| 1,000 | Did not finish | 6.67 | 8.98 |
| 5,000 | Did not finish | 10.49 | 25.18 |
| 10,000 | Did not finish | 12.58 | 38.29 |

These are the measurements documented during the course project, not a fresh benchmark. The sequential version performs a bounded-color backtracking search, while the parallel versions use iterative heuristics. The timing methods also differ between implementations, so the figures are useful for studying their behavior, but should not be interpreted as a controlled, like-for-like speedup measurement.

## Implementation scope

The CUDA solver is designed for **undirected, loop-free graphs**, particularly graphs with relatively small vertex degrees. The current forbidden-color representation uses 32-bit integer shifts; it is not a general implementation for arbitrarily large color indices.

The random generator can produce duplicate edges or self-loops because it samples endpoints independently. Inputs should be checked before using them to evaluate coloring correctness.

The CUDA program prints a label described as the “minimum number of colors,” but the implementation **does not prove optimality**. This value is the maximum assigned color label; the number of distinct colors in a valid coloring is not necessarily the graph's chromatic number.

## Repository layout

```text
.
├── parallelGraphColouring.cu       # CUDA vertex-based coloring
├── sequentialImplementation.c     # CPU backtracking solver
├── openMP/
│   ├── openMP.cpp                  # OpenMP coloring implementation
│   ├── randomGraph.cpp             # Sample graph generator
│   └── *.txt                       # Edge-list test graphs
├── generate_test_cases.py         # Matrix and edge-list generation
├── automateTests.py               # Sequential test runner
├── test_cases/                    # Input graphs
│   └── adjacencyListTestCases/    # Edge-list representations
├── finalSubmission/               # Submitted source, inputs and notes
├── utility/                       # Experimental/reference CUDA implementations and supporting material
├── GraphColoringReport.docx       # Final project report
├── GraphColouringUsingGPU.pptx    # Course presentation
├── Big Assignment Document.pdf    # Assignment specification
└── Research-Paper.pdf             # Related research paper
```

## References

The CUDA approach was developed after studying:

- Mehmet Deveci, Erik G. Boman, Karen D. Devine, and Sivasankaran Rajamanickam, **“Parallel Graph Coloring for Manycore Architectures,”** *IEEE IPDPS*, 2016. [DOI: 10.1109/IPDPS.2016.54](https://doi.org/10.1109/IPDPS.2016.54)
- Additional implementation examples and the *High-Speed Graph Coloring on GPUs* lecture included under [`utility/`](utility/).

The `utility/` directory contains exploratory and reference material; the main project CUDA implementation is [`parallelGraphColouring.cu`](parallelGraphColouring.cu).
