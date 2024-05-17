/*
 * @author: Abdelaziz Salah
 * @author: Ahmed Hosny
 * @date: 17/5/2024
 */

// Include header files
#include <bits/stdc++.h>
#include <cuda.h>

#define MAX_THREADS 1024
#define CEIL(a, b) (((a - 1) / b) + 1)

using namespace std;

float device_time_taken;

// Catch Cuda errors
void catchCudaError(cudaError_t error, const char *function)
{
    if (error != cudaSuccess)
    {
        printf("\n====== Cuda Error Code %i ======\n %s in CUDA %s\n", error, cudaGetErrorString(error), function);
        exit(-1);
    }
}

// Host Memory
void readGraph(int &nodeCount, int &edgeCount,
               int &maxDegree, int **neighbours, int **srcPointers)
{
    int u, v;
    cin >> nodeCount >> edgeCount;

    // Use vector of vectors temporarily to input graph
    vector<int> *adj = new vector<int>[nodeCount];
    for (int i = 0; i < edgeCount; i++)
    {
        cin >> u >> v;

        /// undirected graph
        /// unweighted graph
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // Copy into compressed adjacency List
    *srcPointers = new int[nodeCount + 1];    /// da bykun el soruce, by2uly kol source btbd2 fen w 3ndha kam neighbour
    *neighbours = new int[2 * edgeCount + 1]; /// howa da el destination, byb2a gowaah el neighbours bto3 kol node
    /*
        ie: CSR representation.
            srcPointers = {0,2,5,7,8} =>
                node1 => has 2 neighbours
                node2 => has 3 neighbours
                node3 => has 2 neighbours
                node4 => has only 1 neighbour
            neighbours = {1,2,0,2,3,0,1,1}
                this mean that neighbours of the first node are 1,2
                neighbours of second node are 0,2,3
                neighbours of third node are 0,1
                neigbours of fourth node are 1
    */
    /// filling the array with the correct nodes.
    int pos = 0;
    for (int i = 0; i < nodeCount; i++)
    {
        (*srcPointers)[i] = pos;
        for (int node : adj[i])
            (*neighbours)[pos++] = node;
    }
    /// we append one extra cell at the end, to indicate the number of neighbours for the last node.
    (*srcPointers)[nodeCount] = pos;

    // Calculate max degree
    /// 34an fl CSR, bnb2a 3auzen ne3rf men aktur row 3ndo non-zero elements.
    maxDegree = INT_MIN;
    for (int i = 0; i < nodeCount; i++)
        maxDegree = max(maxDegree, (int)adj[i].size());

    delete[] adj;
}

/*
    This is a vertex-centric approach
    vertex centric is better for low-degree graphs, and this is our assumption.
*/
__global__ void assignColoursKernel(int nodeCount, int *neighbours, int *srcPointers,
                                    int *device_colours, bool *device_conflicts, int maxDegree)
{

    /// each thread is assigned to one node.
    int node = blockIdx.x * blockDim.x + threadIdx.x;

    /// if the node is out of the graph, or it is not conflicting, then return.
    if (node >= nodeCount || !device_conflicts[node])
        return;

    /// 3dd el alwan el hn7tagha, msh hyb2a aktur mn el maxDegree,
    int maxColours = maxDegree + 1;

    // Create forbidden array of size maxDegree
    int *forbidden = new int[CEIL(maxColours + 1, 32)];
    if (forbidden == NULL)
    {
        printf("Cuda Memory Full\n");
        return;
    }

    /// initialize the forbidden array with zeros.
    memset(forbidden, 0, sizeof(int) * CEIL(maxColours + 1, 32));

    /// following the top-down approach with CSRGraph representation like in BFS ...
    /// mehtag afhm el loop de.
    for (int i = srcPointers[node]; i < srcPointers[node + 1]; i++)
    {
        int neighbour = neighbours[i];
        int ind = device_colours[neighbour] % 32; /// the stride.
        forbidden[device_colours[neighbour] / 32] |= (1 << ind);
    }

    /// assign the first available color to the node, to minimize the number of used colours.
    for (int colour = 1; colour <= maxColours; ++colour)
    {
        int ind = colour % 32;
        if ((forbidden[colour / 32] & (1 << ind)) == 0)
        {
            device_colours[node] = colour;
            break;
        }
    }

    delete[] forbidden;
}

void assignColours(int nodeCount, int *neighbours, int *srcPointers,
                   int *device_colours, bool *device_conflicts, int maxDegree)
{

    // Launch assignColoursKernel with nodeCount number of threads
    assignColoursKernel<<<CEIL(nodeCount, MAX_THREADS), MAX_THREADS>>>(nodeCount, neighbours, srcPointers,
                                                                       device_colours, device_conflicts, maxDegree);
    cudaDeviceSynchronize();
}


/*
    This is a vertex-centric approach
*/
__global__ void detectConflictsKernel(int nodeCount, int *neighbours, int *srcPointers,
                                      int *device_colours, bool *device_conflicts, bool *device_conflictExists)
{

    /// each thread is assigned to one node.
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= nodeCount)
        return;

    /// assume no conflicts
    device_conflicts[node] = false;

    /// following the top-down approach with CSRGraph representation like in BFS ...
    /// this is the CSR representation of the graph
    for (int edge = srcPointers[node]; edge < srcPointers[node + 1]; edge++)
    {
        int neighbour = neighbours[edge];

        /// if both me and my neighbour has the same color, then there is a conflict.
        if (device_colours[neighbour] == device_colours[node] && neighbour < node)
        {
            // conflict
            device_conflicts[node] = true;

            /// we should apply another iteration.
            *device_conflictExists = true;
        }
    }
}

bool detectConflicts(int nodeCount, int *neighbours, int *srcPointers,
                     int *device_colours, bool *device_conflicts)
{

    bool *device_conflictExists;
    bool conflictExists = false;

    catchCudaError(cudaMalloc((void **)&device_conflictExists, sizeof(bool)), "Malloc1");
    catchCudaError(cudaMemcpy(device_conflictExists, &conflictExists, sizeof(bool), cudaMemcpyHostToDevice), "Memcpy7");

    // Launch detectConflictsKernel with nodeCount number of threads
    detectConflictsKernel<<<CEIL(nodeCount, MAX_THREADS), MAX_THREADS>>>(nodeCount, neighbours, srcPointers, device_colours, device_conflicts, device_conflictExists);
    cudaDeviceSynchronize();

    // Copy device_conflictExists to conflictExists and return
    catchCudaError(cudaMemcpy(&conflictExists, device_conflictExists, sizeof(bool), cudaMemcpyDeviceToHost), "Memcpy6");

    // Free device memory
    catchCudaError(cudaFree(device_conflictExists), "Free");

    return conflictExists;
}

int *graphColouring(int nodeCount, int *device_neighbours, int *device_srcPointers,
                    int maxDegree)
{

    // Boolean array for conflicts
    bool *host_conflicts = new bool[nodeCount];
    int *host_colours = new int[nodeCount];
    int *device_colours;
    bool *device_conflicts;

    // Initialize all nodes to invalid colour (0)
    memset(host_colours, 0, sizeof(int) * nodeCount);

    // Initialize all nodes into conflict
    memset(host_conflicts, true, sizeof(bool) * nodeCount);

    catchCudaError(cudaMalloc((void **)&device_colours, sizeof(int) * nodeCount), "Malloc2");
    catchCudaError(cudaMemcpy(device_colours, host_colours, sizeof(int) * nodeCount, cudaMemcpyHostToDevice), "Memcpy1");
    catchCudaError(cudaMalloc((void **)&device_conflicts, sizeof(bool) * nodeCount), "Malloc3");
    catchCudaError(cudaMemcpy(device_conflicts, host_conflicts, sizeof(bool) * nodeCount, cudaMemcpyHostToDevice), "Memcpy2");

    // Timer
    cudaEvent_t device_start, device_end;
    catchCudaError(cudaEventCreate(&device_start), "Event Create");
    catchCudaError(cudaEventCreate(&device_end), "Event Create");
    catchCudaError(cudaEventRecord(device_start), "Event Record");

    do
    {
        assignColours(nodeCount, device_neighbours, device_srcPointers, device_colours, device_conflicts, maxDegree);
    } while (detectConflicts(nodeCount, device_neighbours, device_srcPointers, device_colours, device_conflicts));

    // Timer
    catchCudaError(cudaEventRecord(device_end), "Event Record");
    catchCudaError(cudaEventSynchronize(device_end), "Event Synchronize");
    catchCudaError(cudaEventElapsedTime(&device_time_taken, device_start, device_end), "Elapsed time");

    // Copy colours to host and return
    catchCudaError(cudaMemcpy(host_colours, device_colours, sizeof(int) * nodeCount, cudaMemcpyDeviceToHost), "Memcpy3");

    delete[] host_conflicts;
    catchCudaError(cudaFree(device_colours), "Free");
    catchCudaError(cudaFree(device_conflicts), "Free");

    return host_colours;
}

int main(int argc, char *argv[])
{

    if (argc < 2)
    {
        cout << " Wrong number of arguments\n";
        cout << "Usage: " << argv[0] << " <graph_input_file> [output_file]\n";
        return 0;
    }

    int nodeCount, edgeCount, maxDegree;
    int *neighbours = NULL, *srcPointers = NULL;
    int *device_neighbours, *device_srcPointers;

    char choice = 'y';
    // cout << "Would you like to print the colouring of the graph? (y/n) ";
    // cin >> choice;

    freopen(argv[1], "r", stdin);
    readGraph(nodeCount, edgeCount, maxDegree, &neighbours, &srcPointers);

    // Alocate device memory and copy
    /// 2*edgeCount 34an bidirectional?
    catchCudaError(cudaMalloc((void **)&device_neighbours, sizeof(int) * (2 * edgeCount + 1)), "Malloc5");
    catchCudaError(cudaMemcpy(device_neighbours, neighbours, sizeof(int) * (2 * edgeCount + 1), cudaMemcpyHostToDevice), "Memcpy11");

    // Alocate device memory and copy
    catchCudaError(cudaMalloc((void **)&device_srcPointers, sizeof(int) * (nodeCount + 1)), "Malloc6");
    catchCudaError(cudaMemcpy(device_srcPointers, srcPointers, sizeof(int) * (nodeCount + 1), cudaMemcpyHostToDevice), "Memcpy12");

    int *colouring = graphColouring(nodeCount, device_neighbours, device_srcPointers, maxDegree);

    int chromaticNumber = INT_MIN;
    for (int i = 0; i < nodeCount; i++)
    {
        chromaticNumber = max(chromaticNumber, colouring[i]);
        if (choice == 'y' || choice == 'Y')
            printf("Node %d => Colour %d\n", i, colouring[i]);
    }
    cout << endl;
    printf("\nNumber of colours used (chromatic number) ==> %d\n", chromaticNumber);
    printf("Time Taken (Parallel) = %f ms\n", device_time_taken);

    /// writing the result into output file
    if (argc == 3)
    {
        freopen(argv[2], "w", stdout);
        for (int i = 0; i < nodeCount; i++)
            cout << colouring[i] << " ";
        cout << endl;
    }

    // Check for correctness
    int count = 0;
    for (int u = 0; u < nodeCount; u++)
    {
        for (int j = srcPointers[u]; j < srcPointers[u + 1]; j++)
        {
            int v = neighbours[j];
            if (colouring[u] == colouring[v] && u <= v)
            {
                if (count <= 10)
                    cout << "Conflicting color found at nodes " << u << " and " << v << endl;
                count++;
            }
        }
    }
    cout << "Found total " << count << " conflicts!" << endl;

    // Free all memory
    delete[] colouring, neighbours, srcPointers;
    catchCudaError(cudaFree(device_neighbours), "Free");
    catchCudaError(cudaFree(device_srcPointers), "Free");
}