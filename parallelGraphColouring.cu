/*
 * @author: Abdelaziz Salah
 * @author: Ahmed Hosny
 * @date: 30/4/2024
 */

// Include header files
#include <bits/stdc++.h>
#include <cuda.h>

#define MAX_THREADS 1024
/// (n-1) / total size + 1 equavilent to the ceil operation.
#define CEIL(a, b) (((a - 1) / b) + 1)
#define DEBUG 1
using namespace std;

float device_time_taken;

// Catch Cuda errors
void catchCudaError(cudaError_t error, const char *operation_name)
{
    if (error != cudaSuccess)
    {
        printf("\n====== Cuda Error Code %i ======\n %s in CUDA %s\n", error, cudaGetErrorString(error), operation_name);
        exit(-1);
    }
}

// Host Memory
void readGraphInCSRFormat(int &nodeCount, int &edgeCount,
                          int &maxDegree, int **neighbours, int **srcPointers)
{
    cin >> nodeCount >> edgeCount;

    // read the vector as adjacency list, temporarly then remove it
    vector<int> *adj = new vector<int>[nodeCount];
    int src, dest;
    for (int i = 0; i < edgeCount; i++)
    {
        cin >> src >> dest;

        /// undirected graph
        /// unweighted graph
        adj[src].push_back(dest);
        adj[dest].push_back(src);
    }

    // Copy into compressed adjacency List
    /*
        2 * edgeCount + 1, this is because:
            1. the graph is symmetric
            2. the graph in bidirectional
            3. so when we say A-B, we need to reserve 2 cells,
                one for A having B as its neighbour
                and one for B having A as its neigbour.

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
    *srcPointers = new int[nodeCount + 1];    /// da bykun el soruce, by2uly kol source btbd2 fen w 3ndha kam neighbour
    *neighbours = new int[2 * edgeCount + 1]; /// howa da el destination, byb2a gowaah el neighbours bto3 kol node

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
    /*
        34an fl CSR, bnb2a 3auzen ne3rf men aktur row 3ndo non-zero elements.
        w kman 34an according to By Vizing's theorem,
        the number of colors needed to color a simple graph is either its maximum degree Δ or Δ+1.
        It is gaurenteed if we have number of colours = to maxDegree + 1, we will find a solution
    */
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

    /// Create forbidden array of size maxDegree
    /// this array indicates the forbidden colour, and I get them from my neighbours.
    /// we use the bit-wise representation to save the memory, so we represent it in 32 bits.
    /// this also apply the privitization concept.
    // int noOfColors = CEIL(maxColours + 1, 32);
    int noOfColors = maxColours + 1;
    int *forbidden = new int[noOfColors];
    if (forbidden == NULL)
    {
        printf("Cuda Memory Full\n");
        return;
    }

    /// initialize the forbidden array with zeros.
    memset(forbidden, 0, sizeof(int) * noOfColors);

    /// following the top-down approach with CSRGraph representation like in BFS ...
    /// mehtag afhm el loop de.
    for (int i = srcPointers[node]; i < srcPointers[node + 1]; i++)
    {
        /// get the neighbour
        int neighbour = neighbours[i];

        /// get the neighbour color
        int neiColor = device_colours[neighbour];

        /// get its color and convert its index into the 32 bit representation.
        // int ind =  neiColor % 32; /// the stride.
        int ind = neiColor; /// the stride.

        /// set this color as forbidden
        forbidden[neiColor] |= (1 << ind);
    }

    /// assign the first available color to the node, to minimize the number of used colours.
    for (int colour = 1; colour <= maxColours; ++colour)
    {
        // int ind = colour % 32;
        int ind = colour;
        /// if not forbidden, assign our color as this color.
        if ((forbidden[colour] & (1 << ind)) == 0)
        {
            device_colours[node] = colour;
            break;
        }
    }

    delete[] forbidden;
}

void assignColours(int nodeCount, int *neighbours, int *srcPointers,
                   int *device_colours, bool *device_conflicts, int maxDegree, cudaStream_t stream)
{

    // Launch assignColoursKernel with nodeCount number of threads
    /*
        Ceil of node count with Max_Threads, because we know that the one block can have up to 1024 threads.
        so we just define # of blocks to can handle the number of nodes in the array.
    */
    assignColoursKernel<<<CEIL(nodeCount, MAX_THREADS), MAX_THREADS, 0, stream>>>(nodeCount, neighbours, srcPointers,
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

            /// continue to detect all conflicts, because we can work only on conflicting nodes, and this will save time.
        }
    }
}

bool detectConflicts(int nodeCount, int *neighbours, int *srcPointers,
                     int *device_colours, bool *device_conflicts, cudaStream_t stream)
{

    /// this flag indicates that there are still some conflicts
    bool *device_conflictExists;
    bool conflictExists = false;

    catchCudaError(cudaMalloc((void **)&device_conflictExists, sizeof(bool)), "Malloc5");
    catchCudaError(cudaMemcpy(device_conflictExists, &conflictExists, sizeof(bool), cudaMemcpyHostToDevice), "Memcpy5");

    // Launch detectConflictsKernel with nodeCount number of threads
    detectConflictsKernel<<<CEIL(nodeCount, MAX_THREADS), MAX_THREADS, 0, stream>>>(nodeCount, neighbours, srcPointers, device_colours, device_conflicts, device_conflictExists);
    cudaDeviceSynchronize();

    // Copy device_conflictExists to conflictExists and return
    catchCudaError(cudaMemcpy(&conflictExists, device_conflictExists, sizeof(bool), cudaMemcpyDeviceToHost), "Memcpy6");

    // Free device memory
    catchCudaError(cudaFree(device_conflictExists), "Free");

    return conflictExists;
}

/*
    This function returns the colours of the nodes.
*/
int *graphColouring(const int &nodeCount, int *device_neighbours, int *device_srcPointers,
                    const int &maxDegree)
{

    /*
        These two arrays:
            conflicts: if a node has a conflict color with one of its neighbours, 0 indicates invalid colour.
            colours: the colour of each node.
    */
    // bool *host_conflicts = new bool[nodeCount];
    // int *host_colours = new int[nodeCount];

    /// pinned memory
    bool *host_conflicts;
    int *host_colours;
    catchCudaError(cudaMallocHost((void **)&host_conflicts, sizeof(bool) * nodeCount), "MallocHost1");
    catchCudaError(cudaMallocHost((void **)&host_colours, sizeof(int) * nodeCount), "MallocHost2");

    int *device_colours;
    bool *device_conflicts;

    // Initialize all nodes to invalid colour (0)
    memset(host_colours, 0, sizeof(int) * nodeCount);

    // Initialize all nodes into conflict
    memset(host_conflicts, true, sizeof(bool) * nodeCount);

    //! Applying streaming :)
    /// 1. define the number of streams needed
    unsigned int numStreams = 8; // this number can be changed later.

    /// 2. create an array of streams
    cudaStream_t streams[numStreams];

    /// 3. allocate them
    for (unsigned int s = 0; s < numStreams; s++)
        cudaStreamCreate(&streams[s]);

    /// 4. stream the segments
    unsigned int numSegments = numStreams;
    unsigned int segmentSize = CEIL(nodeCount, numSegments);

    //? we need to avoid duplicate allocation, later I will insert this inside the function, and make the allocation
    //? depends on the segment, but for now, lets keep it outside
    catchCudaError(cudaMalloc((void **)&device_colours, sizeof(int) * nodeCount), "Malloc3");
    catchCudaError(cudaMalloc((void **)&device_conflicts, sizeof(bool) * nodeCount), "Malloc4");
    /// 5. loop over the segments
    for (unsigned int s = 0; s < numSegments; s++)
    {
        /// to avoid exceeding the memory, and causing errors.
        if (s * segmentSize >= nodeCount)
            break;

        /// 6. Find the segment bound
        unsigned int startSegmentIdx = s * segmentSize;

        /// we must make sure that we have handeled the last segment correctly,
        /// so if the start + segment size > nodeCount, just make the end = nodeCount
        /// this implies that all segments can have same size except the last which can have smaller size.
        unsigned int endSegmentIdx = (startSegmentIdx + segmentSize < nodeCount) ? startSegmentIdx + segmentSize : nodeCount;

        /// get the segment size
        unsigned int Nsegment = endSegmentIdx - startSegmentIdx; // + 1 => ignore the last element.

        /// device_colours[startSegmentIdx] means that I want to startSegmentIdx copying from the address of the segment
        catchCudaError(cudaMemcpyAsync(&device_colours[startSegmentIdx], &host_colours[startSegmentIdx], sizeof(int) * Nsegment, cudaMemcpyHostToDevice, streams[s]), "Memcpy3");

        catchCudaError(cudaMemcpyAsync(&device_conflicts[startSegmentIdx], &host_conflicts[startSegmentIdx], sizeof(bool) * Nsegment, cudaMemcpyHostToDevice, streams[s]), "Memcpy4");

        // Timer
        cudaEvent_t device_start, device_end;
        catchCudaError(cudaEventCreate(&device_start), "Event Create");
        catchCudaError(cudaEventCreate(&device_end), "Event Create");
        catchCudaError(cudaEventRecord(device_start), "Event Record");

        /// keep trying to assign colours as long as there are conflicts to detect.
        do
        {
            assignColours(Nsegment, device_neighbours, device_srcPointers, &device_colours[startSegmentIdx], &device_conflicts[startSegmentIdx], maxDegree, streams[s]);
        } while (detectConflicts(Nsegment, device_neighbours, device_srcPointers, &device_colours[startSegmentIdx], &device_conflicts[startSegmentIdx], streams[s]));

        // Timer
        catchCudaError(cudaEventRecord(device_end), "Event Record");
        catchCudaError(cudaEventSynchronize(device_end), "Event Synchronize");
        catchCudaError(cudaEventElapsedTime(&device_time_taken, device_start, device_end), "Elapsed time");

        // Copy colours to host and return
        catchCudaError(cudaMemcpyAsync((void **)&host_colours[startSegmentIdx], &device_colours[startSegmentIdx], sizeof(int) * Nsegment, cudaMemcpyDeviceToHost, streams[s]), "Memcpy7");
    }
    cudaDeviceSynchronize();
    /// n. destroy streams
    // for (unsigned int s = 0; s < numStreams; s++)
    //     cudaStreamDestroy(streams[s]);

    delete[] host_conflicts;
    catchCudaError(cudaFree(device_colours), "Free");
    catchCudaError(cudaFree(device_conflicts), "Free");

    // should free the memory
    catchCudaError(cudaFreeHost(host_conflicts), "FreeHost1");
    catchCudaError(cudaFreeHost(host_colours), "FreeHost2");

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

    /// 1. define the variables
    int nodeCount, edgeCount, maxDegree;

    /// these arrays for the GSR representation of the graph
    //? Note that we do not need the value array, because we are working with unweighted graph, all ones.
    int *neighbours = NULL, *srcPointers = NULL;

    /// these for the GPU memory
    int *device_neighbours, *device_srcPointers;

    /// 2. reading the graph.
    freopen(argv[1], "r", stdin);
    readGraphInCSRFormat(nodeCount, edgeCount, maxDegree, &neighbours, &srcPointers);

    // 3. Alocate device memory and copy them there.
    /*
        2 * edgeCount + 1, this is because:
            1. the graph is symmetric
            2. the graph in bidirectional
            3. so when we say A-B, we need to reserve 2 cells,
                one for A having B as its neighbour
                and one for B having A as its neigbour.
    */
    catchCudaError(cudaMalloc((void **)&device_neighbours, sizeof(int) * (2 * edgeCount + 1)), "Malloc1");
    catchCudaError(cudaMemcpy(device_neighbours, neighbours, sizeof(int) * (2 * edgeCount + 1), cudaMemcpyHostToDevice), "Memcpy1");

    // Alocate device memory and copy
    /// +1 to know how many neighbours for the last node.
    catchCudaError(cudaMalloc((void **)&device_srcPointers, sizeof(int) * (nodeCount + 1)), "Malloc2");
    catchCudaError(cudaMemcpy(device_srcPointers, srcPointers, sizeof(int) * (nodeCount + 1), cudaMemcpyHostToDevice), "Memcpy2");

    /// 4. call the graph coloring function, which calls the kernel.
    int *colouring = graphColouring(nodeCount, device_neighbours, device_srcPointers, maxDegree);

    /// 5. printing the results if there is a result.
    int chromaticNumber = INT_MIN;
    for (int i = 0; i < nodeCount; i++)
    {
        chromaticNumber = max(chromaticNumber, colouring[i]);
        printf("Node %d => Colour %d\n", i, colouring[i]);
    }
    cout << endl;
    printf("\nminimum number of colors is:  ==> %d\n", chromaticNumber);
    printf("Time Taken = %f ms\n", device_time_taken);

    /// writing the result into output file
    if (argc == 3)
    {
        freopen(argv[2], "w", stdout);
        for (int i = 0; i < nodeCount; i++)
            cout << colouring[i] << " ";
        cout << endl;
    }

    if (DEBUG)
    {
        // Check for correctness
        /// just to make sure that our kernal is working well
        int count = 0;
        for (int u = 0; u < nodeCount; u++)
        {
            for (int j = srcPointers[u]; j < srcPointers[u + 1]; j++)
            {
                int v = neighbours[j];
                if (colouring[u] == colouring[v] && u <= v)
                {
                    // if (count <= 10)
                    cout << "Conflicting color found at nodes " << u << " and " << v << endl;
                    count++;
                }
            }
        }
        cout << "Found total " << count << " conflicts!" << endl;
    }

    /// 6. Free all memory
    delete[] colouring, neighbours, srcPointers;
    catchCudaError(cudaFree(device_neighbours), "Free");
    catchCudaError(cudaFree(device_srcPointers), "Free");
}