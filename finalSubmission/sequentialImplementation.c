#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h>
#include <time.h>
// Number of vertices in the graph
int V;

// Number of edges

void printSolution(int color[])
{
    printf("Solution Exists:"
           " Following are the assigned colors \n");
    for (int i = 0; i < V; i++)
        printf(" %d ", color[i]);
    printf("\n");
}

/// O(N), where N is the number of nodes.
bool checkNeighbours(int **graph, const int *colours, int numberOfNodes, int currentNode, int currentColour)
{
    /// iterate over all my neighbours
    for (int i = 0; i < numberOfNodes; i++)
        /// if cell == 1, then we are neighbours, check if the color of this neighbour is the same as mine
        if (graph[currentNode][i] == 1 && colours[i] == currentColour)
            /// if so, then this colour is not valid.
            return false;

    return true;
}

bool graphColouring(const int **graph, int *colours, bool *visited, const int numberOfNodes, 
                            const int numberOfColours, int numberOfColouredNodes, int currentNode)
{
    /// base case
    if (numberOfColouredNodes == numberOfNodes)
        return true;

    if (visited[currentNode] == true || colours[currentNode] != 0)
        return false;

    visited[currentNode] = true;
    /// now I need to iterate over all the colours, and check if the neighbours does not have this colour, so we
    /// can assign this colour for me.
    for (int c = 1; c <= numberOfColours; c++)
        /// true means no neighbour has this colour.
        if (checkNeighbours(graph, colours, numberOfNodes, currentNode, c))
        {
            colours[currentNode] = c;
            if (graphColouring(graph, colours, visited, numberOfNodes, numberOfColours, numberOfColouredNodes + 1, currentNode + 1))
                /// here it is not a must that we should visit the neighbours in order like dfs, we can access nodes in any order
                /// and we would get the same result assuming that the nodes are marked from 0 to N-1
                /// so when I try to visit another node, I can just visit the node number n + 1.
                return true;

            /// apply backtracking, and try another colour.
            colours[currentNode] = 0;
        }

    /// back tracking
    visited[currentNode] = false;
    return false;
}


int main()
{
    clock_t start, end;
    double cpu_time_used;

    // printf("Enter Number of vertices\n");
    scanf("%d", &V);
    // printf("Enter Number of edges\n");
    // scanf("%d", &V);
    int **graph = (int **)malloc(V * sizeof(int *));
    if (graph == NULL)
    {
        printf("Memory allocation failed\n");
        return -1;
    }

    for (int i = 0; i < V; i++)
    {
        graph[i] = (int *)malloc(V * sizeof(int));
        if (graph[i] == NULL)
        {
            printf("Memory allocation failed\n");
            return -1;
        }
    }

    // printf("Enter Adjacency Matrix\n");
    for (int i = 0; i < V; i++)
        for (int j = 0; j < V; j++)
            scanf("%d", &graph[i][j]);

    // printf("Enter Number of colours\n");
    int c;

    // Number of colors
    scanf("%d", &c);

    /// allocating memory for color array
    int *color = (int *)malloc(V * sizeof(int));

    /// initializing the color for each vertex as 0.
    for (int i = 0; i < V; i++)
        color[i] = 0;

    /// create a visited array
    bool *visited = (bool *)malloc(V * sizeof(bool));
    for (int i = 0; i < V; i++)
        visited[i] = false;

    start = clock();
    if (graphColouring(graph, color, visited, V, c, 0, 0) == true)
    {
        printSolution(color);
        /// free the memory
        free(color);
        free(visited);
        // Free the allocated memory
        for (int i = 0; i < V; i++)
            free(graph[i]);

        free(graph);
    end = clock();
    cpu_time_used = ((double)(end - start)) * 1000 / CLOCKS_PER_SEC ;
    printf("Time taken: %f ms\n", cpu_time_used);
        return 0;
    }
    printf("Solution does not exist\n");
    // Free the allocated memory
    for (int i = 0; i < V; i++)
        free(graph[i]);

    free(graph);
    free(color);
    free(visited);
    return 0;
}
