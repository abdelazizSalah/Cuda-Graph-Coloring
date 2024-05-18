import numpy as np

# Number of nodes and edges
num_nodes = 10
num_edges = 20

# Generate random edges
edges = np.random.choice(num_nodes, size=(num_edges, 2))

# Initialize adjacency matrix with zeros
adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

# Fill adjacency matrix based on edges
for edge in edges:
    node1, node2 = edge
    adj_matrix[node1, node2] = 1
    adj_matrix[node2, node1] = 1  # Undirected graph, so the matrix is symmetric

# Generate adjacency list
adj_list = [[] for _ in range(num_nodes)]
for edge in edges:
    node1, node2 = edge
    adj_list[node1].append(node2)
    adj_list[node2].append(node1)  # Undirected graph, so we append both nodes

# Print adjacency matrix
print("Adjacency Matrix:")
with open('test_cases/adjacencyListTestCases/auto.txt', 'w') as f:
    # f.write("Adjacency Matrix:\n")
    # np.savetxt(f, adj_matrix, fmt='%d')
    f.write(f"{num_nodes} {num_edges}")
    for row in adj_matrix:
        for val in row:
            f.write(f" {val}")
        f.write("\n")

# Print adjacency list
print("\nAdjacency List:")

with open('test_cases/auto.txt', 'w') as f:
    f.write(f"{num_nodes}")
    for i, neighbors in enumerate(adj_list):
        for nei in neighbors:
            f.write(f" {i} {nei}")
print("use the number of colors generated in the parallel")
