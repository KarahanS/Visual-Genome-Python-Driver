import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def analyze_connectivity(comparison_matrix):
    # Create a directed graph from the comparison matrix
    G = nx.DiGraph()
    n = comparison_matrix.shape[0]

    # Add edges where there are comparisons
    for i in range(n):
        for j in range(n):
            if comparison_matrix[i, j] > 0:
                G.add_edge(i, j, weight=comparison_matrix[i, j])

    # Find strongly connected components
    components = list(nx.strongly_connected_components(G))

    print(f"Number of strongly connected components: {len(components)}")
    for i, comp in enumerate(components):
        print(f"\nComponent {i+1} size: {len(comp)}")
        print(f"Nodes in component {i+1}: {sorted(list(comp))}")

    # Visualize the comparison graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)

    # Draw nodes, colored by component
    colors = plt.cm.rainbow(np.linspace(0, 1, len(components)))
    for comp, color in zip(components, colors):
        nx.draw_networkx_nodes(
            G, pos, nodelist=list(comp), node_color=[color], node_size=100
        )

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2)

    plt.title("Comparison Graph\n(Colors indicate strongly connected components)")
    plt.axis("off")
    plt.show()

    return components


# Print where we have zeros in both directions (no comparisons at all)
def find_uncompared_pairs(comparison_matrix):
    n = comparison_matrix.shape[0]
    uncompared = []
    for i in range(n):
        for j in range(i + 1, n):
            if comparison_matrix[i, j] == 0 and comparison_matrix[j, i] == 0:
                uncompared.append((i, j))

    print(f"\nNumber of uncompared pairs: {len(uncompared)}")
    if len(uncompared) < 20:  # Only print if not too many
        print("Uncompared pairs (indices):", uncompared)


# Analyze the matrix
print("Analyzing comparison matrix connectivity...")
# read comparison_matrix.npy
comparison_matrix = np.load("comparison_matrix.npy")
components = analyze_connectivity(comparison_matrix)
find_uncompared_pairs(comparison_matrix)

# Additional statistics
print("\nMatrix statistics:")
print(f"Shape: {comparison_matrix.shape}")
print(f"Total comparisons: {np.sum(comparison_matrix)}")
print(f"Non-zero entries: {np.count_nonzero(comparison_matrix)}")
print(
    f"Zero entries: {np.size(comparison_matrix) - np.count_nonzero(comparison_matrix)}"
)
