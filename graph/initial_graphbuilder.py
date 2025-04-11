# hi 
import networkx as nx
import matplotlib.pyplot as plt

# Create an undirected graph (like Facebook friendship)
G = nx.Graph()

# Add nodes (people)
G.add_nodes_from(["Alice", "Bob", "Charlie", "Diana"])

# Add edges (friendships)
G.add_edges_from([
    ("Alice", "Bob"),
    ("Alice", "Charlie"),
    ("Bob", "Diana")
])

# Draw the graph
plt.figure(figsize=(6,4))
nx.draw(G, with_labels=True, node_color="lightblue", node_size=1000, font_weight="bold")
plt.title("Undirected Friendship Graph")
plt.show()