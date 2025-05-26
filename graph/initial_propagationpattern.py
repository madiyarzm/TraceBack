import networkx as nx 
import matplotlib.pyplot as plt

G = nx.DiGraph()

G.add_edges_from([
    ("SiteA", "SiteB"),
    ("SiteA", "SiteC"),
    ("SiteB", "SiteD"),
    ("SiteC", "SiteE"),
    ("SiteD", "SiteE"),
    ("SiteD", "SiteF")    
])

out_degrees = dict(G.out_degree())

node_colors = ["red" if node == "SiteA" else "lightgray" for node in G.nodes()]
node_sizes = [ 300 + out_degrees[node]*400 for node in G.nodes()]
edge_widths = [1 + out_degrees[edge[0]] for edge in G.edges()]
# edge[0] means: the source node of the edge (where arrow starts)

pos = nx.spring_layout(G, seed=42)
nx.draw(
    G, pos,
    with_labels=True,
    node_color=node_colors,
    node_size=node_sizes,
    width=edge_widths,
    arrows=True   
)


plt.title("Fake News Propagation from SiteA")
plt.show()