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

nx.draw(G, with_labels=True, arrows=True)
plt.show()