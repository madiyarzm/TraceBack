import networkx as nx #I want to use the networkx library, and I'll call it nx fro now on to keep things short
import matplotlib.pyplot as plt

#create an empty undirected graph 
G = nx.Graph() #make me one actual graph called 'G' using Graph blueprint. 

# add nodes
G.add_nodes_from(["Doc1", "Doc2", "Doc3", "Doc4", "Doc5"])

# add edges
G.add_edges_from([
    ("Doc1", "Doc2"),
    ("Doc2", "Doc3"),
    ("Doc3", "Doc4"),
    ("Doc4", "Doc5"),
    ("Doc5", "Doc1")
])

#draw the graph
nx.draw(G, with_labels=True) #hey, I want to draw the graph with labels shown
plt.show()

# library: networkx -> collection of tools for grpah theory
# class: graph (from networkx) -> a blueprint for undirected graphs
# object: G = nx.Graph () -> personal graph created using blueprint

# library > class > object
