import networkx as nx 
import matplotlib.pyplot as plt

#additional imports needed
import sys
import os

#this file automatically finds needed scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#now you can directly import other script within this project
#means import function, that can be found in search_module folder -> parsing_script.py
from search_module.parsing_script import get_related_articles

#to test script
if __name__ == "__main__":
    articles = get_related_articles("fargo man flamethrower")
    print(articles)

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