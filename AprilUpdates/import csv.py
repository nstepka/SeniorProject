import csv
import networkx as nx
from pyvis.network import Network

file_name = 'causal-graph-2023-04-12.csv'

# Create a new graph
G = nx.DiGraph()

# Read the CSV file
with open(file_name, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        source = row['source']
        target = row['target']
        relationship = row['relationship']

        if row['weight'] == '':
            weight = 1.0
        else:
            weight = float(row['weight'])

        G.add_edge(source, target, weight=weight, label=relationship)

# Create a pyvis network
net = Network(notebook=True)
net.from_nx(G)

# Customize node and edge appearance
net.toggle_physics(False)
for node in net.nodes:
    node['color'] = 'skyblue'
    node['size'] = 30
    node['font'] = {'color': 'black', 'size': 12}

for edge in net.edges:
    edge['color'] = '#888'
    edge['width'] = 1.0
    edge['label'] = G.edges[edge['from'], edge['to']]['label']

# Show the interactive network visualization
net.show('causal_graph.html')
