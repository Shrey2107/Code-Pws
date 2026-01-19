import pandas as pd
import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import csv
from itertools import combinations
import pulp

# =============================================================
# STEP 0 — Load Excel adjacency matrix 
# =============================================================
FILE = "Graaf test 4.xlsx"

df = pd.read_excel(
    r"C:\Users\Shrey\OneDrive\Documents\PWS\Graaf test 4.xlsx",
    index_col=0
)

df.index = df.index.astype(str)
df.columns = df.columns.astype(str)

# Symmetrize adjacency matrix
df = (df + df.T) / 2

nodes = list(df.index)
n = len(nodes)
print(f"Loaded {n}×{n} adjacency matrix.")

# Build undirected multigraph
G = nx.MultiGraph()
G.add_nodes_from(nodes)

# Add edges only once per undirected pair
for i, u in enumerate(nodes):
    for j, v in enumerate(nodes):
        if j <= i:
            continue
        w = df.loc[u, v]
        if w != 0:
            G.add_edge(u, v, weight=w)

print("Original edges:", G.number_of_edges())


# =============================================================
# STEP 1 — Minimum Dominating Set
# =============================================================
R = 500  # zendbereik, pas aan indien nodig

# Bereken all-pairs shortest paths
print("Computing all-pairs shortest paths...")
dist = dict(nx.floyd_warshall(G, weight="weight"))

# Bouw bereik-graaf
G_reach = nx.Graph()
G_reach.add_nodes_from(G.nodes())
for u, v in combinations(G.nodes(), 2):
    if dist[u][v] <= R:
        G_reach.add_edge(u, v)

print("Aantal edges in G_reach:", G_reach.number_of_edges())

# ILP voor Minimum Dominating Set
print("Solving ILP for optimal Minimum Dominating Set...")
prob = pulp.LpProblem("Minimum_Dominating_Set", pulp.LpMinimize)
x = {v: pulp.LpVariable(f"x_{v}", cat="Binary") for v in G_reach.nodes()}

# Objective: minimaliseer aantal zendmasten
prob += pulp.lpSum(x[v] for v in G_reach.nodes())

# Constraints: elke knoop moet gedekt zijn
for v in G_reach.nodes():
    prob += pulp.lpSum(x[u] for u in [v] + list(G_reach.neighbors(v))) >= 1

# Los het ILP probleem op
prob.solve(pulp.PULP_CBC_CMD(msg=True))

# Haal de oplossing eruit
dominating_set = [v for v in G_reach.nodes() if x[v].varValue == 1]
print("Aantal zendmasten (ILP):", len(dominating_set))
print(dominating_set)

# =============================================================
# STEP 2 — Shortest paths (Floyd–Warshall)
# =============================================================
print("Computing all-pairs shortest paths (Floyd–Warshall)...")
dist = dict(nx.floyd_warshall(G, weight="weight"))

# =============================================================
# STEP 3 — Find odd-degree nodes
# =============================================================
odd_nodes = [v for v in G.nodes() if G.degree(v) % 2 == 1]
print("Odd-degree nodes:", len(odd_nodes))

# =============================================================
# STEP 4 — Minimum-weight perfect matching (CORRECT)
# =============================================================
print("Computing minimum-weight perfect matching...")

K = nx.Graph()
for u, v in combinations(odd_nodes, 2):
    K.add_edge(u, v, weight=dist[u][v])

matching = nx.min_weight_matching(K)

print("Matched pairs:", matching)

# =============================================================
# STEP 5 — Add shortest paths for each matched pair
# =============================================================
print("Duplicating edges along matched shortest paths...")

for u, v in matching:
    path = nx.shortest_path(G, u, v, weight="weight")
    for a, b in zip(path[:-1], path[1:]):
        G.add_edge(a, b, weight=df.loc[a, b])

print("After balancing, edges:", G.number_of_edges())

# =============================================================
# Debug: Verify graph is Eulerian
# =============================================================
odd_after = [v for v in G.nodes() if G.degree(v) % 2 == 1]
print("Odd nodes AFTER balancing:", odd_after)
print("Is Eulerian now?", nx.is_eulerian(G))

# If still not Eulerian → something is wrong in input data
if len(odd_after) != 0:
    raise ValueError("Graph is still not Eulerian. Something is wrong with the input graph.")

# =============================================================
# STEP 6 — Eulerian circuit
# =============================================================
print("Computing Eulerian tour...")
tour = list(nx.eulerian_circuit(G, keys=True))

total_cost = 0
for u, v, key in tour:
    total_cost += G[u][v][key]["weight"]

print("Total minimal CPP cost:", total_cost)
print("Number of edges in tour:", len(tour))

# =============================================================
# STEP 7 — Write solution
# =============================================================
output_file = "CPP_solution.csv"

with open(output_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Step", "From", "To", "Weight"])
    
    for i, (u, v, key) in enumerate(tour, start=1):
        writer.writerow([i, u, v, G[u][v][key]["weight"]])

    writer.writerow([])
    writer.writerow(["Total Cost", total_cost])

print(f"Solution written to {output_file}")

# =============================================================
# STEP 8 — Visualization
# =============================================================
print("Generating visualization...")

plt.figure(figsize=(14, 14))
pos = nx.spring_layout(G, seed=42)

nx.draw_networkx_nodes(G, pos, node_size=80, node_color="lightblue")
nx.draw_networkx_edges(G, pos, alpha=0.2)

nx.draw_networkx_nodes(
    G, pos,
    nodelist=dominating_set,
    node_color="orange",
    node_size=200,
    label="Zendmasten"
)


for u, v, key in tour:
    nx.draw_networkx_edges(
        G, pos,
        edgelist=[(u, v)],
        width=2.5,
        edge_color="red"
    )

nx.draw_networkx_labels(G, pos, font_size=6)

plt.title("Chinese Postman Optimal Tour")
plt.axis("off")
plt.tight_layout()
plt.show()
