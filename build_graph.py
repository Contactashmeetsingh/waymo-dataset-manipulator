import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from extract_features import x, y, vx, vy, speed

# 1️⃣ Select a focused subset (a "local scene" instead of first N points)
# This avoids plotting random scattered cars.
# We'll select the first 300 points, then pick 15 that are close together.
subset_size = 300
subset_positions = np.stack([x[:subset_size], y[:subset_size]], axis=1)

# Compute pairwise distances from the first agent
distances = np.linalg.norm(subset_positions - subset_positions[0], axis=1)
close_indices = np.argsort(distances)[:15]  # 15 nearest vehicles

# 2️⃣ Extract their data
positions = subset_positions[close_indices]
vx_subset = vx[close_indices]
vy_subset = vy[close_indices]
speed_subset = speed[close_indices]

# 3️⃣ Normalize for plotting, but preserve relative structure
positions = (positions - positions.mean(axis=0)) * 2  # center + scale slightly

# 4️⃣ Build the graph
G = nx.Graph()
for i in range(len(positions)):
    G.add_node(i, pos=positions[i], speed=float(speed_subset[i]))

# Add edges if vehicles are close (within ~20m equivalent)
for i in range(len(positions)):
    for j in range(i + 1, len(positions)):
        dist = np.linalg.norm(positions[i] - positions[j])
        if dist < 20:  # slightly higher threshold so edges actually appear
            G.add_edge(i, j, weight=dist)

# 5️⃣ Extract attributes for plotting
pos_dict = {i: positions[i] for i in G.nodes}
node_speeds = [G.nodes[i]['speed'] for i in G.nodes]
edge_weights = [G[u][v]['weight'] for u, v in G.edges]

# 6️⃣ Create a clear visualization
plt.figure(figsize=(8, 8))
nodes = nx.draw_networkx_nodes(
    G, pos=pos_dict,
    node_size=200,
    node_color=node_speeds,
    cmap='plasma',  # better gradient visibility
    edgecolors='black',
    linewidths=0.7,
    alpha=0.9
)
edges = nx.draw_networkx_edges(
    G, pos=pos_dict,
    width=1.5,
    edge_color=edge_weights,
    edge_cmap=plt.cm.Blues,
    alpha=0.7
)
nx.draw_networkx_labels(G, pos=pos_dict, font_size=8, font_color='white', font_weight='bold')

# 7️⃣ Optional: Add velocity arrows for intuition
plt.quiver(
    positions[:, 0], positions[:, 1],
    vx_subset, vy_subset,
    angles='xy', scale_units='xy', scale=0.5, color='gray', alpha=0.6
)

# 8️⃣ Final formatting
plt.title("Zoomed-in Driving Graph (color = speed, edges = proximity, arrows = velocity)")
plt.colorbar(nodes, label="Speed (m/s)")
plt.axis("equal")
plt.axis("off")
plt.tight_layout()
plt.savefig("graph_representation_zoomed_for_data_set_2.png", dpi=300)
print("✅ Saved graph as graph_representation_zoomed_final2.png")
