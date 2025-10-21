import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from extract_features import x, y, vx, vy, speed

# 1️⃣ Select a manageable subset — focus on a small scene around one agent
subset_size = 200  # first 200 points
positions_all = np.stack([x[:subset_size], y[:subset_size]], axis=1)

# Find agents closest to the first one (to form a local group)
distances = np.linalg.norm(positions_all - positions_all[0], axis=1)
close_indices = np.argsort(distances)[:10]  # 10 closest agents

# 2️⃣ Extract their coordinates and motion
positions = positions_all[close_indices]
vx_subset = vx[close_indices]
vy_subset = vy[close_indices]
speed_subset = speed[close_indices]

# 3️⃣ Normalize and center for visualization (for cleaner layout)
positions = positions - positions.mean(axis=0)

# 4️⃣ Build graph: each node = agent, edge = proximity (if within 15 m)
G = nx.Graph()
for i in range(len(positions)):
    G.add_node(i, pos=positions[i], speed=float(speed_subset[i]))

for i in range(len(positions)):
    for j in range(i + 1, len(positions)):
        dist = np.linalg.norm(positions[i] - positions[j])
        if dist < 15:
            G.add_edge(i, j, weight=dist)

# 5️⃣ Prepare data for plotting
pos_dict = {i: positions[i] for i in G.nodes}
node_speeds = [G.nodes[i]['speed'] for i in G.nodes]
edge_weights = [G[u][v]['weight'] for u, v in G.edges]

# 6️⃣ Plot setup
plt.figure(figsize=(8, 6))
plt.title("Simplified Driving Interaction Graph", fontsize=13)

# Draw nodes — color = speed
nodes = nx.draw_networkx_nodes(
    G, pos=pos_dict,
    node_size=250,
    node_color=node_speeds,
    cmap='coolwarm',
    edgecolors='black',
    linewidths=0.8,
    alpha=0.9
)

# Draw edges — closer = thicker line
edges = nx.draw_networkx_edges(
    G, pos=pos_dict,
    width=[2.5 - 0.1 * w for w in edge_weights],
    edge_color='gray',
    alpha=0.6
)

# Add labels like “Car 1”, “Car 2”, etc.
labels = {i: f"Car {i+1}" for i in G.nodes}
nx.draw_networkx_labels(G, pos=pos_dict, labels=labels, font_size=9, font_color='black')

# 7️⃣ Add small arrows to show velocity direction
plt.quiver(
    positions[:, 0], positions[:, 1],
    vx_subset, vy_subset,
    angles='xy', scale_units='xy', scale=1.5, color='darkgreen', width=0.004, alpha=0.7
)

# 8️⃣ Final formatting
plt.colorbar(nodes, label="Speed (m/s)")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.axis("equal")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("graph_representation_simple.png", dpi=300)
print("✅ Saved clear graph as graph_representation_simple_dataset_1.png")
