import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Path to your TFRecord (from earlier)
FILENAME = Path(__file__).resolve().parents[1] / "training.tfrecord"
print("Using:", FILENAME)

# Load the first record
dataset = tf.data.TFRecordDataset(str(FILENAME))
raw = next(iter(dataset))
example = tf.train.Example()
example.ParseFromString(raw.numpy())

# Helper to extract float features safely
def get_feature(name):
    feat = example.features.feature[name].float_list.value
    return np.array(feat)

# Decode road graph
roadgraph_xyz = get_feature("roadgraph_samples/xyz").reshape(-1, 3)

# Decode object trajectories (past + current + future)
x = np.concatenate([
    get_feature("state/past/x"),
    get_feature("state/current/x"),
    get_feature("state/future/x")
])
y = np.concatenate([
    get_feature("state/past/y"),
    get_feature("state/current/y"),
    get_feature("state/future/y")
])
# Create clearer visualization
plt.figure(figsize=(10, 10), dpi=200)
plt.style.use("seaborn-v0_8-darkgrid")

# Plot road graph (light gray)
plt.scatter(
    roadgraph_xyz[:, 0],
    roadgraph_xyz[:, 1],
    s=4, c="#b0b0b0", alpha=0.6,
    label="Roadgraph (lanes, boundaries)"
)

# Plot vehicle/object positions
plt.scatter(
    x, y,
    s=12, c="#ff4040", alpha=0.8, edgecolor="k", linewidth=0.2,
    label="Vehicles & Agents"
)

# Highlight the central area for clarity
plt.xlim(np.percentile(x, 2), np.percentile(x, 98))
plt.ylim(np.percentile(y, 2), np.percentile(y, 98))

# Improve labeling and aesthetics
plt.xlabel("X position (m)", fontsize=10)
plt.ylabel("Y position (m)", fontsize=10)
plt.title("Waymo Motion Dataset — Scenario Overview", fontsize=12, fontweight="bold")
plt.legend(loc="upper right", fontsize=9)
plt.tight_layout()

# Save higher-quality image
plt.savefig("scenario_plot_clear.png", dpi=400, bbox_inches="tight")
print("✅ Saved improved plot as scenario_plot_clear.png")