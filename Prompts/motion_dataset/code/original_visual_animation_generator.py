import os
import uuid
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

# ========= CONFIG =========
# ⚠️ Make sure the filename below exists (no spaces or parentheses)
FILENAME = "/home/contactashmeetsingh/WOMD-Reasoning/Prompts/motion_dataset/uncompressed_tf_example_training_training_tfexample_00000-of-01000.tfrecord"
OUTPUT_DIR = "/home/contactashmeetsingh/WOMD-Reasoning/Prompts/motion_dataset/visualization/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========= DEFINE FEATURES =========
num_map_samples = 30000

roadgraph_features = {
    "roadgraph_samples/xyz": tf.io.FixedLenFeature([num_map_samples, 3], tf.float32),
    "roadgraph_samples/valid": tf.io.FixedLenFeature([num_map_samples, 1], tf.int64),
}

state_features = {
    "state/past/x": tf.io.FixedLenFeature([128, 10], tf.float32),
    "state/past/y": tf.io.FixedLenFeature([128, 10], tf.float32),
    "state/past/valid": tf.io.FixedLenFeature([128, 10], tf.int64),
    "state/current/x": tf.io.FixedLenFeature([128, 1], tf.float32),
    "state/current/y": tf.io.FixedLenFeature([128, 1], tf.float32),
    "state/current/valid": tf.io.FixedLenFeature([128, 1], tf.int64),
    "state/future/x": tf.io.FixedLenFeature([128, 80], tf.float32),
    "state/future/y": tf.io.FixedLenFeature([128, 80], tf.float32),
    "state/future/valid": tf.io.FixedLenFeature([128, 80], tf.int64),
}

features_description = {}
features_description.update(roadgraph_features)
features_description.update(state_features)

# ========= PARSE RECORD =========
dataset = tf.data.TFRecordDataset(FILENAME, compression_type="")
example = next(iter(dataset))
parsed = tf.io.parse_single_example(example, features_description)

# ========= HELPER FUNCTIONS =========
def create_figure_and_axes(size_pixels=1000):
    fig, ax = plt.subplots(1, 1, num=uuid.uuid4())
    dpi = 100
    fig.set_size_inches(size_pixels / dpi, size_pixels / dpi)
    ax.set_facecolor("white")
    fig.set_facecolor("white")
    ax.grid(False)
    ax.set_aspect("equal")
    return fig, ax

def fig_canvas_image(fig):
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.canvas.draw()
    # Support both old and new Matplotlib versions
    if hasattr(fig.canvas, "tostring_rgb"):
        buf = fig.canvas.tostring_rgb()
    elif hasattr(fig.canvas, "buffer_rgba"):
        buf = fig.canvas.buffer_rgba()
    elif hasattr(fig.canvas, "tostring_argb"):
        # For Matplotlib 3.9+
        buf = fig.canvas.tostring_argb()
    else:
        raise AttributeError("Cannot find an RGB buffer method in this Matplotlib version.")
    data = np.frombuffer(buf, dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return data.reshape(h, w, -1)


def get_colormap(num_agents):
    cmap = cm.get_cmap("jet", num_agents)
    colors = cmap(np.arange(num_agents))
    np.random.shuffle(colors)
    return colors

# ========= VISUALIZATION =========
def visualize_all_agents(parsed, size_pixels=800):
    past_x = parsed["state/past/x"].numpy()
    past_y = parsed["state/past/y"].numpy()
    curr_x = parsed["state/current/x"].numpy()
    curr_y = parsed["state/current/y"].numpy()
    fut_x = parsed["state/future/x"].numpy()
    fut_y = parsed["state/future/y"].numpy()

    valid = np.concatenate([
        parsed["state/past/valid"].numpy(),
        parsed["state/current/valid"].numpy(),
        parsed["state/future/valid"].numpy()
    ], axis=1) > 0

    all_x = np.concatenate([past_x, curr_x, fut_x], axis=1)
    all_y = np.concatenate([past_y, curr_y, fut_y], axis=1)

    # Mask invalid coordinates
    all_x[~valid] = np.nan
    all_y[~valid] = np.nan

    num_agents, num_steps = all_x.shape
    colors = get_colormap(num_agents)

    # Extract roadgraph
    if "roadgraph_samples/xyz" in parsed:
        roadgraph = parsed["roadgraph_samples/xyz"].numpy()
        roadgraph_valid = parsed["roadgraph_samples/valid"].numpy().astype(bool)
        roadgraph = roadgraph[roadgraph_valid.squeeze()]
    else:
        roadgraph = np.zeros((0, 3))

    # Determine bounding box for viewport
    valid_points = np.column_stack((all_x[valid], all_y[valid]))
    center_x, center_y = np.nanmean(valid_points, axis=0)
    range_x = np.nanmax(valid_points[:, 0]) - np.nanmin(valid_points[:, 0])
    range_y = np.nanmax(valid_points[:, 1]) - np.nanmin(valid_points[:, 1])
    view_size = max(range_x, range_y) * 1.2

    images = []
    for t in range(num_steps):
        fig, ax = create_figure_and_axes(size_pixels)
        # Plot roadgraph first (gray background)
        if len(roadgraph) > 0:
            ax.scatter(roadgraph[:, 0], roadgraph[:, 1], s=0.5, c="gray", alpha=0.5)
        # Plot each agent’s trajectory so far
        for i in range(num_agents):
            if np.any(valid[i, :t + 1]):
                ax.plot(all_x[i, :t + 1], all_y[i, :t + 1], color=colors[i], linewidth=1)
        ax.set_xlim(center_x - view_size / 2, center_x + view_size / 2)
        ax.set_ylim(center_y - view_size / 2, center_y + view_size / 2)
        ax.set_title(f"Frame {t + 1}/{num_steps}")
        ax.axis("equal")
        img = fig_canvas_image(fig)
        plt.close(fig)
        images.append(img)
    return images


# ========= CREATE ANIMATION =========
def create_animation(images, output_path):
    fig, ax = plt.subplots()
    frame = ax.imshow(images[0])
    ax.axis("off")

    def animate(i):
        frame.set_data(images[i])
        return [frame]

    anim = animation.FuncAnimation(fig, animate, frames=len(images), interval=100)
    anim.save(output_path, writer="ffmpeg")
    print(f"✅ Saved animation: {output_path}")

# ========= RUN =========
print("🎬 Generating frames...")
images = visualize_all_agents(parsed)
output_path = os.path.join(OUTPUT_DIR, "motion_tfexample_animation.mp4")
print("🎥 Creating animation...")
create_animation(images[::5], output_path)
print("✅ Done.")
