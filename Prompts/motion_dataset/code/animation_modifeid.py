import os, glob, uuid
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

# ================= CONFIG =================
# Option A: hardcode (recommended after renaming):
FILENAME = "/home/contactashmeetsingh/WOMD-Reasoning/Prompts/motion_dataset/uncompressed_tf_example_training_training_tfexample_00000-of-01000.tfrecord"

# Option B: auto-pick first TFRecord in the folder (uncomment to use)
# files = glob.glob("/home/contactashmeetsingh/WOMD-Reasoning/Prompts/motion_dataset/*.tfrecord*")
# if not files: raise FileNotFoundError("No TFRecord found in motion_dataset/")
# FILENAME = files[0]

OUTPUT_DIR = "/home/contactashmeetsingh/WOMD-Reasoning/Prompts/motion_dataset/visualization"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUT_MP4 = os.path.join(OUTPUT_DIR, "motion_tfexample_animation_modified.mp4")

# Waymo TFExample: keep in sync with official tutorial
num_map_samples = 30000
features_description = {
    # Roadgraph (sparse points: xyz + valid)
    "roadgraph_samples/xyz":   tf.io.FixedLenFeature([num_map_samples, 3], tf.float32),
    "roadgraph_samples/valid": tf.io.FixedLenFeature([num_map_samples, 1], tf.int64),

    # Agents (128 max)
    "state/past/x":    tf.io.FixedLenFeature([128, 10], tf.float32),
    "state/past/y":    tf.io.FixedLenFeature([128, 10], tf.float32),
    "state/past/valid":tf.io.FixedLenFeature([128, 10], tf.int64),

    "state/current/x":    tf.io.FixedLenFeature([128, 1], tf.float32),
    "state/current/y":    tf.io.FixedLenFeature([128, 1], tf.float32),
    "state/current/valid":tf.io.FixedLenFeature([128, 1], tf.int64),

    "state/future/x":    tf.io.FixedLenFeature([128, 80], tf.float32),
    "state/future/y":    tf.io.FixedLenFeature([128, 80], tf.float32),
    "state/future/valid":tf.io.FixedLenFeature([128, 80], tf.int64),

    # SDC indicator (int64 [128]); present in the official tutorial
    "state/is_sdc": tf.io.FixedLenFeature([128], tf.int64),
}

# ================ I/O ================
dataset = tf.data.TFRecordDataset(FILENAME, compression_type="")
example = next(iter(dataset))
decoded = tf.io.parse_single_example(example, features_description)

# ================ Helpers ================
def create_figure_and_axes(size_pixels=1000):
    fig, ax = plt.subplots(1, 1, num=uuid.uuid4())
    dpi = 100
    fig.set_size_inches(size_pixels / dpi, size_pixels / dpi)
    fig.set_dpi(dpi)
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(False)
    ax.set_aspect("equal")
    return fig, ax

def fig_canvas_image(fig):
    # Version-safe buffer extraction for Matplotlib >=3.9
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    fig.canvas.draw()
    if hasattr(fig.canvas, "tostring_rgb"):
        buf = fig.canvas.tostring_rgb()
    elif hasattr(fig.canvas, "buffer_rgba"):
        buf = fig.canvas.buffer_rgba()
    elif hasattr(fig.canvas, "tostring_argb"):
        buf = fig.canvas.tostring_argb()
    else:
        raise AttributeError("Cannot extract RGB buffer from Matplotlib canvas.")
    arr = np.frombuffer(buf, dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    # If ARGB, drop alpha & reorder; if RGB/RGBA, reshape accordingly
    if arr.size == w * h * 4:
        arr = arr.reshape(h, w, 4)[..., :3]
    else:
        arr = arr.reshape(h, w, 3)
    return arr

def get_colormap(n):
    cmap = cm.get_cmap("jet", n)
    colors = cmap(np.arange(n))
    np.random.shuffle(colors)
    return colors

# ============ Decode arrays & masks ============
past_x = decoded["state/past/x"].numpy()        # [A, 10]
past_y = decoded["state/past/y"].numpy()
cur_x  = decoded["state/current/x"].numpy()     # [A, 1]
cur_y  = decoded["state/current/y"].numpy()
fut_x  = decoded["state/future/x"].numpy()      # [A, 80]
fut_y  = decoded["state/future/y"].numpy()

past_v = decoded["state/past/valid"].numpy().astype(bool)
cur_v  = decoded["state/current/valid"].numpy().astype(bool)
fut_v  = decoded["state/future/valid"].numpy().astype(bool)

is_sdc = decoded["state/is_sdc"].numpy().astype(bool)   # [A]

# stack to [A, T]
all_x = np.concatenate([past_x, cur_x, fut_x], axis=1)  # T = 10 + 1 + 80
all_y = np.concatenate([past_y, cur_y, fut_y], axis=1)
valid = np.concatenate([past_v, cur_v, fut_v], axis=1)

# Mask invalid samples
all_x[~valid] = np.nan
all_y[~valid] = np.nan

A, T = all_x.shape
colors = get_colormap(A)

# Roadgraph (plot only valid points)
rg_xyz   = decoded["roadgraph_samples/xyz"].numpy()             # [N,3]
rg_valid = decoded["roadgraph_samples/valid"].numpy().astype(bool)  # [N,1]
if rg_xyz.shape[0] > 0:
    rg_pts = rg_xyz[rg_valid.squeeze(), :2]
else:
    rg_pts = np.empty((0, 2), dtype=np.float32)

# ============ Viewport strategy ============
# Prefer a fixed window around SDC current pos (±50 m). If no SDC, auto-fit.
def viewport_for_frame(t, default_pad=50.0):
    # t is index in [0, T)
    if np.any(is_sdc):
        # If multiple SDC flags (shouldn't happen), take first
        sdc_idx = int(np.where(is_sdc)[0][0])
        x_t = all_x[sdc_idx, t]
        y_t = all_y[sdc_idx, t]
        if np.isfinite(x_t) and np.isfinite(y_t):
            return (x_t - default_pad, x_t + default_pad,
                    y_t - default_pad, y_t + default_pad)
    # fallback: auto-fit all valid points at time t
    x_t = all_x[:, t]
    y_t = all_y[:, t]
    x_t = x_t[np.isfinite(x_t)]
    y_t = y_t[np.isfinite(y_t)]
    if x_t.size == 0 or y_t.size == 0:
        return (-50, 50, -50, 50)
    cx = 0.5 * (np.max(x_t) + np.min(x_t))
    cy = 0.5 * (np.max(y_t) + np.min(y_t))
    span = max(np.ptp(x_t), np.ptp(y_t))
    span = max(span, 20.0)  # at least 20 m
    pad = span * 0.6
    return (cx - pad, cx + pad, cy - pad, cy + pad)

# ============ Frame rendering ============
def render_frames(size_pixels=1000):
    frames = []
    for t in range(T):
        fig, ax = create_figure_and_axes(size_pixels)

        # Dark mode background
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")

        # Roadgraph: bright gray for visibility
        if rg_pts.shape[0] > 0:
            ax.scatter(rg_pts[:, 0], rg_pts[:, 1], s=1.5, c="#6E6E6E", alpha=0.9, zorder=1)

        for i in range(A):
            if not np.any(np.isfinite(all_x[i, :])):
                continue

            # Past: deep cyan
            ax.plot(
                all_x[i, :t],
                all_y[i, :t],
                color="#00FFFF",
                linewidth=1.2,
                alpha=0.8,
                zorder=2
            )

            # Current point: bright white (SDC = yellow square)
            if np.isfinite(all_x[i, t]) and np.isfinite(all_y[i, t]):
                if is_sdc[i]:
                    ax.scatter(
                        all_x[i, t], all_y[i, t],
                        s=60, c="#FFFF00", marker="s", zorder=4, edgecolors="black", linewidths=0.5
                    )
                else:
                    ax.scatter(
                        all_x[i, t], all_y[i, t],
                        s=10, c="white", zorder=3
                    )

            # Future: neon red
            if t + 1 < T:
                ax.plot(
                    all_x[i, t+1:],
                    all_y[i, t+1:],
                    color="#FF3131",
                    linewidth=1.0,
                    alpha=0.7,
                    zorder=2
                )

        # Camera window
        xmin, xmax, ymin, ymax = viewport_for_frame(t, default_pad=50.0)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Prominent title in white
        ax.set_title(
            f"Waymo Motion — Frame {t+1}/{T}",
            color="white",
            fontsize=10,
            pad=6
        )
        ax.tick_params(colors="gray", labelsize=6)
        for spine in ax.spines.values():
            spine.set_color("#333333")

        img = fig_canvas_image(fig)
        plt.close(fig)
        frames.append(img)
    return frames

# ============ Animation =============
def save_animation(images, out_path):
    fig, ax = plt.subplots()
    frame = ax.imshow(images[0])
    ax.axis("off")

    def step(i):
        frame.set_data(images[i])
        return [frame]

    anim = animation.FuncAnimation(fig, step, frames=len(images), interval=100)
    anim.save(out_path, writer="ffmpeg")
    print(f"✅ Saved animation: {out_path}")

# ============ RUN =============
print(f"📂 Using: {FILENAME}")
print("🎬 Rendering frames …")
imgs = render_frames(size_pixels=900)
print("🎥 Writing MP4 …")
# Downsample frames to reduce size if needed: imgs[::5]
save_animation(imgs[::5], OUT_MP4)
print("✅ Done.")
