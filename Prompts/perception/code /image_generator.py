"""
===========================================================
Waymo Perception v2 - Save One Frame per Camera + One Video
===========================================================

Outputs:
  • 5 total images (one per camera angle)
  • 1 video made from all FRONT camera frames
"""

import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
PARQUET_FILE = "/home/contactashmeetsingh/WOMD-Reasoning/Prompts/training_camera_image_10017090168044687777_6380_000_6400_000 (1).parquet"
OUTPUT_DIR = "output_images"
VIDEO_NAME = "drive_segment.mp4"
FPS = 10

CAMERA_MAP = {
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT"
}

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
if not os.path.exists(PARQUET_FILE):
    raise FileNotFoundError(f"❌ File not found: {PARQUET_FILE}")

camera_df = pd.read_parquet(PARQUET_FILE)
print(f"✅ Loaded {len(camera_df)} frames.")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------
# FUNCTION: SAVE IMAGE
# -----------------------------------------------------------
def save_image(image_bytes, cam_name, timestamp):
    """Decode JPEG bytes and save a single image."""
    image_tensor = tf.image.decode_jpeg(image_bytes)
    plt.figure(figsize=(8, 5))
    plt.imshow(image_tensor)
    plt.title(f"{cam_name} | Timestamp: {timestamp}")
    plt.axis("off")

    file_path = os.path.join(OUTPUT_DIR, f"{cam_name}.png")
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved {file_path}")
    return file_path

# -----------------------------------------------------------
# STEP 1: Save exactly ONE frame per camera (5 total)
# -----------------------------------------------------------
print("\n📸 Saving one frame per camera view...")

saved_cameras = set()
for _, row in camera_df.iterrows():
    cam_id = row["key.camera_name"]
    cam_name = CAMERA_MAP.get(cam_id, f"CAM_{cam_id}")

    # Skip if we've already saved this camera
    if cam_name in saved_cameras:
        continue

    save_image(row["[CameraImageComponent].image"], cam_name, row["key.frame_timestamp_micros"])
    saved_cameras.add(cam_name)

    if len(saved_cameras) == 5:
        break

print("✅ Saved 5 total images (one per camera).")

# -----------------------------------------------------------
# STEP 2: Create a video using FRONT camera frames
# -----------------------------------------------------------
print("\n🎥 Creating video using FRONT camera frames...")

video_frames = []
for _, row in camera_df.iterrows():
    if row["key.camera_name"] != 1:  # FRONT camera only
        continue
    img_tensor = tf.image.decode_jpeg(row["[CameraImageComponent].image"])
    np_img = img_tensor.numpy()
    video_frames.append(np_img)

if video_frames:
    height, width, _ = video_frames[0].shape
    out = cv2.VideoWriter(VIDEO_NAME, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))

    for frame in video_frames:
        # Convert RGB → BGR for OpenCV
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"🎬 Video saved as: {VIDEO_NAME}")
else:
    print("⚠️ No FRONT camera frames found.")
