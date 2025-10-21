import tensorflow as tf
import numpy as np
from pathlib import Path

FILENAME = Path(__file__).resolve().parents[1] / "training.tfrecord"
dataset = tf.data.TFRecordDataset(str(FILENAME))
raw = next(iter(dataset))
example = tf.train.Example()
example.ParseFromString(raw.numpy())

def get_feature(name):
    feat = example.features.feature[name].float_list.value
    return np.array(feat)

# Example: extract positions and velocities
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
vx = np.concatenate([
    get_feature("state/past/velocity_x"),
    get_feature("state/current/velocity_x"),
    get_feature("state/future/velocity_x")
])
vy = np.concatenate([
    get_feature("state/past/velocity_y"),
    get_feature("state/current/velocity_y"),
    get_feature("state/future/velocity_y")
])

speed = np.sqrt(vx**2 + vy**2)
print("✅ Positions loaded:", x.shape, y.shape)
print("✅ Velocities loaded:", vx.shape, vy.shape)
print("✅ Example speeds:", speed[:10])
