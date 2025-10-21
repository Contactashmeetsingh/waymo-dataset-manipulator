import tensorflow as tf
import numpy as np
from pathlib import Path

FILENAME = Path("/home/contactashmeetsingh/WOMD-Reasoning/training_tfexample.tfrecord-00001-of-01000")
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
# ✅ Positions loaded: (11648,) (11648,)
# ✅ Velocities loaded: (11648,) (11648,)
# ✅ Example speeds: [6.27331498 6.00192259 5.99332842 5.81583145 5.7053656  5.56871898
#  5.42096326 5.21240463 5.22403553 4.95930115]