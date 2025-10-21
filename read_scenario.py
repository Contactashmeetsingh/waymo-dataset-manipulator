from pathlib import Path
import tensorflow as tf

# Resolve training.tfrecord sitting one level above Prompts/
FILENAME = Path("/home/contactashmeetsingh/WOMD-Reasoning/training_tfexample.tfrecord-00001-of-01000")
print("File path:", FILENAME)
print("File exists?", FILENAME.exists())

ds = tf.data.TFRecordDataset(str(FILENAME))
it = iter(ds)
raw = next(it)  # will throw if file truly missing/corrupt

ex = tf.train.Example()
ex.ParseFromString(raw.numpy())

print("✅ Loaded one record")
print("Feature keys (first 20):")
for i, k in enumerate(ex.features.feature.keys()):
    if i >= 20:
        break
    print("-", k)
