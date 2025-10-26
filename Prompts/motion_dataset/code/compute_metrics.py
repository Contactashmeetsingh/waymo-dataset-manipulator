import numpy as np
from extract_features import x, y, vx, vy, speed

# 1️⃣ Compute acceleration magnitude
# (approximation: difference in consecutive speeds)
acceleration = np.diff(speed)
print("Acceleration samples:", acceleration[:10])

# 2️⃣ Compute heading (direction angle in radians)
heading = np.arctan2(vy, vx)
print("Heading angles:", heading[:10])

# 3️⃣ Compute distance of each point from origin (ego-centered view)
distance_from_origin = np.sqrt(x**2 + y**2)
print("Distance samples:", distance_from_origin[:10])

# 4️⃣ Compute relative motion between pairs of objects (for graphs later)
# NOTE: this assumes consecutive points belong to consecutive agents (approximation)
dx = np.diff(x)
dy = np.diff(y)
displacement = np.sqrt(dx**2 + dy**2)
print("Displacement samples:", displacement[:10])
