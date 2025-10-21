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
# ✅ Positions loaded: (11648,) (11648,)
# ✅ Velocities loaded: (11648,) (11648,)
# ✅ Example speeds: [6.27331498 6.00192259 5.99332842 5.81583145 5.7053656  5.56871898
#  5.42096326 5.21240463 5.22403553 4.95930115]
# Acceleration samples: [-0.27139239 -0.00859417 -0.17749697 -0.11046585 -0.13664661 -0.14775573
#  -0.20855863  0.0116309  -0.26473438 -4.45106628]
# Heading angles: [ 0.00622681 -0.01057624  0.00977665 -0.01175429 -0.01198188 -0.01315281
#  -0.01261052  0.00093677 -0.00373874  0.00886131]
# Distance samples: [7893.21310233 7893.26415859 7893.32732936 7893.37620691 7893.42406804
#  7893.47017256 7893.51538422 7893.56592386 7893.61417896 7893.66623856]
# Displacement samples: [ 0.60019226  0.59933284  0.58158314  0.57053656  0.5568719   0.54209633
#   0.52124046  0.52240355  0.49593012 13.22804715]