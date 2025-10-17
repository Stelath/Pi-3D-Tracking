"""Constants for 3D box representation and quality metrics."""

# 3D Box representation constants
# Box format: [x, y, z, w, l, h, sin_yaw, cos_yaw, vx, vy, vz]

# Position indices
X = 0  # X coordinate
Y = 1  # Y coordinate
Z = 2  # Z coordinate

# Size indices
W = 3  # Width
L = 4  # Length
H = 5  # Height

# Orientation indices
SIN_YAW = 6  # Sin of yaw angle
COS_YAW = 7  # Cos of yaw angle
YAW = 6      # Yaw angle (when stored as angle instead of sin/cos)

# Velocity indices
VX = 8   # Velocity in X
VY = 9   # Velocity in Y
VZ = 10  # Velocity in Z

# Quality indices
CNS = 0  # Centerness score
YNS = 1  # Yawness score

__all__ = [
    'X', 'Y', 'Z',
    'W', 'L', 'H',
    'SIN_YAW', 'COS_YAW', 'YAW',
    'VX', 'VY', 'VZ',
    'CNS', 'YNS',
]
