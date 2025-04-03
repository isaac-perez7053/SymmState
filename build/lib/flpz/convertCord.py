import numpy as np

# Define the primitive lattice vectors (rprim)
rprim = np.array([[1, -1, 0],
                  [1, 1, 0],
                  [0, 0, 1]])

# Calculate the inverse of rprim
rprim_inv = np.linalg.inv(rprim)

# Define the Cartesian coordinates (xcart)
xcart = np.array([
    [0.00000, 0.00000, 0.00000],
    [7.25467, 0.00000, 0.00000],
    [3.62733, 3.62733, 3.62733],
    [3.62733, -3.62733, 3.62733],
    [7.25467, -3.62733, 3.62733],
    [7.25467, 3.62733, 3.62733],
    [3.62733, 3.62733, 0.00000],
    [3.62733, -3.62733, 0.00000],
    [3.62733, 0.00000, 3.62733],
    [10.88200, 0.00000, 3.62733]
])

# Calculate the fractional coordinates
rfrac = np.dot(xcart, rprim_inv)

rfrac
