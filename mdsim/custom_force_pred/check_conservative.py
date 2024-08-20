# calculate how conservative the forces are
import numpy as np

res_1 = np.load("MODELPATH/md17-aspirin_10k_gemnet_t/results/s2ef_predictions_first_half.npz")
res_2 = np.load("MODELPATH/md17-aspirin_10k_gemnet_t/results/s2ef_predictions_second_half.npz")
F = np.vstack((res_1['forces'], res_2['forces']))
print(F.shape)

# Number of points
N = 500
assert(F.shape[0]/21 == N)
# there are forces per atom (so 500*21 forces) but we only care about the forces on the first atom of each molecule (there are N molecules)

# Radius of the circle
radius = 0.5

# Angle increments
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)

# Positions on the circle
x = 2 + radius * np.cos(theta)
y = -0.5 + radius * np.sin(theta)
z = np.zeros_like(x)  # Assuming z = 0 for a circle in the xy-plane

# Compute the work
work = 0.0
for i in range(N):
    # Calculate the difference in position vectors
    if i == N-1:  # Connect last point to the first point to close the loop
        dr = np.array([x[0] - x[i], y[0] - y[i], z[0] - z[i]])
    else:
        dr = np.array([x[i+1] - x[i], y[i+1] - y[i], z[i+1] - z[i]])

    # Calculate the dot product of the force and the displacement
    # multiply i by 21 here because we only care about the force on the first atom of the molecule
    # since the first atom is the one we moved in a circle
    work += np.dot(F[i*21], dr)

print(f"Approximate work done: {work}")