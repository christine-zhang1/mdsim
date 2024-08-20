# calculate how conservative the forces are
import numpy as np

def check_conservative(dt):
    # dt is a boolean saying whether we are using gemnet_dT or gemnet_t
    if dt:
        F = np.load("MODELPATH/md17-aspirin_10k_gemnet_t_dT/results/s2ef_predictions.npz")['forces']
    else:
        top = np.load("MODELPATH/md17-aspirin_10k_gemnet_t/results/s2ef_predictions_top.npz")
        right = np.load("MODELPATH/md17-aspirin_10k_gemnet_t/results/s2ef_predictions_right.npz")
        bottom = np.load("MODELPATH/md17-aspirin_10k_gemnet_t/results/s2ef_predictions_bottom.npz")
        left = np.load("MODELPATH/md17-aspirin_10k_gemnet_t/results/s2ef_predictions_left.npz")
        F = np.vstack((top['forces'], right['forces'], bottom['forces'], left['forces']))

    # Number of points
    N = 800
    assert(F.shape[0]/21 == N)
    # there are forces per atom (so 500*21 forces) but we only care about the forces on the first atom of each molecule (there are N molecules)

    x, y = [], []
    with open("mdsim/custom_force_pred/points_on_square.txt") as file:
        for line in file:
            # Split the line into two numbers
            num1, num2 = line.split()
            # Append the numbers to their respective arrays
            x.append(float(num1))
            y.append(float(num2))

    z = np.zeros_like(x)  # Assuming z = 0 for a square in the xy-plane

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
    
check_conservative(False)