from mace.calculators import mace_off
from ase import Atoms
from ase import build
from ase.md import MDLogger
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.io import Trajectory
from pathlib import Path
from tqdm import tqdm
from mdsim.datasets.lmdb_dataset import LmdbDataset

def data_to_atoms(data):
    numbers = data.atomic_numbers
    positions = data.pos
    cell = data.cell.squeeze()
    atoms = Atoms(numbers=numbers.cpu().detach().numpy(), 
                  positions=positions.cpu().detach().numpy(), 
                  cell=cell.cpu().detach().numpy(),
                  pbc=[True, True, True])
    return atoms

# maceoff = mace_off(model="medium", device='cuda')
maceoff = mace_off()

# md_dir = Path('../MODELPATH/maceoff_split_gemnet_dT_results/md_25ps_maceoff_calc_123_init_7738')
# traj = Trajectory(md_dir / 'atoms.traj')
# atoms = traj[0]

test_dataset = LmdbDataset({'src': '/data/shared/MLFF/SPICE/maceoff_split/test'})
init_data = test_dataset[7738]
atoms = data_to_atoms(init_data)

atoms.calc = maceoff
fs = 0.5

# Initialize velocities.
T_init = 300  # Initial temperature in K
MaxwellBoltzmannDistribution(atoms, temperature_K=T_init)
trajectory = Trajectory('testing_atoms_7738_2.traj', 'w', atoms)

# Set up the VelocityVerlet dynamics engine for NVE ensemble.
dyn = VelocityVerlet(atoms, fs * units.fs)

# Attach the trajectory to the dynamics
dyn.attach(trajectory.write, interval=100)

num_steps = 50000

# Create a tqdm progress bar
with tqdm(total=num_steps) as pbar:
    # Define a function to update the progress bar and run the simulation
    for i in range(num_steps):
        dyn.run(1)  # Run the simulation step by step
        pbar.update(1)  # Update the progress bar after each step
        