"""Demonstrates molecular dynamics with constant energy."""

from ase import units
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.io import Trajectory
from pathlib import Path

# Use Asap for a huge performance increase if it is installed
use_asap = False

if use_asap:
    from asap3 import EMT
    size = 10
else:
    from ase.calculators.emt import EMT
    size = 3

md_dir = Path('../../MDsim/MODELPATH/md17-aspirin_10k_gemnet_t_dT/md_25ps_123')
traj = Trajectory(md_dir / 'atoms.traj')
atoms = traj[0]

# Describe the interatomic interactions with the Effective Medium Theory
atoms.calc = EMT()

# Set the momenta corresponding to T=300K
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

fs = 0.5

# We want to run MD with constant energy using the VelocityVerlet algorithm.
dyn = VelocityVerlet(atoms, fs * units.fs)  # originally 5 fs time step


def printenergy(a):
    """Function to print the potential, kinetic and total energy"""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))
    
def printforces(a):
    """Function to print the forces"""
    forces = a.get_forces()
    print(f'FORCES {forces}')
    
# Now run the dynamics
for i in range(10):
    dyn.run(10)
    printforces(atoms)

# dyn.attach(MDLogger(dyn, atoms, f'md_{fs}.log', header=True, stress=False,
#            peratom=True, mode="w"), interval=2000)
# dyn.run(1000000)

# graph the md_logs in graph_mdlogs.py
