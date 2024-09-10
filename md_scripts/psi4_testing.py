# experimenting with psi4
from ase.calculators.psi4 import Psi4
from ase.build import molecule
from ase.io import Trajectory
from pathlib import Path

md_dir = Path('../MODELPATH/md17-aspirin_10k_gemnet_t/md_25ps_123')
traj = Trajectory(md_dir / 'atoms.traj')
atoms = traj[0]

calc = Psi4(atoms = atoms,
        method = 'pbe',
        memory = '500MB', # this is the default, be aware!
        basis = 'cc-pvdz')

atoms.calc = calc
print(atoms.get_potential_energy())
print(atoms.get_forces())
