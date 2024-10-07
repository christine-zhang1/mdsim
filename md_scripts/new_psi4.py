import random
from ase import Atoms
from ase.calculators.psi4 import Psi4
from ase.io import Trajectory
from pathlib import Path

traj = Trajectory('testing_atoms_7738_2.traj')
print(len(traj))
atoms = traj[0]
calc = Psi4(atoms = atoms,
        method = 'wB97M-D3BJ',
        memory = '2GB',
        basis = 'def2-TZVPPD')
atoms.calc = calc
pe_0 = atoms.get_potential_energy()
print(pe_0)

atoms = traj[-1]
calc = Psi4(atoms = atoms,
        method = 'wB97M-D3BJ',
        memory = '2GB',
        basis = 'def2-TZVPPD')
atoms.calc = calc
pe_last = atoms.get_potential_energy()
print(pe_last)
print(pe_0 - pe_last)
