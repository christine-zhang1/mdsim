# experimenting with psi4
from ase.calculators.psi4 import Psi4
from ase.build import molecule
import numpy as np

atoms = molecule('aspirin')

calc = Psi4(atoms = atoms,
        method = 'b3lyp',
        memory = '500MB', # this is the default, be aware!
        basis = '6-311g_d_p_')

atoms.calc = calc
print(atoms.get_potential_energy())
print(atoms.get_forces())
