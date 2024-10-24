# using psi4 to calculate energies on simulation results using DFT
import sys

from ase import Atoms
from ase.calculators.psi4 import Psi4
from ase.io import Trajectory
from pathlib import Path

# copied this over from ase_utils
def data_to_atoms(data):
    numbers = data.atomic_numbers
    positions = data.pos
    cell = data.cell.squeeze()
    atoms = Atoms(numbers=numbers.cpu().detach().numpy(), 
                  positions=positions.cpu().detach().numpy(), 
                  cell=cell.cpu().detach().numpy(),
                  pbc=[True, True, True])
    return atoms

def get_overall_energy(atoms):
    """
    Calculate the overall energy of the system from the elements and their respective energies.
    """
    # list of valid elements and their respective energies
    valid_elem = {35: -70045.28385080204,  # Br
                  6: -1030.5671648271828,  # C
                  17: -12522.649269035726,  # Cl
                  9: -2715.318528602957,  # F
                  1: -13.571964772646918,  # H
                  53: -8102.524593409054,  # I
                  7: -1486.3750255780376,  # N
                  8: -2043.933693071156,  # O
                  15: -9287.407133426237,  # P
                  16: -10834.4844708122}  # S
    
    total = 0
    for an in atoms.get_atomic_numbers():
        if an in valid_elem:
            total += valid_elem[an]
    return total

# get atoms from the trajectory file from the simulation
def get_atoms_from_simulation(path, traj_idx):
    md_dir = Path(path)
    traj = Trajectory(md_dir / 'atoms.traj')
    atoms = traj[traj_idx]
    return atoms

if __name__ == "__main__":
    init_idx = int(sys.argv[1]) # init_idx
    path = f'/home/christine/mdsim/MODELPATH/spice_all_gemnet_t_maceoff_split/md_25ps_123_init_{init_idx}'
    with open('/home/christine/mdsim/md_scripts/records_gemnet_t.txt', 'a') as f:
        # traj[0]
        atoms = get_atoms_from_simulation(path, traj_idx=0)
        f.write(f'molecule: {str(atoms.symbols)}\n')
        f.write(f'init_idx: {init_idx}\n\n')
        f.write(f'GEMNET-T CHECKPOINT\n')
        calc = Psi4(atoms = atoms,
            method = 'wB97M-D3BJ',
            memory = '2GB',
            basis = 'def2-TZVPPD')
        atoms.calc = calc
        pe_0 = atoms.get_potential_energy()
        f.write(f'traj[0] potential energy: {pe_0}\n')
            
        # traj[-1]
        atoms = get_atoms_from_simulation(path, traj_idx=-1)
        calc = Psi4(atoms = atoms,
            method = 'wB97M-D3BJ',
            memory = '2GB',
            basis = 'def2-TZVPPD')
        atoms.calc = calc
        pe = atoms.get_potential_energy()
        f.write(f'traj[-1] potential energy: {pe}\n')
        f.write(f'energy is off by {(pe - pe_0)*1000} meV\n\n')
        f.write('---------------------------------------------------------\n')
    