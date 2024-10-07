# using psi4 to calculate energies on simulation results using DFT
import sys

from ase import Atoms
from ase.calculators.psi4 import Psi4
from ase.io import Trajectory
from pathlib import Path

from mdsim.datasets.lmdb_dataset import LmdbDataset

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

# get the molecule from the dataset
def get_atoms_from_dataset(init_idx):
    test_dataset = LmdbDataset({'src': '/data/shared/MLFF/SPICE/maceoff_split/test'})
    init_data = test_dataset[init_idx]    
    atoms = data_to_atoms(init_data)
    print(f"reference energy {init_data.reference_energy}")
    print(f'overall energy from atoms {get_overall_energy(atoms)}')
    return atoms

# get atoms from the trajectory file from the simulation
def get_atoms_from_simulation(path, traj_idx):
    md_dir = Path(path)
    # md_dir = Path('../MODELPATH/maceoff_split_gemnet_dT_results/md_25ps_maceoff_calc_123_init_1950')
    traj = Trajectory(md_dir / 'atoms.traj')
    atoms = traj[traj_idx]
    return atoms

if __name__ == "__main__":
    init_idx = int(sys.argv[1]) # init_idx
    path_dt = f'/home/christine/mdsim/MODELPATH/maceoff_split_gemnet_dT_results/md_25ps_123_init_{init_idx}'
    path_calc = f'/home/christine/mdsim/MODELPATH/maceoff_split_gemnet_dT_results/md_25ps_maceoff_calc_123_init_{init_idx}'
    with open('/home/christine/mdsim/md_scripts/records_str_atoms_leq_6.txt', 'a') as f:
        # traj[0]
        atoms = get_atoms_from_simulation(path_dt, traj_idx=0)
        f.write(f'molecule: {str(atoms.symbols)}\n')
        f.write(f'init_idx: {init_idx}\n\n')
        f.write(f'GEMNET-DT CHECKPOINT\n')
        calc = Psi4(atoms = atoms,
            method = 'wB97M-D3BJ',
            memory = '2GB',
            basis = 'def2-TZVPPD')
        atoms.calc = calc
        pe_0 = atoms.get_potential_energy()
        f.write(f'traj[0] potential energy: {pe_0}\n')
            
        # traj[-1]
        atoms = get_atoms_from_simulation(path_dt, traj_idx=-1)
        calc = Psi4(atoms = atoms,
            method = 'wB97M-D3BJ',
            memory = '2GB',
            basis = 'def2-TZVPPD')
        atoms.calc = calc
        pe = atoms.get_potential_energy()
        f.write(f'traj[-1] potential energy: {pe}\n')
        f.write(f'energy is off by {(pe - pe_0)*1000} meV\n\n')
        
        atoms = get_atoms_from_simulation(path_calc, traj_idx=0)
        f.write(f'MACEOFF CALCULATOR\n')
        calc = Psi4(atoms = atoms,
            method = 'wB97M-D3BJ',
            memory = '2GB',
            basis = 'def2-TZVPPD')
        atoms.calc = calc
        pe_0 = atoms.get_potential_energy()
        f.write(f'traj[0] potential energy: {pe_0}\n')
            
        # traj[-1]
        atoms = get_atoms_from_simulation(path_calc, traj_idx=-1)
        calc = Psi4(atoms = atoms,
            method = 'wB97M-D3BJ',
            memory = '2GB',
            basis = 'def2-TZVPPD')
        atoms.calc = calc
        pe = atoms.get_potential_energy()
        f.write(f'traj[-1] potential energy: {pe}\n')
        f.write(f'energy is off by {(pe - pe_0)*1000} meV\n')
        f.write('---------------------------------------------------------\n')
    

# md17 aspirin settings, with coupled cluster
# calc = Psi4(atoms = atoms,
#         method = 'ccsd',
#         memory = '2GB',
#         basis = 'cc-pvdz')

# f.write('forces calculated by DFT:\n')
# for i, row in enumerate(forces):
#     if i != len(forces) - 1:
#         if i == 0:
#             f.write('[')
#         formatted_row = ', '.join([f"{num:.8f}" for num in row])
#         f.write(f"[{formatted_row}],\n")
#     else:
#         formatted_row = ', '.join([f"{num:.8f}" for num in row])
#         f.write(f"[{formatted_row}]]\n")
