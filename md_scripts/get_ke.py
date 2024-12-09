# using psi4 to calculate energies on simulation results using DFT
import sys

from ase import Atoms
from ase.calculators.psi4 import Psi4
from ase.io import Trajectory
from pathlib import Path
import re

from mdsim.datasets.lmdb_dataset import LmdbDataset

# get atoms from the trajectory file from the simulation
def get_atoms_from_simulation(path, traj_idx):
    md_dir = Path(path)
    # md_dir = Path('../MODELPATH/maceoff_split_gemnet_dT_results/md_25ps_maceoff_calc_123_init_1950')
    traj = Trajectory(md_dir / 'atoms.traj')
    atoms = traj[traj_idx]
    return atoms

if __name__ == "__main__":
    txt_path = 'combined_gemnet_dt_t_full.txt'
    gemnet_t = True
    if gemnet_t:
        grad_string = "GEMNET-T CHECKPOINT"
    else:
        grad_string = "MACEOFF CALCULATOR"
    with open(txt_path, 'r') as file:
        file_content = file.read()
    init_idx_values = re.findall(r'init_idx: (\d+)', file_content)
    init_idx_values = [int(idx) for idx in init_idx_values]
    with open('records_gemnet_t_dt_full_ke.txt', 'a') as f:
        for init_idx in init_idx_values:
            path_dt = f'/home/christine/mdsim/MODELPATH/maceoff_split_gemnet_dT_full/md_25ps_123_init_{init_idx}' # toby's gemnet-dT checkpoint
            path_t = f'/home/christine/mdsim/MODELPATH/spice_all_gemnet_t_maceoff_split_mine/md_25ps_123_init_{init_idx}'
            # path_t = f'/home/christine/mdsim/MODELPATH/maceoff_split_gemnet_dT_full/md_25ps_maceoff_calc_123_init_{init_idx}'
            
            # traj[0]
            atoms = get_atoms_from_simulation(path_dt, traj_idx=0)
            f.write(f'molecule: {str(atoms.symbols)}\n')
            f.write(f'init_idx: {init_idx}\n\n')
            f.write(f'GEMNET-DT CHECKPOINT\n')
            ke_0 = atoms.get_kinetic_energy()
            f.write(f'traj[0] kinetic energy: {ke_0}\n')
                
            # traj[-1]
            atoms = get_atoms_from_simulation(path_dt, traj_idx=-1)
            ke = atoms.get_kinetic_energy()
            f.write(f'traj[-1] kinetic energy: {ke}\n')
            f.write(f'energy is off by {(ke - ke_0)*1000} meV\n\n')
            
            atoms = get_atoms_from_simulation(path_t, traj_idx=0)
            f.write(f'{grad_string}\n')
            ke_0 = atoms.get_kinetic_energy()
            f.write(f'traj[0] kinetic energy: {ke_0}\n')
                
            # traj[-1]
            atoms = get_atoms_from_simulation(path_t, traj_idx=-1)
            ke = atoms.get_kinetic_energy()
            f.write(f'traj[-1] kinetic energy: {ke}\n')
            f.write(f'energy is off by {(ke - ke_0)*1000} meV\n')
            f.write('---------------------------------------------------------\n')
    
