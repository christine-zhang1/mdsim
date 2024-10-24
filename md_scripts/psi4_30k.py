# using psi4 to calculate energies on simulation results using DFT
import sys

from ase.calculators.psi4 import Psi4
from ase.io import Trajectory
from pathlib import Path

# get atoms from the trajectory file from the simulation
def get_atoms_from_simulation(path, traj_idx):
    md_dir = Path(path)
    traj = Trajectory(md_dir / 'atoms.traj')
    atoms = traj[traj_idx]
    return atoms

if __name__ == "__main__":
    init_idx = int(sys.argv[1]) # init_idx
    path_dt = f'/home/christine/mdsim/MODELPATH/maceoff_split_gemnet_dT_30k/md_25ps_123_init_{init_idx}' # toby's gemnet-dT checkpoint
    path_t = f'/home/christine/mdsim/MODELPATH/spice_all_gemnet_t_maceoff_split_30k_mine/md_25ps_123_init_{init_idx}'
    with open('/home/christine/mdsim/md_scripts/records_gemnet_t_dt_30k.txt', 'a') as f:
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
        
        atoms = get_atoms_from_simulation(path_t, traj_idx=0)
        f.write(f'GEMNET-T CHECKPOINT\n')
        calc = Psi4(atoms = atoms,
            method = 'wB97M-D3BJ',
            memory = '2GB',
            basis = 'def2-TZVPPD')
        atoms.calc = calc
        pe_0 = atoms.get_potential_energy()
        f.write(f'traj[0] potential energy: {pe_0}\n')
            
        # traj[-1]
        atoms = get_atoms_from_simulation(path_t, traj_idx=-1)
        calc = Psi4(atoms = atoms,
            method = 'wB97M-D3BJ',
            memory = '2GB',
            basis = 'def2-TZVPPD')
        atoms.calc = calc
        pe = atoms.get_potential_energy()
        f.write(f'traj[-1] potential energy: {pe}\n')
        f.write(f'energy is off by {(pe - pe_0)*1000} meV\n')
        f.write('---------------------------------------------------------\n')
    
