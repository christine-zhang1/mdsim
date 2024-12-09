# using psi4 to calculate energies on simulation results using DFT
import csv
import os
import sys

from ase.calculators.psi4 import Psi4
from ase.io import Trajectory
from pathlib import Path

def write_info(file_path, header, info):
    file_exists = os.path.isfile(file_path) and os.path.getsize(file_path) > 0
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(info)

def dft(path, traj_idx):
    md_dir = Path(path)
    traj = Trajectory(md_dir / 'atoms.traj')
    atoms = traj[traj_idx]
    calc = Psi4(atoms = atoms,
        method = 'wB97M-D3BJ',
        memory = '2GB',
        basis = 'def2-TZVPPD',
        num_threads = 16)
    atoms.calc = calc
    pe = atoms.get_potential_energy()
    ke = atoms.get_kinetic_energy() # doesn't invoke dft
    return str(atoms.symbols), pe, ke

if __name__ == "__main__":
    init_idx = int(sys.argv[1]) # init_idx
    print("DFT Calculating for init_idx", init_idx)
    # paths to simulation results
    path_dt = f'/home/christine/mdsim/MODELPATH/maceoff_split_gemnet_dT_full/md_25ps_123_init_{init_idx}'
    path_t = f'/home/christine/mdsim/MODELPATH/spice_all_gemnet_t_maceoff_split_mine/md_25ps_123_init_{init_idx}'
    
    # variables to write to log file
    file_dt = '/home/christine/mdsim/md_scripts/records_spice_molecules_set/gemnet_dt_full.csv'
    file_t = '/home/christine/mdsim/md_scripts/records_spice_molecules_set/gemnet_t_full.csv'
    header = ['molecule', 'init_idx', 'traj[0] potential energy', 'traj[0] kinetic energy', 'traj[-1] potential energy', 'traj[-1] kinetic energy', 'total energy deviation (meV)']
    
    # perform DFT on the first model
    molecule, pe_0, ke_0 = dft(path_dt, traj_idx=0)
    _, pe_final, ke_final = dft(path_dt, traj_idx=-1)
    total_energy_deviation = (pe_final + ke_final - pe_0 - ke_0) * 1000
    info_list = [molecule, init_idx, pe_0, ke_0, pe_final, ke_final, total_energy_deviation]
    
    # write to log file for first model
    write_info(file_dt, header, info_list)
    
    # perform DFT on the second model
    molecule, pe_0, ke_0 = dft(path_t, traj_idx=0)
    _, pe_final, ke_final = dft(path_t, traj_idx=-1)
    total_energy_deviation = (pe_final + ke_final - pe_0 - ke_0) * 1000
    
    info_list = [molecule, init_idx, pe_0, ke_0, pe_final, ke_final, total_energy_deviation]
    # write to log file for second model
    write_info(file_t, header, info_list)
    