# using psi4 to calculate energies on simulation results using DFT
import csv
import os
import sys
import time

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
    start_time = time.time()
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
    elapsed_time = (time.time() - start_time) / 60
    print(f"Time elapsed for traj_idx {traj_idx}: {elapsed_time:.2f} minutes")
    return str(atoms.symbols), pe, ke

if __name__ == "__main__":
    # init_idx = int(sys.argv[1]) # init_idx
    init_idxs = [193, 49489, 43692, 22229, 48228, 14423, 32081, 29366, 24134, 15700]
    for init_idx in init_idxs:
        print("DFT Calculating for init_idx", init_idx)
        # paths to simulation results
        path_dt = f'/home/christine/mdsim/MODELPATH/maceoff_split_gemnet_dT_full/md_25ps_123_init_{init_idx}'
        path_t = f'/home/christine/mdsim/MODELPATH/spice_all_gemnet_t_maceoff_split_mine/md_25ps_123_init_{init_idx}'
        
        # variables to write to log file
        file_dt = '/home/christine/mdsim/md_scripts/records_spice_molecules_set/gemnet_dt_full_traj.csv'
        file_t = '/home/christine/mdsim/md_scripts/records_spice_molecules_set/gemnet_t_full_traj.csv'
        points = [100, 200, 300, 400, -1] # not including 0 point
        header = ['molecule', 'init_idx', 'traj_point', 'potential energy', 'kinetic energy', 'energy deviation from start (meV)']
        
        # perform DFT on the first model
        molecule, pe_0, ke_0 = dft(path_dt, traj_idx=0)
        for pt in points:
            info_list = [molecule, init_idx, pt]
            _, pe, ke = dft(path_dt, traj_idx=pt)
            info_list.append(pe)
            info_list.append(ke)
            info_list.append((pe + ke - pe_0 - ke_0) * 1000)
            write_info(file_dt, header, info_list)
            
        # perform DFT on the second model
        molecule, pe_0, ke_0 = dft(path_t, traj_idx=0)
        for pt in points:
            info_list = [molecule, init_idx, pt]
            _, pe, ke = dft(path_t, traj_idx=pt)
            info_list.append(pe)
            info_list.append(ke)
            info_list.append((pe + ke - pe_0 - ke_0) * 1000)
            write_info(file_t, header, info_list)
        
        print("Done with init_idx", init_idx)