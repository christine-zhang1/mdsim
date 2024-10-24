import bisect
import logging
import pickle
import warnings
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import Subset
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from tqdm import tqdm
import os
from mdsim.common.registry import registry
from mdsim.datasets.lmdb_dataset import LmdbDataset


@registry.register_dataset("multi")
class MultiMoleculeDataset(Dataset):
    r"""Dataset class to load from multiple molecules

    Args:
            config (dict): Dataset configuration
            transform (callable, optional): Data transform function.
                    (default: :obj:`None`)
    """

    def __init__(
        self,
        config,
        normalize_force=False,
        normalize_energy=True,
        transform=None,
        percentages=None,
        return_classical=False,
        return_formation_energy=False,
        negate_force=False,
        filter_small_ions=False,
        carbon_idx=None,
        val=False,
    ):
        super(MultiMoleculeDataset, self).__init__()

        self.config = config
        self.noise_classical_scale = self.config.get("noise_classical_scale", None)
        self.noise_scale_f_std = self.config.get("noise_scale_f_std", 1.1)
        if "return_classical" in self.config:
            # Overriding the default return_classical from config!
            self.return_classical = self.config["return_classical"]
        else:
            self.return_classical = return_classical

        self.pop_classical = self.config.get("pop_classical", False)

        self.return_formation_energy = return_formation_energy
        self.negate_force = negate_force
        self.filter_small_ions = filter_small_ions

        if self.negate_force:
            logging.info("Negating force for formation energy dataset!")

        if self.pop_classical and self.return_classical:
            raise ValueError(
                "pop_classical and return_classical cannot be used together!"
            )

        # configs = [{"src": src} for src in self.config["src"]]
        configs = [{"src": self.config["src"]}]

        self.lmbd_datasets = [LmdbDataset(c) for c in configs]

        logging.info(f"Calculating mean and std for datasets...")
        for dataset in self.lmbd_datasets:
            dataset.calculate_mean_std_energy()

        # Avoid carbon idx if we are using the bucky ball catcher spice as the val.
        self.bucky_high_fidelity_val = (len(self.lmbd_datasets) == 1 and len(self.lmbd_datasets[0]) < 20)
        logging.info(f"Using bucky high fidelity val: {self.bucky_high_fidelity_val}")
        
        if carbon_idx is not None and not self.bucky_high_fidelity_val:
            assert (
                percentages is None and filter_small_ions is False
            ), "Filtering carbon overrides other filters!"
            if val:
                carbon_idices = np.load("idx_c_val.npy")
            else:
                carbon_idices = np.load("idxs_carbon_all.npy")
                carbon_idices = carbon_idices[:carbon_idx]
            self.lmbd_datasets = [Subset(dataset, carbon_idices) for dataset in self.lmbd_datasets]
        percentages = [0.10]
        if percentages is not None:
            self.lmbd_datasets = [
                Subset(dataset, self.get_indices(len(dataset), percentage))
                for dataset, percentage in zip(self.lmbd_datasets, percentages)
            ]

        #logging.warning("FORCING FORCE NORM!")
        #config["force_norm_cutoff"] = 1.7
        #config["force_norm_flip"] = True
        flip = -1 if config.get("force_norm_flip", False) else 1
        if config.get("force_norm_cutoff", None) is not None:
            assert len(self.lmbd_datasets) == 1
            base_path = configs[0]['src']
            fn_path = f'{base_path}/force_norm_idx_cutoff={config["force_norm_cutoff"]}_flip={flip}.npy'
            
            if os.path.isfile(fn_path):
                idices_per_dataset = np.load(fn_path)
            else:
                cutoff = config["force_norm_cutoff"]
                idices_per_dataset = []

                for d in self.lmbd_datasets:
                    idx = []
                    for i, x in tqdm(enumerate(d)):
                        if x.force.norm(dim=-1).mean() * flip < cutoff * flip:
                            idx.append(i)
                    idices_per_dataset.append(idx)

                np.save(fn_path, idices_per_dataset)
            self.lmbd_datasets = [Subset(self.lmbd_datasets[i], idices_per_dataset[i]) for i in range(len(self.lmbd_datasets))]

        self.dataset = torch.utils.data.ConcatDataset(self.lmbd_datasets)
       

    def get_indices(self, length, percentage):
        idx = np.random.permutation(length)[: int(length * percentage)]

        if self.filter_small_ions:
            assert (
                len(self.lmbd_datasets) == 1
            ), "Filtering small ions only supported for single dataset!"
            # Make sure idx doesn't include small ions
            idx_to_replace = []
            logging.info("Removing small ions from formation energy dataset!")
            for i in range(len(idx)):
                obj = self.lmbd_datasets[0][idx[i]]
                if obj.natoms < 5:
                    idx_to_replace.append(i)

            logging.info(
                f"Replacing {len(idx_to_replace)} small ions with random indices!"
            )
            idx = np.delete(idx, idx_to_replace)
            for i in range(len(idx_to_replace)):
                new_idx = np.random.randint(0, length)
                while new_idx in idx:
                    new_idx = np.random.randint(0, length)
                idx = np.append(idx, new_idx)

        return idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        obj = self.dataset[idx]

        if self.return_classical:
            if self.noise_classical_scale is not None:
                obj.energy_classical = (
                    obj.energy_classical
                    + torch.randn_like(obj.energy_classical)
                    * self.noise_classical_scale
                    * obj.energy_classical_std
                )
                obj.forces_classical = (
                    obj.forces_classical
                    + torch.randn_like(obj.forces_classical)
                    * self.noise_classical_scale
                    * self.noise_scale_f_std
                )

            if (
                "forces_classical" in obj
            ):  # joint training dataset so set target to classical
                obj.force = torch.tensor(obj.forces_classical)
                obj.y = torch.tensor(obj.energy_classical)
                obj.energy_mean = obj.energy_classical_mean
                obj.energy_std = obj.energy_classical_std

        if self.pop_classical:
            if "forces_classical" in obj:
                obj.pop("forces_classical")
                obj.pop("energy_classical")
                if "energy_classical_mean" in obj:
                    obj.pop("energy_classical_mean")
                    obj.pop("energy_classical_std")

        # For spice dataset
        if self.return_formation_energy and not self.bucky_high_fidelity_val:
            mean_key = "energy_mean"
            std_key = "energy_std"
            target_key = "formation_energy"

            if "formation_energy" not in obj:
                target_key = "reference_energy"
                mean_key = "re_mean"
                std_key = "re_std"
                
            obj.y = torch.tensor(obj[target_key])
            obj.cell = (torch.eye(3) * 1000.0).unsqueeze(dim=0)
            obj.energy_mean = torch.tensor(obj[mean_key])
            obj.energy_std = torch.tensor(obj[std_key])
            obj.fixed = torch.zeros(obj.natoms, dtype=torch.bool)

            # IMPORTANT!
            # Since the preprocessing has the force gradient!!
            if self.negate_force:
                obj.force *= -1

        return obj

    def close_db(self):
        for dataset in self.lmbd_datasets:
            if type(dataset) == LmdbDataset:
                dataset.close_db()
            else:
                dataset.dataset.close_db()
