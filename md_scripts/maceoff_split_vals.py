from tqdm import tqdm
from mdsim.datasets.lmdb_dataset import LmdbDataset

test_dataset = LmdbDataset({'src': '/data/shared/spice/maceoff_split/train'})

with open("maceoff_split_ref_energies.txt", "w") as file:
    # Loop through the dataset with tqdm to show progress
    for i in tqdm(range(len(test_dataset))):
        # Extract the reference energy value
        val = test_dataset[i].reference_energy[0].item()

        # Write the value to the file, adding a newline for each entry
        file.write(f"{val}\n")
