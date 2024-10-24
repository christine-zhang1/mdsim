import re
import csv

# Function to parse the file and extract data
def parse_file_to_csv(input_file, output_csv):
    with open(input_file, 'r') as file:
        data = file.read()

    # Regular expressions to extract the relevant fields
    molecule_pattern = re.compile(r'molecule:\s+(\S+)')
    init_idx_pattern = re.compile(r'init_idx:\s+(\d+)')

    energy_pattern_gemnet = re.compile(r'GEMNET-DT CHECKPOINT\ntraj\[0\] potential energy: ([\-\d\.]+)\ntraj\[\-1\] potential energy: ([\-\d\.]+)\nenergy is off by ([\-\d\.]+) meV')
    energy_pattern_maceoff = re.compile(r'GEMNET-DT CHECKPOINT\ntraj\[0\] potential energy: ([\-\d\.]+)\ntraj\[\-1\] potential energy: ([\-\d\.]+)\nenergy is off by ([\-\d\.]+) meV')

    # Find all molecule sections and store the relevant information
    molecules = molecule_pattern.findall(data)
    init_idxs = init_idx_pattern.findall(data)
    gemnet_data = energy_pattern_gemnet.findall(data)
    maceoff_data = energy_pattern_maceoff.findall(data)

    # Write the data to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t')
        # Write the header row
        csvwriter.writerow([
            'molecule', 'init_idx', 
            'gemnet_dT_traj0_energy', 'gemnet_dT_trajN_energy', 'gemnet_dT_energy_off_by_meV', 
            'gemnet_T_traj0_energy', 'gemnet_T_trajN_energy', 'gemnet_T_energy_off_by_meV'
        ])

        # Write each row of data
        for i in range(len(molecules)):
            csvwriter.writerow([
                molecules[i], init_idxs[i], 
                gemnet_data[i][0], gemnet_data[i][1], gemnet_data[i][2],
                maceoff_data[i][0], maceoff_data[i][1], maceoff_data[i][2]
            ])

# Define the input and output file paths
input_file = 'records_gemnet_t_dt.txt'  # Path to the input file you provided
output_csv = 'records_gemnet_t_dt.csv'  # Path where you want the CSV output

# Run the function to convert the file
parse_file_to_csv(input_file, output_csv)
