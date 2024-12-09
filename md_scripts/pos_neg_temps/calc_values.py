import re

# Function to parse the file and extract the energy information
def parse_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()

    # Regular expression to match each molecule block
    molecule_pattern = re.compile(r'''
        molecule:\s+(\w+)                    # Extract the molecule name
        .*?                                  # Skip until
        GEMNET-DT\sCHECKPOINT.*?             # Match the GEMNET-DT section
        energy\sis\soff\sby\s([-\d.]+)\smeV  # Extract GEMNET-DT energy off value
        .*?                                  # Skip until
        MACEOFF\sCALCULATOR.*?               # Match the MACEOFF section
        energy\sis\soff\sby\s([-\d.]+)\smeV  # Extract MACEOFF energy off value
    ''', re.DOTALL | re.VERBOSE)

    options = ['both_pos', 'both_neg', 'only_gemnet_neg', 'only_maceoff_neg']
    # Dictionary to store the results
    results = {op: [] for op in options}

    # Find all matches for molecule blocks
    for match in molecule_pattern.finditer(data):
        molecule = match.group(1)  # Molecule name
        gemnet_energy_off = float(match.group(2))  # GEMNET-DT energy off
        maceoff_energy_off = float(match.group(3))  # MACEOFF energy off

        if gemnet_energy_off > 0 and maceoff_energy_off > 0:
            results['both_pos'].append(molecule)
        elif gemnet_energy_off < 0 and maceoff_energy_off < 0:
            results['both_neg'].append(molecule)
        elif gemnet_energy_off < 0 and maceoff_energy_off > 0:
            results['only_gemnet_neg'].append(molecule)
        elif gemnet_energy_off > 0 and maceoff_energy_off < 0:
            results['only_maceoff_neg'].append(molecule)

    return results

# File path to the input file
file_path = 'records_str_atoms_leq_6.txt'

# Parse the file and get results
molecule_results = parse_file(file_path)

with open('pos_neg_res.txt', 'w') as f:
    for option, molecules in molecule_results.items():
        f.write(f'{option}\n')
        f.write('\n'.join(molecules))
        f.write('\n\n')
