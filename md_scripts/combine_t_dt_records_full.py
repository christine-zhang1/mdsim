import re

# File paths
file1_path = 'records_gemnet_fulldt_maceoff.txt'
file2_path = 'records_gemnet_t_fulldataset.txt'
output_file = 'combined_gemnet_dt_t.txt'

# Function to parse and clean file by init_idx, keeping only GEMNET sections
def parse_and_clean_file(file_path):
    data = {}
    with open(file_path, 'r') as f:
        content = f.read()
        entries = content.split("---------------------------------------------------------")
        
        for entry in entries:
            # Remove MACEOFF CALCULATOR section if present
            cleaned_entry = re.sub(r"MACEOFF CALCULATOR.*?(?=GEMNET|$)", "", entry, flags=re.DOTALL).strip()
            
            # Extract molecule, init_idx, and store the cleaned entry
            molecule_match = re.search(r"molecule:\s+(\S+)", cleaned_entry)
            init_idx_match = re.search(r"init_idx:\s+(\d+)", cleaned_entry)
            if molecule_match and init_idx_match:
                init_idx = int(init_idx_match.group(1))
                data[init_idx] = cleaned_entry
    return data

# Parse and clean both files
gemnet_dt_data = parse_and_clean_file(file1_path)
gemnet_t_data = parse_and_clean_file(file2_path)

# Write combined data to the output file
with open(output_file, 'w') as f:
    for init_idx in gemnet_dt_data:
        # Check if there's a corresponding GEMNET-T entry
        if init_idx in gemnet_t_data:
            # Write GEMNET-DT entry
            f.write(gemnet_dt_data[init_idx] + '\n')
            
            # Filter out "molecule" and "init_idx" lines from GEMNET-T entry
            gemnet_t_lines = gemnet_t_data[init_idx].splitlines()
            filtered_gemnet_t = [
                line for line in gemnet_t_lines
                if not line.startswith("molecule:") and not line.startswith("init_idx:")
            ]
            
            # Write the filtered GEMNET-T entry
            f.write('\n'.join(filtered_gemnet_t) + '\n')
            
            # Add separator
            f.write("---------------------------------------------------------\n")

print(f"Combined data written to {output_file}")
