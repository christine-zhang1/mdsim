import re

# Paths to the input files
# potential_energy_file = 'records_gemnet_t_dt_100k.txt'
# kinetic_energy_file = 'records_gemnet_t_dt_100k_with_ke.txt'
# output_file = 'total_energy_deviation_100k.txt'

potential_energy_file = 'combined_gemnet_dt_t_full.txt'
kinetic_energy_file = 'records_gemnet_t_dt_full_ke.txt'
output_file = 'total_energy_deviation_full.txt'

gemnet_t = True
if gemnet_t:
    grad_string = "GEMNET-T CHECKPOINT"
else:
    grad_string = "MACEOFF CALCULATOR"

# Function to parse "energy is off by" values for each model from a file
def parse_energy_off(file_path):
    energy_off_values = {}
    with open(file_path, 'r') as file:
        content = file.read()
        # Regex to match 'init_idx' and "energy is off by" values for each model
        matches = re.findall(rf'init_idx: (\d+).*?GEMNET-DT CHECKPOINT\s.*?energy is off by ([\d\.\-e]+) meV.*?{grad_string}\s.*?energy is off by ([\d\.\-e]+) meV', content, re.DOTALL)
        # Populate the dictionary
        for init_idx, dt_off, t_off in matches:
            if init_idx == '1871' or init_idx == '12088':
                continue
            energy_off_values[int(init_idx)] = {
                'GEMNET-DT': float(dt_off),
                'grad_model': float(t_off)
            }
    return energy_off_values

# Parse both files
potential_energy_off = parse_energy_off(potential_energy_file)
kinetic_energy_off = parse_energy_off(kinetic_energy_file)

# Initialize sums and counts for averages
total_dt_sum = 0
total_t_sum = 0
count = 0

# Calculate total "energy is off by" values for each model and write to a new file
with open(output_file, 'w') as file:
    file.write(f"init_idx\tTotal Energy Off By GEMNET-DT (meV)\tTotal Energy Off By {grad_string} (meV)\n")  # Header
    for init_idx in potential_energy_off:
        if init_idx in kinetic_energy_off:
            # Sum the "energy is off by" values for GEMNET-DT and GEMNET-T separately
            total_dt_off_by = potential_energy_off[init_idx]['GEMNET-DT'] + kinetic_energy_off[init_idx]['GEMNET-DT']
            total_t_off_by = potential_energy_off[init_idx]['grad_model'] + kinetic_energy_off[init_idx]['grad_model']
            
            # Write to file
            file.write(f"{init_idx}\t{total_dt_off_by:.6f} meV\t{total_t_off_by:.6f} meV\n")
            
            # Update totals for averages
            total_dt_sum += abs(total_dt_off_by)
            total_t_sum += abs(total_t_off_by)
            count += 1

    # Calculate averages
    avg_dt_off_by = total_dt_sum / count if count > 0 else 0
    avg_t_off_by = total_t_sum / count if count > 0 else 0

    # Write averages to file
    file.write("\nAverage Energy Off By Values:\n")
    file.write(f"GEMNET-DT: {avg_dt_off_by:.6f} meV\n")
    file.write(f"{grad_string}: {avg_t_off_by:.6f} meV\n")
    file.write(f"Total count: {count}\n")

print("Total 'energy is off by' values and averages for each model have been written to", output_file)