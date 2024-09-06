import pandas as pd
import matplotlib.pyplot as plt
import glob

# Step 1: Load the data from multiple log files
file_list = sorted(glob.glob("*.log"))  # Adjust the path to your log files
all_data = []

for file in file_list:
    df = pd.read_csv(file, sep='\\s+')  # Assuming the log files are space-delimited
    all_data.append(df['Etot/N[eV]'])  # Extract the 2nd column

# Step 2: Plot the data
plt.figure(figsize=(10, 6))

for i, data in enumerate(all_data):
    plt.plot(data, label=file_list[i])  # Plot each file's data as a line on the graph

plt.xlabel('Time Step')
plt.ylabel('Etot/N[eV]')
plt.title('Etot/N[eV] vs. Time')
plt.legend()
plt.grid(True)
plt.title("gemnet-T energy drift")
# plt.show()

plt.savefig('etot_vs_time_plot.png', format='png', dpi=300)  # Save as PNG with 300 DPI
