import numpy as np
import matplotlib.pyplot as plt
import os

# Sample data
original_mi = [0.5020857130361625, 0.29415375190871995, 0.9364613699660129, 0.9787088939029014]
sample_sizes = [150.0, 487.5, 825.0, 1162.5, 1500.0]
mi_scores = [
    [0.5076628683312917, 0.28893289079474627, 0.9392745340717175, 0.9770430257841828],
    [0.5076628683312917, 0.28893289079474627, 0.9392745340717175, 0.9770430257841828],
    [0.5076628683312917, 0.28893289079474627, 0.9392745340717175, 0.9770430257841828],
    [0.5076628683312917, 0.28893289079474627, 0.9392745340717175, 0.9770430257841828],
    [0.5076628683312917, 0.28893289079474627, 0.9392745340717175, 0.9770430257841828]
]

# Convert the data into a format suitable for plotting
mi_scores = np.array(mi_scores).T  # Transpose to have each row represent a feature's MI scores

# Plotting the data
plt.figure(figsize=(10, 6))

for i, original_score in enumerate(original_mi):
    plt.plot(sample_sizes, mi_scores[i], marker='o', label=f'Feature {i+1}')
    plt.axhline(y=original_score, color='gray', linestyle='--', label=f'Original MI Feature {i+1}')

# Customizing the plot
plt.title('Mutual Information Scores Across Different Sample Sizes')
plt.xlabel('Sample Size')
plt.ylabel('Mutual Information Score')
plt.legend()
plt.grid(True)

# Save the plot in the results folder
results_folder = './results'
os.makedirs(results_folder, exist_ok=True)  # Create results folder if it doesn't exist
output_file = os.path.join(results_folder, 'mi_scores_plot.png')
plt.savefig(output_file)

# Show the plot
plt.show()
