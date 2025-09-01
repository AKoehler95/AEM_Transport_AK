import re
import matplotlib.pyplot as plt

# Path to your file
file_path = "output_values_O2_05.txt"

# Extract alpha_t values using regex
alpha_t_values = []
with open(file_path, "r") as file:
    for line in file:
        match = re.search(r"αₜ=([\d\.]+)", line)
        if match:
            alpha_t_values.append(float(match.group(1)))

# Remove all values above 0.1
alpha_t_values = [val for val in alpha_t_values if val <= 0.1]

# Histogram (frequency plot with 4 decimals on x-axis)
plt.figure(figsize=(7,5))
plt.hist(alpha_t_values, bins=20, edgecolor="black")
plt.title("Frequency Plot of αₜ Values (≤ 0.1)")
plt.xlabel("αₜ")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.gca().set_xticklabels([f"{x:.4f}" for x in plt.gca().get_xticks()])
plt.tight_layout()
plt.show()

# Scatter plot (index vs alpha_t)
plt.figure(figsize=(7,5))
plt.scatter(range(len(alpha_t_values)), alpha_t_values, c="blue")
plt.title("Scatter Plot of αₜ Values (≤ 0.1)")
plt.xlabel("Index")
plt.ylabel("αₜ")
plt.tight_layout()
plt.show()

# Box plot
plt.figure(figsize=(7,5))
plt.boxplot(alpha_t_values, vert=True)
plt.title("Box Plot of αₜ Values (≤ 0.1)")
plt.ylabel("αₜ")
plt.tight_layout()
plt.show()
