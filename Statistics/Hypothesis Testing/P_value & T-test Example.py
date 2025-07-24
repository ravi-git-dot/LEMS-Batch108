import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# X-axis range (test scores from 50 to 100)
x = np.linspace(50, 100, 1000)

# Group A: 8 hours sleep, mean score = 85, std dev = 3
mean_A, std_A = 85, 3
y_A = norm.pdf(x, mean_A, std_A)

# Group B: 4 hours sleep, mean score = 70, std dev = 3
mean_B, std_B = 70, 3
y_B = norm.pdf(x, mean_B, std_B)

# Plot the normal distributions
plt.figure(figsize=(10, 6))
plt.plot(x, y_A, label='Group A (8 hrs sleep)', color='blue')
plt.plot(x, y_B, label='Group B (4 hrs sleep)', color='red')

# Vertical lines for means
plt.axvline(mean_A, color='blue', linestyle='--', linewidth=1)
plt.axvline(mean_B, color='red', linestyle='--', linewidth=1)

# Labels and title
plt.title("Test Score Distributions: 8 hrs Sleep vs 4 hrs Sleep")
plt.xlabel("Test Score")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
