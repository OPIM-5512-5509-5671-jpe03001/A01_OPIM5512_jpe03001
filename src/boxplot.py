import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing

# 1. Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# 2. Features + target as a single DataFrame
df = housing.frame

# 3. Quick check
print(df.head())
print(df.shape)

# 4. Ensure the output directory exists
os.makedirs("figs", exist_ok=True)

# 5. Generate the boxplot for Median Income (MedInc)
plt.figure(figsize=(8, 6))
plt.boxplot(df["MedInc"])
plt.title("Boxplot of California Housing - Median Income (MedInc)")
plt.ylabel("Median Income ($10,000s)")

# 6. Save the figure to figs/boxplot.png
plt.savefig("figs/boxplot.png", bbox_inches="tight")
plt.close()

print("Boxplot successfully saved to figs/boxplot.png")