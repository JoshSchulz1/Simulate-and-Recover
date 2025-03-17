import pandas as pd

df_simulated = pd.read_csv("results/output.csv")
df_recovered = pd.read_csv("results/recovered.csv")  # Load CSV files

# Remove rows with NaN values for relevant columns
df_simulated = df_simulated.dropna(subset=["a", "v", "t"])
df_recovered = df_recovered.dropna(subset=["a_est", "v_est", "tau_est"])

# Convert columns to numeric (in case they were read as strings)
df_simulated["a"] = pd.to_numeric(df_simulated["a"], errors='coerce')
df_simulated["v"] = pd.to_numeric(df_simulated["v"], errors='coerce')
df_simulated["t"] = pd.to_numeric(df_simulated["t"], errors='coerce')

df_recovered["a_est"] = pd.to_numeric(df_recovered["a_est"], errors='coerce')
df_recovered["v_est"] = pd.to_numeric(df_recovered["v_est"], errors='coerce')
df_recovered["tau_est"] = pd.to_numeric(df_recovered["tau_est"], errors='coerce')

# Merge the two DataFrames based on "N"
df_merged = df_simulated[["N", "a", "v", "t"]].merge(df_recovered, on="N", suffixes=("_true", "_est"))

# Drop any remaining NaN values after the merge
df_merged = df_merged.dropna()

# Group by 'N' to calculate bias and squared error for each N size
for N_value in [10, 40, 4000]:
    df_n = df_merged[df_merged["N"] == N_value]
    
    # Calculate bias for each parameter
    bias_a = df_n["a_est"].mean() - df_n["a"].mean()
    bias_v = df_n["v_est"].mean() - df_n["v"].mean()
    bias_tau = df_n["tau_est"].mean() - df_n["t"].mean()

    # Calculate squared error for each parameter
    squared_error_a = ((df_n["a_est"] - df_n["a"]) ** 2).mean()
    squared_error_v = ((df_n["v_est"] - df_n["v"]) ** 2).mean()
    squared_error_tau = ((df_n["tau_est"] - df_n["t"]) ** 2).mean()

    # Print results for this specific N value
    print(f"\nResults for N = {N_value}")
    print("Bias")
    print(f"Bias for a: {bias_a}")
    print(f"Bias for v: {bias_v}")
    print(f"Bias for tau: {bias_tau}")

    print("\nSquared Error")
    print(f"Squared Error for a: {squared_error_a}")
    print(f"Squared Error for v: {squared_error_v}")
    print(f"Squared Error for tau: {squared_error_tau}")


