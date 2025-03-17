import numpy as np
import pandas as pd

def recover_parameters():
    df = pd.read_csv("results/output.csv") #Reads results from simulate csv
    recovered_results = []

    for _, row in df.iterrows():
        N, T_obs, M_obs, V_obs = row["N"], row["T_obs"], row["M_obs"], row["V_obs"]

        # Adjust R_obs to prevent extreme values
        R_obs = np.clip((T_obs + 0.5) / (N + 1), 0.01, 0.99)
        L = np.log(R_obs / (1 - R_obs))

        # Ensure V_obs is not too small (avoiding division instability)
        V_obs = max(V_obs, 1e-4)

        try:
            # Compute v_est while ensuring numerical stability
            v_est = np.sign(R_obs - 0.5) * np.sqrt(L * ((R_obs**2 * L) - (R_obs * L) + (R_obs - 0.5)) / V_obs)

            # Clip v_est to stay within a reasonable range
            v_est = np.clip(v_est, 0.5, 2.0)

            a_est = L / v_est

            # Clip a_est to stay within theoretical range
            a_est = np.clip(a_est, 0.5, 2.0)

            tau_est = M_obs - (a_est / (2 * v_est)) * ((1 - np.exp(-v_est * a_est)) / (1 + np.exp(-v_est * a_est)))

            # Clip tau_est to prevent instability
            tau_est = np.clip(tau_est, 0.1, 0.5)

            recovered_results.append([N, v_est, a_est, tau_est])

        except (ValueError, FloatingPointError) as e:
            print(f"Error in computation: {e}")
            continue

    df_recovered = pd.DataFrame(recovered_results, columns=["N", "v_est", "a_est", "tau_est"])
    df_recovered.to_csv("results/recovered.csv", index=False)

if __name__ == "__main__":
    recover_parameters()

