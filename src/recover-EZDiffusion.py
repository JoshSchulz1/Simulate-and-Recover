import numpy as np
import pandas as pd

def recover_parameters():
    df = pd.read_csv("results/output.csv") #Reads results from simulate csv
    results_recover = [] #empty list for storage

    for _, row in df.iterrows():
        N, T_obs, M_obs, V_obs = row["N"], row["T_obs"], row["M_obs"], row["V_obs"]

        #Chatgpt suggested for limiting extreme value error
        R_obs = np.clip((T_obs + 0.5) / (N + 1), 0.01, 0.99)
        L = np.log(R_obs / (1 - R_obs))

        #Chatgpt fix to division by 0 problem
        V_obs = max(V_obs, 1e-4)

        try:
            # Compute v_est 
            v_est = np.sign(R_obs - 0.5) * np.sqrt(L * ((R_obs**2 * L) - (R_obs * L) + (R_obs - 0.5)) / V_obs)

            #Clip to stay in correct range
            v_est = np.clip(v_est, 0.5, 2.0)

            a_est = L / v_est 

            #Clip used again per chatgpt
            a_est = np.clip(a_est, 0.5, 2.0)
            
            #Inverse equation in slides
            tau_est = M_obs - (a_est / (2 * v_est)) * ((1 - np.exp(-v_est * a_est)) / (1 + np.exp(-v_est * a_est))) 
           
            tau_est = np.clip(tau_est, 0.1, 0.5) #stay in the bounds

            results_recover.append([N, v_est, a_est, tau_est]) #append results

        except (ValueError, FloatingPointError) as e: #Will print error if value error or floating point 
            print(f"Error: {e}")
            continue

    df_recovered = pd.DataFrame(results_recover, columns=["N", "v_est", "a_est", "tau_est"]) #file structure
    df_recovered.to_csv("results/recovered.csv", index=False)

if __name__ == "__main__":
    recover_parameters()

