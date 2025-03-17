import numpy as np
import scipy.stats as stats
import pandas as pd 
import os


A_RANGE = (0.5, 2) #Boundary Seperation Range
V_RANGE = (0.5, 2) #Drift rate range
T_RANGE = (0.1, 0.5) #NonDescision Time


N_VALUES = [10, 40, 4000] #Sample sizes N for 3x1000 iterations
ITERATIONS = 1000

def simulate_ez_diffusion():
    results = []

    for N in N_VALUES:
        for _ in range(ITERATIONS):
            a = np.random.uniform(*A_RANGE)
            v = np.random.uniform(*V_RANGE)
            t = np.random.uniform(*T_RANGE)  #Using np random to generate random parameters
            
            y = np.exp(-a * v) #Definition of y for Forward EZ Diffusion equations from slides
            R_pred = 1 / (1 + y)
            M_pred = t + (a / (2 * v)) * ((1 - y) / (1 + y))
            V_pred = (a / (2 * v**3)) * (1 - (2 * a * v * y + y**2) / (1 + y)**2) #EZ Diffusion equations from slides

            T_obs = np.random.binomial(N, R_pred)
            M_obs = np.random.normal(M_pred, np.sqrt(V_pred / N))
            V_obs = stats.gamma.rvs((N - 1) / 2, scale=(2 * V_pred / (N - 1)))

            results.append([N, a, v, t, T_obs, M_obs, V_obs])

    df = pd.DataFrame(results, columns=["N", "a", "v", "t", "T_obs", "M_obs", "V_obs"])
    os.makedirs("results", exist_ok=True) #Suggestion from Chatgpt for debugging

    df.to_csv("results/output.csv", index=False) #Chatgpt wrote a path to create output.csv to store data

if __name__ == "__main__":
    simulate_ez_diffusion()
