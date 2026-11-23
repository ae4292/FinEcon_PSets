#%%
import numpy as np

alpha = 0.33
delta = 0.02
beta = 0.99
sigma = 0.01
rho = 0.95
A = 1.0

K_bar = ((1-beta*(1-delta))/(beta*alpha*A)) **(1/(alpha-1))
# %%
np.random.seed(42)
from quantecon.markov.approximation import tauchen
mc = tauchen(7, rho, sigma, mu=(1-rho)*np.log(A), n_std=3)
logA_grid = mc.state_values   
A_grid = np.exp(logA_grid)       
P_mat = mc.P                         

#%%
lower = 0.1*K_bar
upper = 1.9*K_bar
Kvec= np.linspace(lower, upper, 1000)

k_curr = Kvec[:,None,None]     # axis 0: today's K
A_curr = A_grid[None,:,None]   # axis 1: today's A
k_next = Kvec[None,None,:]     # axis 2: tomorrow's K'

dividend = A_curr*k_curr**alpha + (1-delta)*k_curr - k_next   # (1000, 7, 1000)
V = np.zeros((len(Kvec), len(A_grid)))
diff = 1.0

while diff > 1e-6:
    EV = V @ P_mat.T                                  # (K', A): E[V(K',A')|A]
    objective = dividend + beta * EV.T[None, :, :]    # (K, A, K')
    V_new = np.max(objective, axis=2)
    diff = np.max(np.abs(V - V_new))
    V = V_new

EV = V @ P_mat.T
objective = dividend + beta * EV.T[None, :, :]
optimal_policy_idx = np.argmax(objective, axis=2)     # (K, A) index of best K'
k_optimal = Kvec[optimal_policy_idx]                  # (K, A) value of best K'
# %%
A_idx  = mc.simulate_indices(ts_length=100000, init=3) 
K_idx = np.empty(len(A_idx), dtype=int)
first_K = np.argmin(np.abs(Kvec - K_bar))
K_idx[0] = first_K
for t in range(1, len(A_idx)):
    K_idx[t] = optimal_policy_idx[K_idx[t-1], A_idx[t-1]]
optimal_K = Kvec[K_idx]
a_vec = A_grid[A_idx[:-1]]  # (T-1,)
investment = optimal_K[1:] - (1-delta)*optimal_K[:-1]  # (T-1,)
div = a_vec * optimal_K[:-1]**alpha - investment  # (T-1,)
# %%
print(f'Correlation between A and I: {np.corrcoef(a_vec, investment)[0,1]:.4f}')
print(f'Correlation between A and D: {np.corrcoef(a_vec, div)[0,1]:.4f}')

# %%
print(f'risk free rate: {1/beta - 1:.4f}')

# %%
returns = V[K_idx[1:], A_idx[1:]] / (V[K_idx[:-1], A_idx[:-1]] - div)
excess_returns = 1 + returns - 1/beta -1
print(f'average excess return: {np.mean(excess_returns):.4f}')