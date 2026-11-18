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
log_at = np.zeros(100000)
kt = np.full(100000, K_bar)
for i in range(1, 100000):
    log_at[i] = (1-rho)*np.log(A) + rho*log_at[i-1] + sigma*np.random.normal(0, 1)
    
#%%
lower = 0.1*K_bar
upper = 1.9*K_bar
