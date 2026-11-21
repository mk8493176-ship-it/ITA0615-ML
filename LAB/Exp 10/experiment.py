# Experiment 10: Expectation-Maximization for a 2-component Gaussian Mixture (1-D)
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(42)
data = np.concatenate([rng.normal(2, 1.0, 300), rng.normal(7, 1.5, 200)])

mu = np.array([1.0, 5.0]); sd = np.array([1.0, 1.0]); pi = np.array([0.5, 0.5])   # initial guess

for it in range(1, 31):
    # E-step: responsibilities
    r = np.array([pi[k] * norm.pdf(data, mu[k], sd[k]) for k in range(2)])
    r /= r.sum(axis=0)
    # M-step: update parameters
    Nk = r.sum(axis=1)
    mu = (r * data).sum(axis=1) / Nk
    sd = np.sqrt((r * (data - mu[:, None]) ** 2).sum(axis=1) / Nk)
    pi = Nk / len(data)
    if it in (1, 5, 10, 20, 30):
        print(f"Iter {it:2d}: mu = {mu.round(3)}  sigma = {sd.round(3)}  weights = {pi.round(3)}")

print("\nTrue values : mu = [2, 7]  sigma = [1.0, 1.5]  weights = [0.6, 0.4]")
