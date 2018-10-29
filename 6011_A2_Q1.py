# STAT6011 Assignment 2
# Author: LAM Hin Tai
# UID: 2004062587

import numpy as np
from scipy import stats
from scipy.stats import laplace, norm, gamma, invgamma, t
import matplotlib.pyplot as plt

# Make it reproducible
np.random.seed(299)

# Q1c - Direct sample from Laplace & sample from Normal conditioned on Gamma
nSamples = 10000

a = 1
b = 1/2

directSample = laplace.rvs(size=nSamples)
gammaSample = gamma.rvs(a=a, scale=1/b, size=nSamples)
condSample = norm.rvs(0, scale=np.sqrt(gammaSample), size=nSamples)

directSampleKDE = stats.gaussian_kde(directSample)
condSampleKDE = stats.gaussian_kde(condSample)
X = np.linspace(-10, 10, 1000)

plt.hist(directSample, bins=100, density=True, label="Sample from Laplace", alpha=0.5)
plt.hist(condSample, bins=100, density=True, label="Sample from Normal|Gamma", alpha=0.5)
plt.plot(X, directSampleKDE(X), label="Laplace sample KDE", alpha=0.7)
plt.plot(X, condSampleKDE(X), label="Conditional Normal sample KDE", alpha=0.7)
plt.legend(fontsize=7)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig('q1c.png',dpi=720)
plt.show()

# Q1e - Direct sample from t(2) & sample from Normal conditioned on Gamma
b = 2

directSample2 = laplace.rvs(size=nSamples)
igSample = invgamma.rvs(a=b/2, scale=b/2, size=nSamples)
condSample2 = norm.rvs(0, scale=np.sqrt(igSample), size=nSamples)

directSample2KDE = stats.gaussian_kde(directSample2)
condSample2KDE = stats.gaussian_kde(condSample2)
X = np.linspace(-10, 10, 1000)

plt.hist(directSample2, bins=100, density=True, label="Sample from t(2)", alpha=0.5)
plt.hist(condSample2, bins=100, density=True, label="Sample from Normal|Gamma", alpha=0.5)
plt.plot(X, directSample2KDE(X), label="t2 sample KDE", alpha=0.7)
plt.plot(X, condSample2KDE(X), label="Conditional Normal sample KDE", alpha=0.7)
plt.legend(fontsize=7)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig('q1e.png',dpi=720)
plt.show()

