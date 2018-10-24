# STAT6011 Assignment 2
# Author: LAM Hin Tai
# UID: 2004062587

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma, lognorm

# Make it reproducible
np.random.seed(123)

# Q3a - SIR
# Sample from normal distribution and transfrom to log-normal
jNormSamples = 100000
mGammaSamples = 10000
mu = 0
sigma2 = 4
n_samples = norm.rvs(mu, np.sqrt(sigma2), size=jNormSamples)
ln_samples = np.exp(n_samples)

print(ln_samples[0:10])
print(n_samples[0:10])

# Get the weights from ratio of the pdfs
alpha = 2
beta = 0.5
w = gamma.pdf(ln_samples, alpha, scale=1/beta) / lognorm.pdf(ln_samples, s=np.sqrt(sigma2), scale=np.exp(mu))
W = w / np.sum(w)

re_sample = np.random.choice(ln_samples, size=mGammaSamples, replace=False, p=W)

# Plot histogram and overlay with pdf
plt.hist(re_sample, bins=100, normed=True)
X = np.linspace(0, 20, 1000)
plt.plot(X, gamma.pdf(X, alpha, scale=1/beta), label="f(x)=Gamma(2,1/2)")
plt.plot(X, lognorm.pdf(X, s=np.exp(mu), scale=np.sqrt(sigma2)), label="g(x)=lognormal(0,4)")
plt.xlabel("x")
plt.ylabel("pdf")
plt.legend()
plt.title("SIR")
plt.show

# Q3b - Laplace approximation
# Exact probability from cdf
def diff_cdf(x_mode, dist):
    d = gamma.cdf(x_mode+dist, alpha, scale=1/beta) - gamma.cdf(x_mode-dist, alpha, scale=1/beta)
    return d

# Approximate probability by Laplace approximation
def diff_approx(x_mode, dist):
    a = #
    return a
