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
plt.hist(re_sample, bins=100, normed=True, label="SIR sample histogram")
X = np.linspace(0, 20, 1000)
plt.plot(X, gamma.pdf(X, alpha, scale=1/beta), label="f(x)=Gamma(2,1/2)")
plt.plot(X, lognorm.pdf(X, s=np.exp(mu), scale=np.sqrt(sigma2)), label="g(x)=lognormal(0,4)")
plt.xlabel("x")
plt.ylabel("pdf")
plt.legend()
plt.title("SIR")
plt.savefig('q3a.png',dpi=720)
plt.show

# Q3b - Laplace approximation
# Exact probability from cdf
def diff_cdf(x_mode, dist):
    area = gamma.cdf(x_mode+dist, alpha, scale=1/beta) - gamma.cdf(x_mode-dist, alpha, scale=1/beta)
    return area

# Approximate probability by Laplace approximation
# h(x)   = ln(f(x)) = ln(1/4) + ln(x) - x/2
# h'(x)  = 1/x -1/2 ; Set to 0 => x_mode = 2
# h''(x) = -1/x^2
# h''(x_mode=2) = -1/4
# Normal > mean = x_mode, variance 4
negD2_h_x_mode = -0.25
def diff_approx(x_mode, dist):
    normVar = -1/negD2_h_x_mode
    normArea = norm.cdf(x_mode+dist, x_mode, np.sqrt(normVar)) - \
               norm.cdf(x_mode-dist, x_mode, np.sqrt(normVar))
    area = gamma.pdf(x_mode, alpha, scale=1/beta)*np.sqrt(2*np.pi/-negD2_h_x_mode)*normArea
    return area

def Approx_Error(x_mode, dist):
    err = diff_cdf(x_mode, dist) - diff_approx(x_mode, dist)
    return err

x_mode = 2
Err = []
Dist = []
# Dist should be [1,3], centered around x_mode=2
for dist in np.linspace(0, 1, 100):
    Dist.append(dist)
    Err.append(Approx_Error(x_mode, dist))

# Plot of the Gamma(2, 1/2) pdf and the Laplace approximation to it
normVar = -1/negD2_h_x_mode
scale = gamma.pdf(x_mode, alpha, scale=1/beta)*np.sqrt(2*np.pi/-negD2_h_x_mode)
X = np.linspace(0, 20, 1000)
ax = plt.subplot()
ax.plot(X, gamma.pdf(X, alpha, scale=1/beta), label="f(x)=Gamma(2,1/2)")
plt.plot(X, scale*norm.pdf(X, 2, np.sqrt(normVar)), label="Normal(2,4)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.axvline(x=2)
plt.savefig('q3a.png',dpi=720)
plt.title('Gamma(2,1/2) pdf and Normal(2,4) pdf')
plt.show()

# Plot of the approximation error against distance away from the mode (moving away symmetrically)
plt.plot(Dist, np.abs(Err))
plt.xlabel('sigma')
plt.ylabel('Approximation Error')
plt.title('Laplace Approximation Error for Gamma(2,1/2) Distribution')
plt.show()

# Integration from 1 to 3 based on cdf
print(diff_cdf(x_mode=x_mode, dist=1))
# Laplace approxmation to integration from 1 to 3
print(diff_approx(x_mode=x_mode, dist=1))
