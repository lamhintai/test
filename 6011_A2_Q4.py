# STAT6011 Assignment 2
# Author: LAM Hin Tai
# UID: 2004062587

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy

# Make it reproducible
np.random.seed(123)

nSamples = 10**6
nBurnin = 100
nIter = nBurnin + nSamples

# Q4a - Gibbs sampler for truncated Poisson process and waiting time
# Hyperparameters - given
Lambda = 1

# Zero Truncated Poisson
def randomZtp(Lambda):
    r = 0
    while r < 1:
        r = np.random.poisson(lam=Lambda)
    return r


# Initialization
R_T = 1
T_R = 1
chain_R = []
chain_T = []

# Gibbs sampler step
for k in range(nIter):
    # R|T
    R_T = randomZtp(Lambda*T_R)
    
    # T|R
    T_R = np.random.gamma(R_T, Lambda)
    
    chain_R.append(R_T)
    chain_T.append(T_R)

sample_R = chain_R[nBurnin:]
sample_T = chain_T[nBurnin:]

# plot MC chain
plt.subplot(411)
plt.plot(chain_R, label='MCMC chain of R*')
plt.legend(fontsize=7)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

plt.subplot(412)
plt.plot(chain_T, label='MCMC chain of T')
plt.legend(fontsize=7)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

plt.subplot(413)
plt.hist(sample_R, bins=100, label='Conditional distribution of R*')
plt.legend(fontsize=7)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

plt.subplot(414)
plt.hist(sample_T, bins=100, label='Conditional distribution of T*')
plt.legend(fontsize=7)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

plt.tight_layout(pad=0.5)

plt.savefig('q4a.png',dpi=720)
plt.show()

# Q4b - Gibbs sampler for (Cauchy, Normal)
# Make it reproducible
np.random.seed(778)

# Hyperparameters - given
normVar = 1
cauchyScale = 1

# Initialization
X_Y = 0
Y_X = 0
chain_X = []
chain_Y = []

# Gibbs sampler step
for k in range(nIter):
    # X|Y
    X_Y = cauchy.rvs(Y_X, cauchyScale)
    
    # Y|X
    Y_X = np.random.normal(X_Y, np.sqrt(normVar))
    
    chain_X.append(X_Y)
    chain_Y.append(Y_X)

sample_X = chain_X[nBurnin:]
sample_Y = chain_Y[nBurnin:]

# plot MC chain
plt.subplot(411)
plt.plot(chain_X, label='MCMC chain of X')
plt.legend(fontsize=7)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

plt.subplot(412)
plt.plot(chain_Y, label='MCMC chain of Y')
plt.legend(fontsize=7)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

plt.subplot(413)
plt.hist(sample_X, bins=100, label='Conditional distribution of X')
plt.legend(fontsize=7)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

plt.subplot(414)
plt.hist(sample_Y, bins=100, label='Conditional distribution of Y')
plt.legend(fontsize=7)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

plt.tight_layout(pad=0.5)

plt.savefig('q4b.png',dpi=720)
plt.show()
