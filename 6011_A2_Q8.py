# STAT6011 Assignment 2
# Author: LAM Hin Tai
# UID: 2004062587

import pandas as pd
import numpy as np
from scipy.stats import invgamma
import matplotlib.pyplot as plt
import os

# Make it reproducible
np.random.seed(299)

dataFilePath = "P:\\__personal\\MStat\\_S4 6011\\A2"
dataFile = os.path.join(dataFilePath, "beer_data.txt")
df = pd.read_csv(dataFile, sep="\t")

# ANOVA - i = 1-6; j = 1-8, (i.e. n_i = 8 for all i)
I = max(df["Beer"])
J = max(df["Obs"])
n = np.array([J] * I)
N = np.sum(n)

# y - 6x8
y = np.array(df["Sodium"]).reshape(I, J)
yMean = np.mean(y, 1)   # vector of y_i's means
yGrandMean = np.mean(yMean)

# Q8b - One-way ANOVA with Gibbs Sampler
nIter = 100000
nBurnin = 10000

# Hyperparameters
a = 0.1
b = 0.1

# Gibbs sampler update functions for mu, alpha_i, sigmaSq, tauSq
def updateMu(currAlpha, currSigmaSq):
    muMean = yGrandMean - np.sum(np.multiply(n, currAlpha))/N
    muVar = currSigmaSq/N
    mu = np.random.normal(muMean, np.sqrt(muVar))
    return mu

def updateAlpha(currMu, currSigmaSq, currTauSq):
    alphaMean = currTauSq*(yMean - currMu) / (currSigmaSq/n + currTauSq)
    alphaVar = currSigmaSq*currTauSq / (currSigmaSq + currTauSq*n)
    alpha = np.zeros(I)

    for i in range(I):
        alpha[i] = np.random.normal(alphaMean[i], np.sqrt(alphaVar[i]))
    return alpha

def updateSigmaSq(currMu, currAlpha, a, b):
    sigmaSq_a = 0.5 * N + a
    
    # collapse the J columns by summing - result is a (Ix1) column vector
    # tempSum = np.zeros(I)
    tempSum = 0
    for j in range(J):
        for i in range(I):
            tempSum += (y[i,j] - currMu - currAlpha[i])**2
    # now take row sum across I rows
    # sigmaSq_b = 0.5 * np.sum(tempSum) + b
    sigmaSq_b = 0.5 * tempSum + b
    
    sigmaSq = invgamma.rvs(sigmaSq_a, scale=sigmaSq_b)
    return sigmaSq

def updateTauSq(currAlpha, a, b):
    tauSq_a = 0.5 * I + a
    tauSq_b = 0.5 * np.sum(np.square(currAlpha)) + b
    tauSq = invgamma.rvs(tauSq_a, scale=tauSq_b)
    return tauSq

muChain = []
sigmaSqChain = []
tauSqChain = []
# list of 6 lists
alphaChain = [[] for i in range(I)]

# Use random initial values as it's assumed to stabilize afterwards
currMu = np.random.rand()
currAlpha = np.random.rand(I)
currSigmaSq = np.random.rand()
currTauSq = np.random.rand()

# Test code
# currMu = 0.5
# currAlpha = [0.5,0.5,0.5,0.5,0.5,0.5]
# currSigmaSq = 0.5
# currTauSq = 0.5

# Gibbs sampler update step
for k in range(nIter):
    currMu = updateMu(currAlpha, currSigmaSq)
    currAlpha = updateAlpha(currMu, currSigmaSq, currTauSq)
    currSigmaSq = updateSigmaSq(currMu, currAlpha, a, b)
    currTauSq = updateTauSq(currAlpha, a, b)
    
    muChain.append(currMu)
    sigmaSqChain.append(currSigmaSq)
    tauSqChain.append(currTauSq)
    for i in range(I):
        alphaChain[i].append(currAlpha[i])

plt.plot(muChain[nBurnin:], label='MCMC chain of $\mu$')
plt.legend(fontsize=7)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.savefig('q8b_mu.png', dpi=720)
plt.show()

plt.plot(sigmaSqChain[nBurnin:], label='MCMC chain of $\sigma^{2}$')
plt.legend(fontsize=7)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.savefig('q8b_sigma.png', dpi=720)
plt.show()

plt.plot(tauSqChain[nBurnin:], label='MCMC chain of $\tau^{2}$')
plt.legend(fontsize=7)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.savefig('q8b_tau.png', dpi=720)
plt.show()

# alpha's
for i in range(I):
    plt.plot(alphaChain[i][nBurnin:], label="MCMC chain of alpha%1d" % (i+1))
    plt.legend(fontsize=7)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.savefig('q8b_alpha_%1d.png' % (i+1), dpi=720)
    plt.show()

print("Mu: mean=%f, variance=%f" % (np.mean(muChain[nBurnin:]), np.var(muChain[nBurnin:])))
print("Sigma^2: mean=%f, variance=%f" % (np.mean(sigmaSqChain[nBurnin:]), np.var(sigmaSqChain[nBurnin:])))
print("Tau^2: mean=%f, variance=%f" % (np.mean(tauSqChain[nBurnin:]), np.var(tauSqChain[nBurnin:])))
for i in range(I):
    print("Alpha_%d: mean=%f, variance=%f" % ((i+1), np.mean(alphaChain[i][nBurnin:]), np.var(alphaChain[i][nBurnin:])))

# Q8c - Use improper prior and check MCMC convergence
nIter = 1000000
nBurnin = 10000

a = 0
b = 0

muChain_c = []
sigmaSqChain_c = []
tauSqChain_c = []
# list of 6 lists
alphaChain_c = [[] for i in range(I)]

# Use random initial values as it's assumed to stabilize afterwards
currMu = np.random.rand()
currAlpha = np.random.rand(I)
currSigmaSq = np.random.rand()
currTauSq = np.random.rand()

# Gibbs sampler update step
for k in range(nIter):
    currMu = updateMu(currAlpha, currSigmaSq)
    currAlpha = updateAlpha(currMu, currSigmaSq, currTauSq)
    currSigmaSq = updateSigmaSq(currMu, currAlpha, a, b)
    currTauSq = updateTauSq(currAlpha, a, b)
    
    muChain_c.append(currMu)
    sigmaSqChain_c.append(currSigmaSq)
    tauSqChain_c.append(currTauSq)
    for i in range(I):
        alphaChain_c[i].append(currAlpha[i])

plt.plot(muChain_c[nBurnin:], label='MCMC chain of $\mu$')
plt.legend(fontsize=7)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.savefig('q8c_mu.png')
plt.show()

plt.plot(sigmaSqChain_c[nBurnin:], label='MCMC chain of $\sigma^{2}$')
plt.legend(fontsize=7)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.savefig('q8c_sigma.png')
plt.show()

plt.plot(tauSqChain_c[nBurnin:], label='MCMC chain of $\tau^{2}$')
plt.legend(fontsize=7)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.savefig('q8c_tau.png')
plt.show()

# alpha's
for i in range(I):
    plt.plot(alphaChain_c[i][nBurnin:], label="MCMC chain of alpha%1d" % (i+1))
    plt.legend(fontsize=7)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.savefig('q8c_alpha_%1d.png' % (i+1))
    plt.show()

print("Mu: mean=%f, variance=%f" % (np.mean(muChain_c[nBurnin:]), np.var(muChain_c[nBurnin:])))
print("Sigma^2: mean=%f, variance=%f" % (np.mean(sigmaSqChain_c[nBurnin:]), np.var(sigmaSqChain_c[nBurnin:])))
print("Tau^2: mean=%f, variance=%f" % (np.mean(tauSqChain_c[nBurnin:]), np.var(tauSqChain_c[nBurnin:])))
for i in range(I):
    print("Alpha_%d: mean=%f, variance=%f" % ((i+1), np.mean(alphaChain_c[i][nBurnin:]), np.var(alphaChain_c[i][nBurnin:])))
