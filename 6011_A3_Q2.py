# -*- coding: utf-8 -*-
"""
STAT6011 Assignment 3 Q2
Name: LAM Hin Tai
UID: 2004062587
"""

import numpy as np
import pandas as pd
import os

# Make it reproducible
np.random.seed(123)

# Some settings
# dataFilePath = 'C:\\Users\\tai\\documents\\MStat 2018\\S4 6011 Computational Statistics\\Assignment 3'
dataFilePath = 'P:\\__personal\\MStat\\_S4 6011\\A3'
dataFile = os.path.join(dataFilePath, 'q2.csv')
y = np.array(pd.read_csv(dataFile, header=None))
n = len(y)
yBar = np.mean(y)
y_sSq = np.var(y)*n/(n-1)

# Set the min & max possible values of K
# To be a mixture distribution we need K to be at least 2
# To avoid dengenerate Gaussian, we need K to be at most n/2 to ensure any
# Gaussian component to at least generate 2 data points (so its variance >0)
minK = 2
maxK = int(np.floor(n/2))

nIterations = 5000

# Saves the history of the estimates
KList = []
phiList = []
muList = []
sigmaSqList = []

# Setup the vectors
phi = np.array([np.nan] * maxK)
mu = np.array([np.nan] * maxK)
sigmaSq = np.array([np.nan] * maxK)

# Initialize the parameters
K = np.random.randint(low=minK, high=maxK)
phi[:K] = np.random.rand(K)  # Given K, random generate the first K probability
mu[:K] = np.random.uniform(low=min(y), high=max(y), size=K)
sigmaSq[:K] = np.random.uniform(high=y_sSq)

for iIteration in range(nIterations):
    
    #
