# STAT6011 Assignment 2
# Author: LAM Hin Tai
# UID: 2004062587

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy, gamma, poisson

# Make it reproducible
np.random.seed(123)

nSamples = 10**6
# Q4a - Gibbs sampler for Poisson process
lambda = 1

# Q4b - Gibbs sampler for (Cauchy, Normal)
