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
dataFilePath = 'C:\\Users\\tai\\documents\\MStat 2018\\S4 6011 Computational Statistics\\Assignment 3'
dataFile = os.path.join(dataFilePath, 'q2.csv')
y = np.array(pd.read_csv(dataFile, header=None))



