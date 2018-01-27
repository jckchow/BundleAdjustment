#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 19:07:52 2018

@author: jckchow
"""

import numpy as np
from time import time
from matplotlib import pyplot as plt

eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/IOP_ROP/EOP.jck'
eopTruthFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1Truth.eop'

##########################################
### read in data
##########################################
eop = np.genfromtxt(eopFilename, delimiter=' ', skip_header=0, usecols = (0,1,2,3,4,5,6,7,8,9,10,11,12))
eopTruth = np.genfromtxt(eopTruthFilename, delimiter=' ', skip_header=0, usecols = (0,1,2,3,4,5,6))
