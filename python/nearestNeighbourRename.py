# -*- coding: utf-8 -*-
"""
/////////////////////////////////////////////////////////////////////////////
//
//   Project/Path:      %M%
//   Last Change Set:   %L% (%G% %U%)
//
/////////////////////////////////////////////////////////////////////////////
//
//   COPYRIGHT Vusion Technologies, all rights reserved.
//
//   No part of this software may be reproduced or modified in any
//   form or by any means - electronic, mechanical, photocopying,
//   recording, or otherwise - without the prior written consent of
//   Vusion Technologies.
//

@author: jacky.chow
"""

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


import numpy as np
from sklearn.externals import joblib

inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'

print ("-----------Copy previous K-Nearest Neighbour Model to current KNN model-----------")

##########################################
### read in the residuals output from bundle adjustment
# x, y, v_x, v_y, redu_x, redu_y, vStdDev_x, vStdDev_y
##########################################
image = np.genfromtxt(inputFilename, delimiter=' ', skip_header=0, usecols = (2,3,4,5,6,7,8,9,10))

sensorsUnique = np.unique(image[:,0])
##########################################
### Try to load preprocessing and ML model if it exists from previous iteration to make a copy for when the iterations end before the max iterations then we should use the previous model instead
##########################################
try:
    for iter in range(0,len(sensorsUnique)):
        sensorID = sensorsUnique[iter] #currently sensor ID
        # load the preprocessing info
        [min_x, min_y, max_x, max_y, desire_min, desire_max, mean_label] = joblib.load('/home/jckchow/BundleAdjustment/build/NNPreprocessing'+str(sensorID.astype(int))+'Temp.pkl')
        # load the learned NN model
        reg = joblib.load('/home/jckchow/BundleAdjustment/build/NNModel'+str(sensorID.astype(int))+'Temp.pkl')
        # save copy of previous preprocessing
        joblib.dump([min_x, min_y, max_x, max_y, desire_min, desire_max, mean_label], '/home/jckchow/BundleAdjustment/build/NNPreprocessing'+str(sensorID.astype(int))+'.pkl')
        # save the previously learned NN model
        joblib.dump(reg, '/home/jckchow/BundleAdjustment/build/NNModel'+str(sensorID.astype(int))+'.pkl')
        print('Found previous ML preprocessing and model found, done copying NNModel'+str(sensorID.astype(int))+'Temp.pkl to NNModel'+str(sensorID.astype(int))+'.pkl')
except:
    print('No previous ML preprocessing and model found, not copying/renaming')