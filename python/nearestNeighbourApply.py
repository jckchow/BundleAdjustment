#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 18:27:18 2018

@author: jckchow
"""


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
import numpy as np
from time import time
from sklearn import neighbors
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.mlab import griddata
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.colors import ListedColormap

##################################
### User defined parameters
##################################   
#inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'
#phoFilename = '/home/jckchow/BundleAdjustment/Data/Dcs28mmTemp.pho'
#iopFilename = '/home/jckchow/BundleAdjustment/Data/Dcs28mm.iop'
#eopFilename = '/home/jckchow/BundleAdjustment/Data/Dcs28mm.eop'

#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1Training.pho'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1Training.eop'
#outputFilename = '/home/jckchow/BundleAdjustment/xrayData1/NewResults/xray1Training_CalibratedSeparate.pho'

#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1TrainingA.pho'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1TrainingA.eop'
#outputFilename = '/home/jckchow/BundleAdjustment/xrayData1/NewResults/xray1TrainingA_CalibratedA.pho'

#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1Training.pho'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1Training.eop'
#outputFilename = '/home/jckchow/BundleAdjustment/xrayData1/IOP/xray1TrainingAB_CalibratedAB_IOP.pho'
#NNModelFilename = '/home/jckchow/BundleAdjustment/xrayData1/IOP/Train_AB/NNModel'

#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1TrainingB.pho'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1TrainingB.eop'
#outputFilename = '/home/jckchow/BundleAdjustment/xrayData1/IOP/xray1TrainingB_CalibratedB_moreIter_IOP.pho'
#NNModelFilename = '/home/jckchow/BundleAdjustment/xrayData1/IOP/Train_B_moreIter/NNModel'

#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1TrainingA.pho'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1TrainingA.eop'
#outputFilename = '/home/jckchow/BundleAdjustment/xrayData1/IOP/xray1TrainingA_CalibratedAB_IOP.pho'
#NNModelFilename = '/home/jckchow/BundleAdjustment/xrayData1/IOP/Train_AB/NNModel'

### X-ray fluroscopy paper 2
## for resuming
#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training150A.pho'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training150A.eop'
#NNModelFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingResults/Training150A_photoROP_robust/NNModel'
#preprocessingFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingResults/Training150A_photoROP_robust/preprocessing'
#outputFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training150A_continue.pho'

# for testing
phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TestingResults/xray1TestingA.pho'
eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TestingResults/xray1TestingA.eop'
NNModelFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingResults/Training150A_photoROP_robust_2000iter/NNModel'
preprocessingFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingResults/Training150A_photoROP_robust_2000iter/preprocessing'
outputFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TestingResults/Output/xray1TestingA_Training150A_photoROP_robust_2000iter.pho'

##########################################
### read in the residuals output from bundle adjustment
# x, y, v_x, v_y, redu_x, redu_y, vStdDev_x, vStdDev_y
##########################################
eop = np.genfromtxt(eopFilename, delimiter=' ', skip_header=0, usecols = (0,1)).astype(int)
pho = np.genfromtxt(phoFilename, delimiter=' ', skip_header=0, usecols = (0,1,2,3,4,5,6,7))

##########################################
### Apply calibraiton parameters
##########################################

sensorsUnique = np.unique(eop[:,1])

cost = 0.0
numSamples = 0.0
errors = []
outputCost = []
##########################################
### Predicting per sensor
##########################################
for iter in range(0,len(sensorsUnique)): # iterate and calibrate each sensor
    
    sensorCost = 0.0
    avgSensorCost = 0.0
    
    sensorID = sensorsUnique[iter] #currently sensor ID
    indexEOP = np.argwhere(eop[:,1] == sensorID) # eop of the current sensor

    print "Processing sensor: ", sensorID
    
    print "Loading processing info and trained ML model..."
    [min_x, min_y, max_x, max_y, desire_min, desire_max, mean_label] = joblib.load(preprocessingFilename + str(sensorID.astype(int)) + ".pkl")
    
    print "Loaded preprocessing: ", preprocessingFilename + str(sensorID.astype(int)) + ".pkl"
    
    reg = joblib.load(NNModelFilename + str(sensorID.astype(int)) + ".pkl")
    
    print "Loaded model: ", NNModelFilename + str(sensorID.astype(int)) + ".pkl"
    
    #########################################
    ### Predicting per eop
    ##########################################
    for iteration in range(0,len(indexEOP)):        
        eopID = eop[indexEOP[iteration],0]
        
        print "  Processing eop: ", eopID

        indexPho = np.argwhere(pho[:,1] == eopID)
        
        # scaling the features
        features = pho[indexPho,(2,3)]
        x_std = (features[:,0] - min_x) / (max_x - min_x)
        x_scaled = x_std * (desire_max - desire_min) + desire_min 
        y_std = (features[:,1] - min_y) / (max_y - min_y)
        y_scaled = y_std * (desire_max - desire_min) + desire_min 
        features_predict = np.concatenate((x_scaled, y_scaled)).transpose()
        
        # make the prediction
        labels_predict = reg.predict(features_predict)
        
        # add the mean back
        pho[indexPho, (6,7)] = labels_predict + mean_label

############################
### Output predicted corrections
############################

np.savetxt(outputFilename, pho, '%i %i %f %f %f %f %f %f', delimiter=' ', newline='\n')
print "Program Succcessful ^-^"
