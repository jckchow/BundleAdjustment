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
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.mlab import griddata
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata as griddataScipy


##########################################
### User parameters
##########################################

## for plotting residuals
#inputFilename  = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingResults/Training30A_before/image.jck'
#iopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1A.iop'
## for plotting models
#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training30A.pho'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training30A.eop'
#NNModelFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingResults/Training30A_before/NNModel'
#preprocessingFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingResults/Training30A_before/preprocessing'

## for plotting residuals
#inputFilename  = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingResults/Training30A_photoROP/image.jck'
#iopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1A.iop'
## for plotting models
#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training30A.pho'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training30A.eop'
#NNModelFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingResults/Training30A_photoROP/NNModel'
#preprocessingFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingResults/Training30A_photoROP/preprocessing'

#
## for plotting residuals
#inputFilename  = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingResults/Training30A_photoROP_linearSmoothing200/image.jck'
#iopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1A.iop'
## for plotting models
#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training30A.pho'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training30A.eop'
#NNModelFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingResults/Training30A_photoROP_linearSmoothing200/NNModel'
#preprocessingFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingResults/Training30A_photoROP_linearSmoothing200/preprocessing'



## for plotting residuals
#inputFilename  = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingResults/Training150A_before/image.jck'
#iopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1A.iop'
## for plotting models
#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training150A.pho'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training150A.eop'
#NNModelFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingResults/Training150A_before/NNModel'
#preprocessingFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingResults/Training150A_before/preprocessing'

# for plotting residuals
inputFilename  = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingResults/Training150A_photoROP/image.jck'
iopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1A.iop'
# for plotting models
phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training150A.pho'
eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training150A.eop'
NNModelFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingResults/Training150A_photoROP/NNModel'
preprocessingFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingResults/Training150A_photoROP/preprocessing'


## for plotting residuals
#inputFilename  = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingResults/Training150A_photoROP_linearSmoothing200/image.jck'
#iopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1A.iop'
## for plotting models
#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training150A.pho'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training150A.eop'
#NNModelFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingResults/Training150A_photoROP_linearSmoothing200/NNModel'
#preprocessingFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingResults/Training150A_photoROP_linearSmoothing200/preprocessing'
#

# do we want to apply linear or cubic smoothing to the predictions
doSmoothing = False
smoothingMethod = 'nearest' # 'linear' or 'nearest'

##########################################
### read in the residuals output from bundle adjustment
# x, y, v_x, v_y, redu_x, redu_y, vStdDev_x, vStdDev_y
##########################################
image = np.genfromtxt(inputFilename, delimiter=' ', skip_header=0, usecols = (0,1,2,3,4,5,6,7,8))
iop =  np.genfromtxt(iopFilename, delimiter=' ', skip_header=0, usecols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15))
iop = np.atleast_2d(iop)


sensorsUnique = np.unique(image[:,0])

cost = 0.0
numSamples = 0.0
errors = []
outputCost = []
##########################################
### Learn for each sensor separately
##########################################
for iter in range(0,len(sensorsUnique)): # iterate and calibrate each sensor
    
    sensorCost = 0.0
    avgSensorCost = 0.0
    
    sensorID = sensorsUnique[iter] #currently sensor ID
    indexImageAll = np.argwhere(image[:,0] == sensorID) #image residuals that are inliers + outliers
    indexImage = np.argwhere(image[:,0] == sensorID) #image residuals that are inliers
    indexIOP = np.argwhere(iop[:,0] == sensorID) # iop of the current sensor
    

sensorsUnique = np.unique(image[:,0])

cost = 0.0
numSamples = 0.0
errors = []
outputCost = []
##########################################
### Learn for each sensor separately
##########################################
for iter in range(0,len(sensorsUnique)): # iterate and calibrate each sensor
    
    sensorCost = 0.0
    avgSensorCost = 0.0
    
    sensorID = sensorsUnique[iter] #currently sensor ID
    indexImage = np.argwhere(image[:,0] == sensorID) #image residuals that are inliers
    indexIOP = np.argwhere(iop[:,0] == sensorID) # iop of the current sensor

    print "  Processing sensor: ", sensorID
    
#    ##########################################
#    ### Plotting
#    ##########################################
    resampleSizeX = iop[indexIOP,4];
    resampleSizeY = iop[indexIOP,5];

    xx, yy = np.meshgrid(np.linspace(iop[indexIOP,2], iop[indexIOP,4], num=resampleSizeX, endpoint=False),
                         np.linspace(iop[indexIOP,3], -iop[indexIOP,5], num=resampleSizeY, endpoint=False)) # endpoint should be true but numpy does something weird, probably a glitch

    
    grid_interpolatedResidualsX = griddataScipy(image[:,(1,2)], image[:,3], (xx,yy), method='nearest')
    grid_interpolatedResidualsY = griddataScipy(image[:,(1,2)], image[:,4], (xx,yy), method='nearest')
 
    plt.figure()
    plt.imshow(grid_interpolatedResidualsX)
    plt.colorbar();
    plt.title('x residuals')
    
    print 'min x-res: ', min(image[:,3])
    print 'max x-res: ', max(image[:,3])
    
    plt.figure()
    plt.imshow(grid_interpolatedResidualsX, vmin=-4, vmax=5)
    plt.colorbar();
    plt.title('x residuals')

    plt.figure()
    plt.imshow(grid_interpolatedResidualsY)
    plt.colorbar();
    plt.title('y residuals')
    
    print 'min y-res: ', min(image[:,4])
    print 'max y-res: ', max(image[:,4])
    
    plt.figure()
    plt.imshow(grid_interpolatedResidualsY, vmin=-4, vmax=3.5)
    plt.colorbar();
    plt.title('y residuals')
    

    

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
        
    

    grid_interpolatedResidualsX = griddataScipy(pho[:,(2,3)], pho[:,6], (xx,yy), method='nearest')
    grid_interpolatedResidualsY = griddataScipy(pho[:,(2,3)], pho[:,7], (xx,yy), method='nearest')
    
    plt.figure()
    plt.imshow(grid_interpolatedResidualsX)
    plt.colorbar();
    plt.title('x model')
    print 'min x: ', min(pho[:,6])
    print 'max x: ', max(pho[:,6])

    plt.figure()
    plt.imshow(grid_interpolatedResidualsX, vmin=-36.5, vmax=22)
    plt.colorbar();
    plt.title('x model')

    plt.figure()
    plt.imshow(grid_interpolatedResidualsY)
    plt.colorbar();
    plt.title('y model')
    print 'min y: ', min(pho[:,7])
    print 'max y: ', max(pho[:,7])

    plt.figure()
    plt.imshow(grid_interpolatedResidualsY, vmin=-35, vmax=22.5)
    plt.colorbar();
    plt.title('y model')
