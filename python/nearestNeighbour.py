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

import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


import numpy as np
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
#from matplotlib.mlab import griddata
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata as griddataScipy
from sklearn import metrics

##################################
### User defined parameters
##################################   
#inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'
#phoFilename = '/home/jckchow/BundleAdjustment/Data/Dcs28mmTemp.pho'
#iopFilename = '/home/jckchow/BundleAdjustment/Data/Dcs28mm.iop'
#eopFilename = '/home/jckchow/BundleAdjustment/Data/Dcs28mm.eop'

#inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'
#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1TestingTemp.pho'
#iopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1.iop'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1Testing.eop'

#inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'
#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1TestingTemp.pho'
#iopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1A.iop'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1TestingA.eop'

#inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'
#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1TestingTemp.pho'
#iopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1B.iop'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1TestingB.eop'

##########################
#### Paper 2: Training 150, Testing 150
###########################
########
### SensorA
########

#inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'
#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho'
#iopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1A.iop'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training30A.eop'

#inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'
#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho'
#iopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1A.iop'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training60A.eop'

#inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'
#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho'
#iopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1A.iop'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training90A.eop'

#inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'
#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho'
#iopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1A.iop'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training120A.eop'

#inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'
#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho'
#iopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1A.iop'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training150A.eop'

########
### Sensor B
########

#inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'
#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho'
#iopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1B.iop'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training60B.eop'

#inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'
#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho'
#iopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1B.iop'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training90B.eop'

#inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'
#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho'
#iopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1B.iop'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training120B.eop'

#inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'
#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho'
#iopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1B.iop'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training150B.eop'

########
### Sensors A + B together
########
#inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'
#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho'
#iopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1.iop'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training30.eop'

#inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'
#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho'
#iopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1.iop'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training60.eop'

#inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'
#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho'
#iopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1.iop'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training90.eop'

#inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'
#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho'
#iopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1.iop'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training120.eop'

#inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'
#phoFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho'
#iopFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1.iop'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training150.eop'

#########################
### Paper 1 TC 1: Omnidirectional camera calibration
##########################
#inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'
#phoFilename = '/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/nikonLessTemp.pho'
#iopFilename = '/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/nikon.iop'
#eopFilename = '/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/nikon.eop'

# nikon D600 DSLR
#inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'
#phoFilename = '/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/TrainingTesting/nikonTrainingTemp.pho'
#iopFilename = '/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/nikon.iop'
#eopFilename = '/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/TrainingTesting/nikonTraining.eop'

## GoPro Hero 3 Silver Edition
#inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'
#phoFilename = '/home/jckchow/BundleAdjustment/omnidirectionalCamera/gopro/TrainingTesting/goproTrainingTemp.pho'
#iopFilename = '/home/jckchow/BundleAdjustment/omnidirectionalCamera/gopro/gopro.iop'
#eopFilename = '/home/jckchow/BundleAdjustment/omnidirectionalCamera/gopro/TrainingTesting/goproTraining.eop'


#########################
### Paper 2 Omnidirectional camera calibration
##########################
## nikon D600 DSLR
#inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'
#phoFilename = '/home/jckchow/BundleAdjustment/build/temp.pho'
#iopFilename = '/media/sf_UbuntuVirtualShared/bundleAdjustment/omnidirectionalCamera/nikon_2020_03_23/nikon_updated.iop'
#eopFilename = '/media/sf_UbuntuVirtualShared/bundleAdjustment/omnidirectionalCamera/nikon_2020_03_23/nikon_updated.eop'

## Go Pro 3 Silver Edition
#inputFilename   = '/home/jckchow/BundleAdjustment/build/image.jck'
#phoFilename     = '/home/jckchow/BundleAdjustment/build/temp.pho'
##iopFilename = '/media/sf_UbuntuVirtualShared/bundleAdjustment/omnidirectionalCamera/gopro_2020_04_01/gopro_stereographic.iop'
#iopFilename     = '/media/sf_UbuntuVirtualShared/bundleAdjustment/omnidirectionalCamera/gopro_2020_04_01/gopro.iop'
#eopFilename     = '/media/sf_UbuntuVirtualShared/bundleAdjustment/omnidirectionalCamera/gopro_2020_04_01/gopro.eop'

## Go Pro Training
inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'
phoFilename    = '/home/jckchow/BundleAdjustment/build/temp.pho'
#iopFilename   = '/media/sf_UbuntuVirtualShared/bundleAdjustment/omnidirectionalCamera/gopro_2020_04_01/gopro_stereographic.iop'
iopFilename   = '/home/jckchow/BundleAdjustment/build/temp.iop'
#iopFilename    = '/media/sf_UbuntuVirtualShared/bundleAdjustment/omnidirectionalCamera/gopro_2020_04_01/TrainingTesting/goproTraining.iop'
eopFilename    = '/home/jckchow/BundleAdjustment/build/temp.eop'



# Maximum number of neighbours to test (+1 of what you actually want)
#minK = 3
#maxK = 1001
#minK = 3; maxK = 4
minK = 1; maxK = 2

# do we want to plot things (True or False)
doPlot = False

# do we want to apply linear or cubic smoothing to the predictions
doSmoothing = False
smoothingMethod = 'linear' # 'linear' or 'nearest' or 'cubic'

# do ensemble bagging
doBagging = True
numEstimators = 25
minNumSamples = 0.1
maxNumSamples = 1.0

print ("-----------K-Nearest Neighbour Modelling-----------")

##########################################
### read in the residuals output from bundle adjustment
# x, y, v_x, v_y, redu_x, redu_y, vStdDev_x, vStdDev_y
##########################################
image = np.genfromtxt(inputFilename, delimiter=' ', skip_header=0, usecols = (2,3,4,5,6,7,8,9,10))
iop =  np.genfromtxt(iopFilename, delimiter=' ', skip_header=0, usecols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15))
iop = np.atleast_2d(iop)
eop = np.genfromtxt(eopFilename, delimiter=' ', skip_header=0, usecols = (0,1)).astype(int)
pho = np.genfromtxt(phoFilename, delimiter=' ', skip_header=0, usecols = (0,1,2,3,4,5,6,7))

##########################################
### filtering out the outliers
##########################################

w = np.divide(image[:,(3,4)], image[:,(7,8)]) #normalized residuals

# 95% is 1.96
outlierThreshold = np.inf; #outlierThreshold = 3000.0
#outlierThreshold = 1.96
#outlierThreshold = 3.291 #99.9%
outlierIndex = np.argwhere(np.fabs(w) > outlierThreshold)

print ("  Outlier removal threshold: ", outlierThreshold, " x sigma")

inliers = np.delete(image, outlierIndex[:,0], axis=0)
prevCorr = np.delete(pho, outlierIndex[:,0], axis=0)

sensorsUnique = np.unique(image[:,0])

cost = 0.0
numSamples = 0.0
errors = []
outputCost = []

##########################################
### Try to load preprocessing and ML model if it exists from previous iteration to make a copy for when the iterations end before the max iterations then we should use the previous model instead
##########################################
try:
    for iter in range(0,len(sensorsUnique)):
        sensorID = sensorsUnique[iter] #currently sensor ID
        # load the preprocessing info
        [min_x, min_y, max_x, max_y, desire_min, desire_max, mean_label] = joblib.load('/home/jckchow/BundleAdjustment/build/NNPreprocessing'+str(sensorID.astype(int))+'.pkl')
        # load the learned NN model
        reg = joblib.load('/home/jckchow/BundleAdjustment/build/NNModel'+str(sensorID.astype(int))+'.pkl')
        # save copy of previous preprocessing
        joblib.dump([min_x, min_y, max_x, max_y, desire_min, desire_max, mean_label], '/home/jckchow/BundleAdjustment/build/NNPreprocessing'+str(sensorID.astype(int))+'Temp.pkl')
        # save the previously learned NN model
        joblib.dump(reg, '/home/jckchow/BundleAdjustment/build/NNModel'+str(sensorID.astype(int))+'Temp.pkl')
        print('  Found previous ML preprocessing and model, making a copy and saving it as temp')
except:
    print('No previous ML preprocessing and model found')
    
##########################################
### Learn for each sensor separately
##########################################
for iter in range(0,len(sensorsUnique)): # iterate and calibrate each sensor
    
    sensorCost = 0.0
    avgSensorCost = 0.0
    
    sensorID = sensorsUnique[iter] #currently sensor ID
    indexImageAll = np.argwhere(image[:,0] == sensorID) #image residuals that are inliers + outliers
    indexImage = np.argwhere(inliers[:,0] == sensorID) #image residuals that are inliers
    indexIOP = np.argwhere(iop[:,0] == sensorID) # iop of the current sensor
    indexEOP = np.argwhere(eop[:,1] == sensorID) # eop of the current sensor

    print ("  Processing sensor: ", sensorID)
    
    ##########################################
    ### Training
    ##########################################
    features_train = inliers[indexImage,(1,2)]
    labels_train = inliers[indexImage,(3,4)] + prevCorr[indexImage,(6,7)] # this is the iterative process
    features_train_original = features_train
    # apply some scaling to the features to be between -1 and 1 in x and y
    min_x = iop[indexIOP, 2]
    min_y = -iop[indexIOP, 5]
    max_x = iop[indexIOP, 4]
    max_y = iop[indexIOP, 3]
    desire_min = -1
    desire_max = 1
    x_std = (features_train[:,0] - min_x) / (max_x - min_x)
    x_scaled = x_std * (desire_max - desire_min) + desire_min 
    y_std = (features_train[:,1] - min_y) / (max_y - min_y)
    y_scaled = y_std * (desire_max - desire_min) + desire_min 
    features_train = np.concatenate((x_scaled, y_scaled)).transpose()
    
    # apply scale to remove the mean of labels, so we have zero mean
    mean_label = np.mean(labels_train,axis=0);
    labels_train -= mean_label
    
#    reg = neighbors.KNeighborsRegressor(n_neighbors=15, weights='uniform')
#    reg.fit(features_train, labels_train)
    # score = clf.score(features_test, labels_test)    
    
    # smooth and rasterize the data before doing kNN
    if (doSmoothing):
        resampleSizeX = 200;
        resampleSizeY = 200;
        
        print('  Using Smoothing Method: ' + smoothingMethod)
        print('  Resample residuals to image with dimensions: ' + str(resampleSizeX) + ' x ' + str(resampleSizeY) + ' pixels')
#        xx, yy = np.meshgrid(np.arange(iop[indexIOP,2], iop[indexIOP,4], 4),
#                             np.arange(iop[indexIOP,3], -iop[indexIOP,5], -4))
        xx, yy = np.meshgrid(np.linspace(iop[indexIOP,2], iop[indexIOP,4], num=resampleSizeX, endpoint=False),
                             np.linspace(iop[indexIOP,3], -iop[indexIOP,5], num=resampleSizeY, endpoint=False)) # endpoint should be true but numpy does something weird, probably a glitch
    
        # scale it first
        X = np.hstack((np.reshape(xx,(-1,1)), np.reshape(yy,(-1,1))))
        x_std = (X[:,0] - min_x) / (max_x - min_x)
        x_scaled = x_std * (desire_max - desire_min) + desire_min 
        y_std = (X[:,1] - min_y) / (max_y - min_y)
        y_scaled = y_std * (desire_max - desire_min) + desire_min 
        X = np.concatenate((x_scaled, y_scaled)).transpose()
        
        xx_std = (xx - min_x) / (max_x - min_x)
        xx_scaled = xx_std * (desire_max - desire_min) + desire_min 
        
        yy_std = (yy - min_y) / (max_y - min_y)
        yy_scaled = yy_std * (desire_max - desire_min) + desire_min 
        
        grid_interpolatedResidualsX = griddataScipy(features_train, labels_train[:,0], (xx_scaled,yy_scaled), method=smoothingMethod)
        grid_interpolatedResidualsY = griddataScipy(features_train, labels_train[:,1], (xx_scaled,yy_scaled), method=smoothingMethod)
        
        if (doPlot):
            plt.figure()
            plt.imshow(grid_interpolatedResidualsX, vmin=-abs(labels_train[:,0]).max(), vmax=abs(labels_train[:,0]).max())
            plt.colorbar();
            plt.title('Interpolated x residuals')
            
            plt.figure()
            plt.imshow(grid_interpolatedResidualsY)
            plt.colorbar();
            plt.title('Interpolated y residuals')
        
        # stack the labels into the right format and remove nans from the training inputs and outputs
        interpolatedResiduals = np.column_stack((grid_interpolatedResidualsX.flatten(), grid_interpolatedResidualsY.flatten()))
#        removeList = np.isnan(interpolatedResiduals[:,0])
#        interpolatedResiduals[removeList,:] = [];
        removeList = ~np.isnan(interpolatedResiduals).any(axis=1);
        interpolatedTraining = X[removeList]
        interpolatedResiduals = interpolatedResiduals[removeList]

        # Tune kNN using CV
        t0 = time()
        param_grid = [ {'n_neighbors' : range(minK,maxK,1)} ] # test only up to 50 neighbours
#        param_grid = [ {'n_neighbors' : range(3,51,1)} ] # test only up to 50 neighbours
        regCV = GridSearchCV(neighbors.KNeighborsRegressor(weights='uniform'), param_grid, cv=10, verbose = 0)
        regCV.fit(interpolatedTraining, interpolatedResiduals)
#        regCV.fit(np.row_stack((features_train, interpolatedTraining)), np.row_stack((labels_train, interpolatedResiduals)))
        print ("    Best in sample score: ", regCV.best_score_)
        print ("    CV value for K ( between",minK," and", maxK-1,"): ", regCV.best_estimator_.n_neighbors)
        print ("    Training NN-Regressor + CV time:", round(time()-t0, 3), "s")
        
        # train with the best K parameter
#        t0 = time()   
#        reg = neighbors.KNeighborsRegressor(n_neighbors=regCV.best_estimator_.n_neighbors, weights='uniform', n_jobs=1)
#        reg.fit(interpolatedTraining, interpolatedResiduals)
        reg = regCV.best_estimator_;
        bestK = regCV.best_estimator_.n_neighbors;
        bestMaxSamples = 1.0
        print ("    Training Final NN-Regressor:", round(time()-t0, 3), "s")
        
#        # Tune rNN using CV
#        minSpacing = 0.5*( (xx_scaled[0,1] - xx_scaled[0,0]) + (yy_scaled[0,0] - yy_scaled[1,0]) )
#        minSpacingBuffered = minSpacing + 0.001 * minSpacing; # give it a buffer to ensure we capture it in the radius search
##        minSpacing /= 2.0;
#        t0 = time()
#        param_grid = [ {'radius' : np.arange(1.0*minSpacingBuffered,25.0*minSpacingBuffered, (minSpacing/2.0))} ] # test only up to 50 neighbours
##        regCV = GridSearchCV(neighbors.RadiusNeighborsRegressor(weights='uniform'), param_grid, cv=10, verbose = 0)
#        regCV = GridSearchCV(neighbors.RadiusNeighborsRegressor(), param_grid, cv=10)
#        regCV.fit(interpolatedTraining, interpolatedResiduals)
##        regCV.fit(np.row_stack((features_train, interpolatedTraining)), np.row_stack((labels_train, interpolatedResiduals)))
#        print "    Best in sample score: ", regCV.best_score_
#        print "    CV value for K (between 3 and 50): ", regCV.best_estimator_.n_neighbors
#        print "    Training NN-Regressor + CV time:", round(time()-t0, 3), "s"
#        
#        # train with the best radius parameter
#        t0 = time()   
#        reg = neighbors.RadiusNeighborsRegressor(radius=regCV.best_estimator_.radius, weights='uniform', n_jobs=1)
#        reg.fit(interpolatedTraining, interpolatedResiduals)
#        print "    Training Final NN-Regressor:", round(time()-t0, 3), "s"
#        
#        reg = neighbors.RadiusNeighborsRegressor(radius=0.02, weights='uniform', n_jobs=1)
#        reg.fit(interpolatedTraining, interpolatedResiduals)

    
    else:
        print('  Not doing Smoothing')
        
        if (doBagging):
            print('     Doing Ensemble: Bagging')
            t0 = time()
#    #        param_grid = [ {'n_neighbors' : range(3,np.min((51,len(features_train[:,0])/10)))} ] # test only up to 50 neighbours
##            param_grid = [ {'n_neighbors' : range(3,maxK,1)} ] # test only up to 50 neighbours
#            param_grid = [ {'n_neighbors' : range(minK,maxK,1)} ] # test only up to 50 neighbours
##            regCV = GridSearchCV(neighbors.KNeighborsRegressor(weights='uniform'), param_grid, cv=10, verbose = 0, n_jobs=1, scoring='neg_mean_squared_error')
#            regCV = GridSearchCV(neighbors.KNeighborsRegressor(weights='uniform'), param_grid, cv=10, verbose = 0,n_jobs=1)
#            regCV.fit(features_train, labels_train)
#            bestK = regCV.best_estimator_.n_neighbors
#            print ("        Best in sample score: ", regCV.best_score_)
#            print ("        CV value for K ( between", minK, " and", maxK-1,"): ", regCV.best_estimator_.n_neighbors)
                        
            param_grid = [ {'max_samples' : np.arange(minNumSamples,maxNumSamples,0.1)} ] # test only up to 50 neighbours
#            reg = BaggingRegressor(base_estimator=neighbors.KNeighborsRegressor(weights='uniform', n_neighbors=1), n_estimators=10, max_samples=1.0, oob_score=False, random_state=0)
#            reg.fit(features_train, labels_train)
            regCV = GridSearchCV(BaggingRegressor(base_estimator=neighbors.KNeighborsRegressor(weights='uniform', n_neighbors=1), n_estimators=numEstimators, oob_score=False, random_state=0), param_grid, cv=10, verbose = 0, n_jobs=1)
            regCV.fit(features_train, labels_train)
            print ("        Best in sample score: ", regCV.best_score_)
            print ("        CV value for max_samples: ( between", minNumSamples, " and", maxNumSamples-0.1,"): ", regCV.best_estimator_.max_samples)
            print ("        # Estimators: ", numEstimators)
            
            reg = regCV.best_estimator_;
            bestK = 1
            bestMaxSamples = regCV.best_estimator_.max_samples;
            print ("        Training NN-Regressor + Bagging time:", round(time()-t0, 3), "s")
    
            # train with the best K parameter
#            t0 = time()   
    #        reg = neighbors.KNeighborsRegressor(n_neighbors=1, weights='uniform', n_jobs=1)
    #        reg = neighbors.KNeighborsRegressor(n_neighbors=regCV.best_estimator_.n_neighbors, weights='uniform', n_jobs=1)
    #        reg.fit(features_train, labels_train)
#            reg = regCV.best_estimator_;
#            print ("    Training Final NN-Regressor:", round(time()-t0, 3), "s")
        else:      
            # Tune kNN using CV
            t0 = time()
    #        param_grid = [ {'n_neighbors' : range(3,np.min((51,len(features_train[:,0])/10)))} ] # test only up to 50 neighbours
    #        param_grid = [ {'n_neighbors' : range(3,maxK,1)} ] # test only up to 50 neighbours
            param_grid = [ {'n_neighbors' : range(minK,maxK,1)} ] # test only up to 50 neighbours
    #        regCV = GridSearchCV(neighbors.KNeighborsRegressor(weights='uniform'), param_grid, cv=10, verbose = 0, n_jobs=1, scoring='neg_mean_squared_error')
            regCV = GridSearchCV(neighbors.KNeighborsRegressor(weights='uniform'), param_grid, cv=10, verbose = 0, n_jobs=1)
            regCV.fit(features_train, labels_train)
            print ("    Best in sample score: ", regCV.best_score_)
            print ("    CV value for K ( between", minK, " and", maxK-1,"): ", regCV.best_estimator_.n_neighbors)
            print ("    Training NN-Regressor + CV time:", round(time()-t0, 3), "s")
    
            # train with the best K parameter
#            t0 = time()   
    #        reg = neighbors.KNeighborsRegressor(n_neighbors=1, weights='uniform', n_jobs=1)
    #        reg = neighbors.KNeighborsRegressor(n_neighbors=regCV.best_estimator_.n_neighbors, weights='uniform', n_jobs=1)
    #        reg.fit(features_train, labels_train)
            reg = regCV.best_estimator_;
            bestK = regCV.best_estimator_.n_neighbors;
            bestMaxSamples = 1.0
            print ("    Training Final NN-Regressor:", round(time()-t0, 3), "s")

    print ("    Done Training")
    # save the preprocessing info
    joblib.dump([min_x, min_y, max_x, max_y, desire_min, desire_max, mean_label], 'NNPreprocessing'+str(sensorID.astype(int))+'.pkl')     
    # save the learned NN model
    joblib.dump(reg, 'NNModel'+str(sensorID.astype(int))+'.pkl')     
    ##########################################
    ### Prediction
    ########################################## 
    print ("    Start Prediction...")
    EOP2IOP = np.unique(eop[indexEOP,0]) # should not be needed since it should be unique already
    
    stationsWithThisIOP = []
    for n in range(0,len(EOP2IOP)):
        temp = np.argwhere(pho[:,1] == EOP2IOP[n])
        stationsWithThisIOP.append(temp.flatten().tolist())
    stationsWithThisIOP = np.concatenate(np.asarray(stationsWithThisIOP))
    
    t0 = time()
    stationsWithThisIOP = np.reshape(stationsWithThisIOP, (-1,1))
    # apply the scaling
    pho_scaled = pho[stationsWithThisIOP,(2,3)]
    x_std = (pho_scaled[:,0] - min_x) / (max_x - min_x)
    x_scaled = x_std * (desire_max - desire_min) + desire_min 
    y_std = (pho_scaled[:,1] - min_y) / (max_y - min_y)
    y_scaled = y_std * (desire_max - desire_min) + desire_min 
    pho_scaled = np.concatenate((x_scaled, y_scaled)).transpose()   

    correction = reg.predict(pho_scaled) + mean_label # add the mean back        
    
    pho[stationsWithThisIOP,(6,7)] = correction
    print ("    Done predicting:", round(time()-t0, 3), "s")
    
    ##########################################
    ### Calculate Error from inliers only
    ########################################## 
    t0 = time()
 
    ### x component
    v = (np.reshape(labels_train[:,0],(-1,1)) - np.reshape(reg.predict(features_train)[:,0],(-1,1))) / inliers[indexImage,7]
    weightedScore = np.matmul(v.transpose(), v)[0,0]
    sensorCost += weightedScore
    avgSensorCost += weightedScore/float(len(indexImage))
    print ("    Weighted x score: ", weightedScore)
    print ("    Average weighted x score: ", weightedScore/len(indexImage))

    v = (np.reshape(labels_train[:,1],(-1,1)) - np.reshape(reg.predict(features_train)[:,1],(-1,1))) / inliers[indexImage,8]   
    weightedScore = np.matmul(v.transpose(), v)[0,0]
    sensorCost += weightedScore
    avgSensorCost += weightedScore/float(len(indexImage))
    print ("    Weighted y score: ", weightedScore)
    print ("    Average weighted y score: ", weightedScore/len(indexImage))
    print ("      Weighted total score: ", sensorCost)    
    print ("      Average weighted total score: ", sensorCost/float(len(indexImage)))    
    print ("      Number of samples: ", len(indexImage), " inliers out of a total of ", len(indexImageAll), " (", round(100.0*(float(len(indexImage))/float(len(indexImageAll))),1), "%)")
    print ("    Avg RMSE in x: ", np.sqrt( metrics.mean_squared_error(labels_train[:,0], reg.predict(features_train)[:,0]) ))
    print ("    Avg RMSE in y: ", np.sqrt( metrics.mean_squared_error(labels_train[:,1], reg.predict(features_train)[:,1]) ))
    print ("    Avg Overall RMSE: ", np.sqrt( metrics.mean_squared_error(labels_train, reg.predict(features_train)) ))
    print ("    Done calculating error:", round(time()-t0, 3), "s") 
    
    # log total cost and total number of samples for output
    cost += sensorCost
    numSamples += 2.0*len(indexImage)
    
    errors.append([sensorID, sensorCost, 2.0*len(indexImage), bestK])
#    ##########################################
#    ### Plotting
#    ##########################################

    if (doPlot):
        xx, yy = np.meshgrid(np.arange(iop[indexIOP,2], iop[indexIOP,4], 1),
                             np.arange(iop[indexIOP,3], -iop[indexIOP,5], -1))
    
        # scale it first
        X = np.hstack((np.reshape(xx,(-1,1)), np.reshape(yy,(-1,1))))
        x_std = (X[:,0] - min_x) / (max_x - min_x)
        x_scaled = x_std * (desire_max - desire_min) + desire_min 
        y_std = (X[:,1] - min_y) / (max_y - min_y)
        y_scaled = y_std * (desire_max - desire_min) + desire_min 
        X = np.concatenate((x_scaled, y_scaled)).transpose()
        
        pred = reg.predict(X) + mean_label
        
        xp = iop[indexIOP,6]
        yp = iop[indexIOP,7]
            
        # Plot the training points    
        plt.figure()
        plt.scatter(features_train[:, 0], features_train[:, 1], color = 'darkorange')
        plt.title('Image measurements for sensor ' + str(sensorID))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    #    plt.xlim([iop[indexIOP,2], iop[indexIOP,4]]) # set it to the image format
    #    plt.ylim([iop[indexIOP,3], -iop[indexIOP,5]])
        
        # define grid.
        xi = np.arange(iop[indexIOP,2], iop[indexIOP,4], 1)
        yi = np.arange(iop[indexIOP,3], -iop[indexIOP,5], -1)
        # scale it 
        x_std = (xi - min_x) / (max_x - min_x)
        x_scaled = x_std * (desire_max - desire_min) + desire_min 
        y_std = (yi - min_y) / (max_y - min_y)
        y_scaled = y_std * (desire_max - desire_min) + desire_min 
        xi = x_scaled.flatten()
        yi = y_scaled.flatten()
       
#        ####################### Converting to scipy
#        ###############grid_x, grid_y = np.mgrid(xi, yi)
    
        # define grid.
        resampleSizeX = iop[indexIOP,4];
        resampleSizeY = iop[indexIOP,5];

        xx, yy = np.meshgrid(np.linspace(iop[indexIOP,2], iop[indexIOP,4], num=resampleSizeX, endpoint=False),
                             np.linspace(iop[indexIOP,3], -iop[indexIOP,5], num=resampleSizeY, endpoint=False)) # endpoint should be true but numpy does something weird, probably a glitch
    
        # scale it first
        X = np.hstack((np.reshape(xx,(-1,1)), np.reshape(yy,(-1,1))))
        x_std = (X[:,0] - min_x) / (max_x - min_x)
        x_scaled = x_std * (desire_max - desire_min) + desire_min 
        y_std = (X[:,1] - min_y) / (max_y - min_y)
        y_scaled = y_std * (desire_max - desire_min) + desire_min 
        X = np.concatenate((x_scaled, y_scaled)).transpose()
        
        xx_std = (xx - min_x) / (max_x - min_x)
        xx_scaled = xx_std * (desire_max - desire_min) + desire_min 
        
        yy_std = (yy - min_y) / (max_y - min_y)
        yy_scaled = yy_std * (desire_max - desire_min) + desire_min 
    
        plt.figure()
        # grid the data.
        zi = griddataScipy(features_train, labels_train[:, 0] + mean_label[0], (xx_scaled,yy_scaled), method='linear')

        # contour the gridded data, plotting dots at the nonuniform data points.
        CS = plt.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
        CS = plt.contourf(xi, yi, zi, 15,
                          vmax=abs(zi[~np.isnan(zi)]).max(), vmin=-abs(zi[~np.isnan(zi)]).max())
        plt.colorbar()  # draw colorbar
        plt.title('x residuals: Sensor ' + str(sensorID))
        plt.show()
                    
        plt.figure()
        # grid the data.
        #zi = griddata(features_train[:, 0], features_train[:, 1], labels_train[:, 0] + mean_label[0], xi, yi, interp='linear')
        zi = griddataScipy(features_train, labels_train[:, 0] + mean_label[0], (xx_scaled,yy_scaled), method='linear')
 
       # contour the gridded data, plotting dots at the nonuniform data points.
        CS = plt.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
        CS = plt.contourf(xi, yi, zi, 15,
                          vmax=abs(zi[~np.isnan(zi)]).max(), vmin=-abs(zi[~np.isnan(zi)]).max())
        plt.colorbar()  # draw colorbar
        # plot data points.
        plt.scatter(features_train[:, 0], features_train[:, 1], marker='o', color='red', s=5, zorder=10)
        plt.title('x residuals: Sensor ' + str(sensorID))
        plt.show()
    
        # predicting at every grid cell location for visualization as an image
        zz = np.reshape(pred[:,0], np.shape(xx))
        plt.figure()
        # contour the gridded data, plotting dots at the nonuniform data points.
        CS = plt.contour(xx, yy, zz, 15, linewidths=0.5, colors='k')
        CS = plt.contourf(xx, yy, zz, 15,
                          vmax=abs(zz).max(), vmin=-abs(zz).max())
        plt.colorbar()  # draw colorbar
        # plot data points.
    #    plt.scatter(features_train[:, 0], features_train[:, 1], marker='o', color='red', s=5, zorder=10)
        plt.title('x model: Sensor ' + str(sensorID))
        plt.show()
        
        plt.figure()
        # contour the gridded data, plotting dots at the nonuniform data points.
        CS = plt.contour(xx, yy, zz, 15, linewidths=0.5, colors='k')
        CS = plt.contourf(xx, yy, zz, 15,
                          vmax=abs(zz).max(), vmin=-abs(zz).max())
        plt.colorbar()  # draw colorbar
        # plot data points.
        plt.scatter(features_train_original[:, 0], features_train_original[:, 1], marker='o', color='red', s=5, zorder=10)
        plt.title('x model: Sensor ' + str(sensorID))
        plt.show()
    
        plt.figure()
        # contour the gridded data, plotting dots at the nonuniform data points.
        CS = plt.contourf(xx, yy, zz, 100)
        plt.colorbar()  # draw colorbar
        plt.title('x model: Sensor ' + str(sensorID))
        plt.show()

       
        plt.figure()
        # contour the gridded data, plotting dots at the nonuniform data points.
        CS = plt.imshow(zz, vmin=-abs(labels_train[:,0]+mean_label[0]).max(), vmax=abs(labels_train[:,0]+mean_label[0]).max())
        plt.colorbar()  # draw colorbar
        plt.title('x model: Sensor ' + str(sensorID))
        plt.show()
        
#        resampleSizeX = iop[indexIOP,4];
#        resampleSizeY = iop[indexIOP,5];
#
#        xx, yy = np.meshgrid(np.linspace(iop[indexIOP,2], iop[indexIOP,4], num=resampleSizeX, endpoint=False),
#                             np.linspace(iop[indexIOP,3], -iop[indexIOP,5], num=resampleSizeY, endpoint=False)) # endpoint should be true but numpy does something weird, probably a glitch
#    
#        # scale it first
#        X = np.hstack((np.reshape(xx,(-1,1)), np.reshape(yy,(-1,1))))
#        x_std = (X[:,0] - min_x) / (max_x - min_x)
#        x_scaled = x_std * (desire_max - desire_min) + desire_min 
#        y_std = (X[:,1] - min_y) / (max_y - min_y)
#        y_scaled = y_std * (desire_max - desire_min) + desire_min 
#        X = np.concatenate((x_scaled, y_scaled)).transpose()
#        
#        xx_std = (xx - min_x) / (max_x - min_x)
#        xx_scaled = xx_std * (desire_max - desire_min) + desire_min 
#        
#        yy_std = (yy - min_y) / (max_y - min_y)
#        yy_scaled = yy_std * (desire_max - desire_min) + desire_min 
        
        grid_interpolatedResidualsX = griddataScipy(features_train, labels_train[:,0]+mean_label[0], (xx_scaled,yy_scaled), method='nearest')
        grid_interpolatedResidualsY = griddataScipy(features_train, labels_train[:,1]+mean_label[1], (xx_scaled,yy_scaled), method='nearest')
        
        plt.figure()
        plt.imshow(grid_interpolatedResidualsX, vmin=-abs(labels_train[:,0]+mean_label[0]).max(), vmax=abs(labels_train[:,0]+mean_label[0]).max())
        plt.colorbar();
        plt.title('Interpolated x residuals')
        plt.show()
        

        
    #    fig = plt.figure()
    #    ax = fig.gca(projection='3d')
    #    X = xx
    #    Y = yy
    #    Z = zi
    ##    X, Y, Z = axes3d.get_test_data(0.05)
    #    ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
    ##    cset = ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
    #    cset = ax.contourf(X, Y, Z, zdir='x', offset=0, cmap=cm.coolwarm)
    #    cset = ax.contourf(X, Y, Z, zdir='y', offset=0, cmap=cm.coolwarm)
    
        plt.figure()
        # grid the data.
#        zi = griddata(features_train[:, 0], features_train[:, 1], labels_train[:, 1] + mean_label[1], xi, yi, interp='linear')
        zi = griddataScipy(features_train, labels_train[:, 1] + mean_label[1], (xx_scaled,yy_scaled), method='linear')
        # contour the gridded data, plotting dots at the nonuniform data points.
        CS = plt.contourf(xi, yi, zi, 15, linewidths=0.5, colors='k')
        CS = plt.contourf(xi, yi, zi, 15,
                          vmax=abs(zi[~np.isnan(zi)]).max(), vmin=-abs(zi[~np.isnan(zi)]).max())
        plt.colorbar()  # draw colorbar
        # plot data points.
        plt.scatter(features_train[:, 0], features_train[:, 1], marker='o', color='red', s=5, zorder=10)
        plt.title('y residuals: Sensor ' + str(sensorID))
        plt.show()
    
        plt.figure()
        # grid the data.
#        zi = griddata(features_train[:, 0], features_train[:, 1], labels_train[:, 1] + mean_label[1], xi, yi, interp='linear')
        zi = griddataScipy(features_train, labels_train[:, 1] + mean_label[1], (xx_scaled,yy_scaled), method='linear')
        # contour the gridded data, plotting dots at the nonuniform data points.
        CS = plt.contourf(xi, yi, zi, 15, linewidths=0.5, colors='k')
        CS = plt.contourf(xi, yi, zi, 15,
                          vmax=abs(zi[~np.isnan(zi)]).max(), vmin=-abs(zi[~np.isnan(zi)]).max())
        plt.colorbar()  # draw colorbar
        # plot data points.
    #    plt.scatter(features_train[:, 0], features_train[:, 1], marker='o', color='red', s=5, zorder=10)
        plt.title('y residuals: Sensor ' + str(sensorID))
        plt.show()
     
        zz = np.reshape(pred[:,1], np.shape(xx))
        plt.figure()
        # contour the gridded data, plotting dots at the nonuniform data points.
        CS = plt.contour(xx, yy, zz, 15, linewidths=0.5, colors='k')
        CS = plt.contourf(xx, yy, zz, 15,
                          vmax=abs(zz).max(), vmin=-abs(zz).max())
        plt.colorbar()  # draw colorbar
        # plot data points.
    #    plt.scatter(features_train[:, 0], features_train[:, 1], marker='o', color='red', s=5, zorder=10)
        plt.title('y model: Sensor ' + str(sensorID))
        plt.show()
        
        plt.figure()
        # contour the gridded data, plotting dots at the nonuniform data points.
        CS = plt.contour(xx, yy, zz, 15, linewidths=0.5, colors='k')
        CS = plt.contourf(xx, yy, zz, 15,
                          vmax=abs(zz).max(), vmin=-abs(zz).max())
        plt.colorbar()  # draw colorbar
        # plot data points.
        plt.scatter(features_train_original[:, 0], features_train_original[:, 1], marker='o', color='red', s=5, zorder=10)
        plt.title('y model: Sensor ' + str(sensorID))
        plt.show()
        
                
        plt.figure()
        # contour the gridded data, plotting dots at the nonuniform data points.
        CS = plt.contourf(xx, yy, zz, 100)
        plt.colorbar()  # draw colorbar
        # plot data points.
        plt.title('y model: Sensor ' + str(sensorID))
        plt.show()
        
        
        
        
        plt.figure()
        # contour the gridded data, plotting dots at the nonuniform data points.
        CS = plt.imshow(zz, vmin=-abs(labels_train[:,1]+mean_label[1]).max(), vmax=abs(labels_train[:,1]+mean_label[1]).max())
        plt.colorbar()  # draw colorbar
        plt.title('y model: Sensor ' + str(sensorID))
        plt.show()
        
        resampleSizeX = iop[indexIOP,4];
        resampleSizeY = iop[indexIOP,5];

        xx, yy = np.meshgrid(np.linspace(iop[indexIOP,2], iop[indexIOP,4], num=resampleSizeX, endpoint=False),
                             np.linspace(iop[indexIOP,3], -iop[indexIOP,5], num=resampleSizeY, endpoint=False)) # endpoint should be true but numpy does something weird, probably a glitch
    
        # scale it first
        X = np.hstack((np.reshape(xx,(-1,1)), np.reshape(yy,(-1,1))))
        x_std = (X[:,0] - min_x) / (max_x - min_x)
        x_scaled = x_std * (desire_max - desire_min) + desire_min 
        y_std = (X[:,1] - min_y) / (max_y - min_y)
        y_scaled = y_std * (desire_max - desire_min) + desire_min 
        X = np.concatenate((x_scaled, y_scaled)).transpose()
        
        xx_std = (xx - min_x) / (max_x - min_x)
        xx_scaled = xx_std * (desire_max - desire_min) + desire_min 
        
        yy_std = (yy - min_y) / (max_y - min_y)
        yy_scaled = yy_std * (desire_max - desire_min) + desire_min 
        
        grid_interpolatedResidualsY = griddataScipy(features_train, labels_train[:,1]+mean_label[1], (xx_scaled,yy_scaled), method='nearest')
        
        plt.figure()
        plt.imshow(grid_interpolatedResidualsY, vmin=-abs(labels_train[:,1]+mean_label[1]).max(), vmax=abs(labels_train[:,1]+mean_label[1]).max())
        plt.colorbar();
        plt.title('Interpolated y residuals')
        plt.show()            
            
        ### 1D Plots
        plt.figure()
        plt.scatter( features_train[:, 0], labels_train[:, 0] + mean_label[0])
        plt.title('Horizontal: Sensor ' + str(sensorID))
        plt.xlabel('x')
        plt.ylabel('x residuals')
        plt.show()
    
        plt.figure()
        plt.scatter( features_train[:, 0], labels_train[:, 1] + mean_label[1])
        plt.title('Horizontal: Sensor ' + str(sensorID))
        plt.xlabel('x')
        plt.ylabel('y residuals')  
        plt.show()
    
        plt.figure()
        plt.scatter( features_train[:, 1], labels_train[:, 0] + mean_label[0])
        plt.title('Vertical: Sensor ' + str(sensorID))
        plt.xlabel('y')
        plt.ylabel('x residuals')
        plt.show()
    
        plt.figure()
        plt.scatter( features_train[:, 1], labels_train[:, 1] + mean_label[1])
        plt.title('Vertical: Sensor ' + str(sensorID))
        plt.xlabel('y')
        plt.ylabel('y residuals')  
        plt.show()
        
        plt.figure()
        plt.scatter( np.sqrt((np.reshape(xx,(-1,1))-xp)**2 + (np.reshape(yy,(-1,1))-yp)**2), pred[:, 0], label='model', color='blue')
        plt.scatter( np.sqrt((features_train_original[:, 0]-xp)**2 + (features_train_original[:, 1]-yp)**2), labels_train[:, 0] + mean_label[0], label='Data', color='red')
        plt.title('Radial: Sensor ' + str(sensorID))
        plt.xlabel('r')
        plt.ylabel('x residuals')
        plt.legend(loc="best")    
        plt.show()
    
        plt.figure()
        plt.scatter( np.sqrt((np.reshape(xx,(-1,1))-xp)**2 + (np.reshape(yy,(-1,1))-yp)**2), pred[:, 1], label='model', color='blue')
        plt.scatter( np.sqrt((features_train_original[:, 0]-xp)**2 + (features_train_original[:, 1]-yp)**2), labels_train[:, 1] + mean_label[1], label='Data', color='red')
        plt.title('Radial: Sensor ' + str(sensorID))
        plt.xlabel('r')
        plt.ylabel('y residuals')
        plt.legend(loc="best")    
        plt.show()


errors = np.asarray(errors)
print ("SensorID, Cost, NumSamples, K")
print (errors)
 
############################
### Output predicted corrections
############################
outputCost.append([cost, numSamples, bestK, bestMaxSamples, bestMaxSamples])
outputCost = np.asarray(outputCost)

t0 = time()
np.savetxt(phoFilename, pho, '%i %i %f %f %f %f %f %f', delimiter=' ', newline='\n')

print ("TotalCost, Redundancy, hyperparmeters")
print (outputCost)

print ("outputting KNNCost.jck")
np.savetxt('/home/jckchow/BundleAdjustment/build/kNNCost.jck', outputCost, '%f %f %f %f %f', delimiter=' ', newline='\n')
print ("Done outputting results:", round(time()-t0, 3), "s")
