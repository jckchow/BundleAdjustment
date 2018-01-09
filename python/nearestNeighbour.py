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
from sklearn.grid_search import GridSearchCV
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
inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'
phoFilename = '/home/jckchow/BundleAdjustment/Data/Dcs28mmTemp.pho'
iopFilename = '/home/jckchow/BundleAdjustment/Data/Dcs28mm.iop'
eopFilename = '/home/jckchow/BundleAdjustment/Data/Dcs28mm.eop'

output = np.genfromtxt(phoFilename, delimiter=' ', skip_header=0, usecols = (0,1,2,3,4,5,6,7))

##########################################
### read in the residuals output from bundle adjustment
# x, y, v_x, v_y, redu_x, redu_y, vStdDev_x, vStdDev_y
##########################################
data = np.genfromtxt(inputFilename, delimiter=' ', skip_header=0, usecols = (0,1,2,3,4,5,6,7,8))
iop =  np.genfromtxt(iopFilename, delimiter=' ', skip_header=0, usecols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15))
iop = np.atleast_2d(iop)
eop = np.genfromtxt(eopFilename, delimiter=' ', skip_header=0, usecols = (0,1)).astype(int)

w = np.divide(data[:,(3,4)], data[:,(7,8)])

# 95% is 1.96
outlierThreshold = 1.96
outlierIndex = np.argwhere(np.fabs(w) > outlierThreshold)

inliers = np.delete(data, outlierIndex[:,0], axis=0)

sensorsUnique = np.unique(data[:,0])


##########################################
### Learn for each sensor separately
##########################################
for iter in range(0,len(sensorsUnique)):
    
    sensorID = sensorsUnique[iter]
    indexImage = np.argwhere(inliers[:,0] == sensorID)
    indexIOP = np.argwhere(iop[:,0] == sensorID)
    indexEOP = np.argwhere(eop[:,1] == sensorID)   
    
    stationsWithThisIOP = np.unique(eop[indexEOP,0]) # should not be needed since it should be unique
    
    stationsWithThisEOP = []
    for n in range(0,len(stationsWithThisIOP)):
        temp = np.argwhere(output[:,1] == stationsWithThisIOP[n])
        stationsWithThisEOP.append(temp.flatten().tolist())
    stationsWithThisEOP = np.concatenate(np.asarray(stationsWithThisEOP))
    ##########################################
    ### Training
    ##########################################

    
    features_train = inliers[indexImage,(1,2)]
    labels_train = inliers[indexImage,(3,4)]
    
#    reg = neighbors.KNeighborsRegressor(n_neighbors=15, weights='uniform')
#    reg.fit(features_train, labels_train)
    # score = clf.score(features_test, labels_test)    
    
    t0 = time()
    param_grid = [ {'n_neighbors' : range(5,50)} ]
    reg = GridSearchCV(neighbors.KNeighborsRegressor(weights='uniform'), param_grid, cv=10, verbose = 0)
    reg.fit(features_train, labels_train)
    print "Best in sample score: ", reg.best_score_
    print "CV value for K: ", reg.best_estimator_.n_neighbors
    print "Training NN-Regressor + CV time:", round(time()-t0, 3), "s"
    
    reg = neighbors.KNeighborsRegressor(n_neighbors=reg.best_estimator_.n_neighbors, weights='uniform')
    reg.fit(features_train, labels_train)
    
    ##########################################
    ### Prediction
    ########################################## 
    t0 = time()
    stationsWithThisEOP = np.reshape(stationsWithThisEOP, (-1,1))
    correction = reg.predict(output[stationsWithThisEOP,(2,3)])  
    output[stationsWithThisEOP,(6,7)] = correction
    print"Done predicting:", round(time()-t0, 3), "s"    
    
    ##########################################
    ### Plotting
    ##########################################


    xx, yy = np.meshgrid(np.arange(iop[indexIOP,2], iop[indexIOP,4], 1),
                         np.arange(iop[indexIOP,3], -iop[indexIOP,5], -1))

    pred = reg.predict(np.hstack((np.reshape(xx,(-1,1)), np.reshape(yy,(-1,1)))))
    
    xp = iop[indexIOP,6]
    yp = iop[indexIOP,7]
        
    # Plot the training points    
    plt.figure()
    plt.scatter(features_train[:, 0], features_train[:, 1], color = 'darkorange')
    plt.title('Image measurements for sensor ' + str(sensorID))
    plt.xlabel('x')
    plt.ylabel('y')
#    plt.xlim([iop[indexIOP,2], iop[indexIOP,4]]) # set it to the image format
#    plt.ylim([iop[indexIOP,3], -iop[indexIOP,5]])
    
    # define grid.
    xi = np.arange(iop[indexIOP,2], iop[indexIOP,4], 1)
    yi = np.arange(iop[indexIOP,3], -iop[indexIOP,5], -1)
    
    plt.figure()
    # grid the data.
    zi = griddata(features_train[:, 0], features_train[:, 1], labels_train[:, 0], xi, yi, interp='linear')
    # contour the gridded data, plotting dots at the nonuniform data points.
    CS = plt.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
    CS = plt.contourf(xi, yi, zi, 15,
                      vmax=abs(zi).max(), vmin=-abs(zi).max())
    plt.colorbar()  # draw colorbar
    # plot data points.
    plt.scatter(features_train[:, 0], features_train[:, 1], marker='o', color='red', s=5, zorder=10)
    plt.title('x residuals: Sensor ' + str(sensorID))

    zz = np.reshape(pred[:,0], np.shape(xx))
    plt.figure()
    # contour the gridded data, plotting dots at the nonuniform data points.
    CS = plt.contour(xx, yy, zz, 15, linewidths=0.5, colors='k')
    CS = plt.contourf(xx, yy, zz, 15,
                      vmax=abs(zi).max(), vmin=-abs(zi).max())
    plt.colorbar()  # draw colorbar
    # plot data points.
    plt.scatter(features_train[:, 0], features_train[:, 1], marker='o', color='red', s=5, zorder=10)
    plt.title('x model: Sensor ' + str(sensorID))
    
    plt.figure()
    # contour the gridded data, plotting dots at the nonuniform data points.
    CS = plt.contour(xx, yy, zz, 15, linewidths=0.5, colors='k')
    CS = plt.contourf(xx, yy, zz, 15,
                      vmax=abs(zz).max(), vmin=-abs(zz).max())
    plt.colorbar()  # draw colorbar
    # plot data points.
    plt.scatter(features_train[:, 0], features_train[:, 1], marker='o', color='red', s=5, zorder=10)
    plt.title('x model: Sensor ' + str(sensorID))

    
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
    zi = griddata(features_train[:, 0], features_train[:, 1], labels_train[:, 1], xi, yi, interp='linear')
    # contour the gridded data, plotting dots at the nonuniform data points.
    CS = plt.contourf(xi, yi, zi, 15, linewidths=0.5, colors='k')
    CS = plt.contourf(xi, yi, zi, 15,
                      vmax=abs(zi).max(), vmin=-abs(zi).max())
    plt.colorbar()  # draw colorbar
    # plot data points.
    plt.scatter(features_train[:, 0], features_train[:, 1], marker='o', color='red', s=5, zorder=10)
    plt.title('y residuals: Sensor ' + str(sensorID))
    plt.show()
 
    zz = np.reshape(pred[:,1], np.shape(xx))
    plt.figure()
    # contour the gridded data, plotting dots at the nonuniform data points.
    CS = plt.contour(xx, yy, zz, 15, linewidths=0.5, colors='k')
    CS = plt.contourf(xx, yy, zz, 15,
                      vmax=abs(zi).max(), vmin=-abs(zi).max())
    plt.colorbar()  # draw colorbar
    # plot data points.
    plt.scatter(features_train[:, 0], features_train[:, 1], marker='o', color='red', s=5, zorder=10)
    plt.title('y model: Sensor ' + str(sensorID))
    
    plt.figure()
    # contour the gridded data, plotting dots at the nonuniform data points.
    CS = plt.contour(xx, yy, zz, 15, linewidths=0.5, colors='k')
    CS = plt.contourf(xx, yy, zz, 15,
                      vmax=abs(zz).max(), vmin=-abs(zz).max())
    plt.colorbar()  # draw colorbar
    # plot data points.
    plt.scatter(features_train[:, 0], features_train[:, 1], marker='o', color='red', s=5, zorder=10)
    plt.title('y model: Sensor ' + str(sensorID))
        
    ### 1D Plots
    plt.figure()
    plt.scatter( features_train[:, 0], labels_train[:, 0])
    plt.title('Horizontal: Sensor ' + str(sensorID))
    plt.xlabel('x')
    plt.ylabel('x residuals')

    plt.figure()
    plt.scatter( features_train[:, 0], labels_train[:, 1])
    plt.title('Horizontal: Sensor ' + str(sensorID))
    plt.xlabel('x')
    plt.ylabel('y residuals')  

    plt.figure()
    plt.scatter( features_train[:, 1], labels_train[:, 0])
    plt.title('Vertical: Sensor ' + str(sensorID))
    plt.xlabel('y')
    plt.ylabel('x residuals')

    plt.figure()
    plt.scatter( features_train[:, 1], labels_train[:, 1])
    plt.title('Vertical: Sensor ' + str(sensorID))
    plt.xlabel('y')
    plt.ylabel('y residuals')  
    
    plt.figure()
    plt.scatter( np.sqrt((np.reshape(xx,(-1,1))-xp)**2 + (np.reshape(yy,(-1,1))-yp)**2), pred[:, 0], label='model', color='blue')
    plt.scatter( np.sqrt((features_train[:, 0]-xp)**2 + (features_train[:, 1]-yp)**2), labels_train[:, 0], label='Data', color='red')
    plt.title('Radial: Sensor ' + str(sensorID))
    plt.xlabel('r')
    plt.ylabel('x residuals')

    plt.figure()
    plt.scatter( np.sqrt((np.reshape(xx,(-1,1))-xp)**2 + (np.reshape(yy,(-1,1))-yp)**2), pred[:, 1], label='model', color='blue')
    plt.scatter( np.sqrt((features_train[:, 0]-xp)**2 + (features_train[:, 1]-yp)**2), labels_train[:, 1], label='Data', color='red')
    plt.title('Radial: Sensor ' + str(sensorID))
    plt.xlabel('r')
    plt.ylabel('y residuals')
    plt.legend(loc="best")    

    
############################
### Output predicted corrections
############################
t0 = time()
np.savetxt(phoFilename, output, '%i %i %f %f %f %f %f %f', delimiter=' ', newline='\n')
print"Done outputting image corrections:", round(time()-t0, 3), "s"