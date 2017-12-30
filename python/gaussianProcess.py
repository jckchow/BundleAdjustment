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
from sklearn import gaussian_process
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

##################################
### User defined parameters
##################################   
inputFilename  = '/home/jckchow/BundleAdjustment/build/image.jck'
#phoFile = '/home/jckchow/BundleAdjustment/Data/Dcs28mmTemp.pho'
phoFile = '/home/jckchow/BundleAdjustment/xrayData1/xray1TrainingTemp.pho'


numGrid = 100
##########################################
### read in the residuals output from bundle adjustment
# x, y, v_x, v_y
##########################################
data = np.genfromtxt(inputFilename, delimiter=' ', skip_header=0, usecols = (0,1,2,3))

############################
### Remove the mean
############################
X = data[:,0:2]
y = data[:,2:4]

centroid = np.mean(y, axis=0)
y -= centroid
############################
### GP Training
############################
sigma2 = gaussian_process.kernels.ConstantKernel(1.0, constant_value_bounds=(1e-5, 1e5))
RBF = gaussian_process.kernels.RBF(0.1, length_scale_bounds=(1e-5, 1e5))
#noise = gaussian_process.kernels.WhiteKernel (0.025, noise_level_bounds=(1e-5, 1.0e5))

# define the kernel
#K = sigma2 * RBF + noise
K = sigma2 * RBF

GP = gaussian_process.GaussianProcessRegressor(kernel=K, alpha=0.025**2, n_restarts_optimizer=100)
#GP = gaussian_process.GaussianProcessRegressor(kernel=K, n_restarts_optimizer=100)

t0 = time()
GP.fit(X, y)
#GP.get_params()
#GP.log_marginal_likelihood()
#GP.kernel_.theta
print "GP Log Marginal Likelihood:", GP.log_marginal_likelihood()
print "GP Learning Time:", round(time()-t0, 3), "s"

# save the learned GP
joblib.dump(GP, 'GPLearnedCalibrationModel.pkl') 
#############################
#### GP Prediction
#############################
## predict on a grid for purely visualization purposes
#t0 = time()
#minX = np.min(X,axis=0)
#maxX = np.max(X,axis=0)
#xPred1 = np.linspace(minX[0], maxX[0], numGrid)
#xPred2 = np.linspace(minX[1], maxX[1], numGrid)
#
##xPred1 = np.concatenate( (xPred1, X[:,0].flatten()) , axis=0)
##xPred2 = np.concatenate( (xPred2, X[:,1].flatten()) , axis=0)
#
#U, V  = np.meshgrid(xPred1, xPred2)
#u = U.flatten()
#v = V.flatten()
#u = np.reshape(u, (-1,1))
#v = np.reshape(v, (-1,1))
#xPred = np.hstack((u,v))
#
##yPred, sigma = GP.predict(xPred, return_std=True)
#yPred = GP.predict(xPred, return_std=False)
#
##yPred2 = yPred
##for n in range(0,len(xPred[:,0])):
##    yPredTemp, sigmaTemp = GP.predict(np.reshape(xPred[n,:], (1,-1)), return_std=True)
##    yPred2[n,0] = yPredTemp[0,0]
##    yPred2[n,1] = yPredTemp[0,1]
#
#print"GP Predicting Time:", round(time()-t0, 3), "s"
#
## restore the centroid
#yPred += centroid
#y += centroid
#############################
#### Plot the GP prediction
#############################
#xResidual = np.reshape(yPred[:,0],(numGrid,numGrid))
#yResidual = np.reshape(yPred[:,1],(numGrid,numGrid))
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
## Plot the surface.
#surf = ax.plot_surface(U, V, xResidual, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('x-residuals')
### Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)
#
#ax.scatter(X[:,0], X[:,1], y[:,0], c='g', marker='*')
#
#plt.show()
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
## Plot the surface.
#surf = ax.plot_surface(U, V, yResidual, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('y-residuals')
### Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)
#
#ax.scatter(X[:,0], X[:,1], y[:,1], c='g', marker='*')
#
#plt.show()

############################
### Output predicted corrections
############################
t0 = time()
data = np.genfromtxt(phoFile, delimiter=' ', skip_header=0, usecols = (0,1,2,3,4,5,6,7))

xPred = data[:,2:4]
yPred, sigma = GP.predict(xPred, return_std=True)

yPred += centroid

#we want to output the corrections so we want the negative of what we predicted
#yPred = -yPred
for n in range(0,len(data)):
    data[n][6] = yPred[n,0]
    data[n][7] = yPred[n,1]

np.savetxt(phoFile, data, '%i %i %f %f %f %f %f %f', delimiter=' ', newline='\n')
print"Done outputting image corrections:", round(time()-t0, 3), "s"