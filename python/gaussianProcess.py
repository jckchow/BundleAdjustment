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
#outputFilename = 'Z:/2017_Drone/2017-02-02_M100v2/slam_out_pose_GPSmoothed.csv'

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
sigma2 = gaussian_process.kernels.ConstantKernel(1.0, constant_value_bounds=(1e-010, 1e10))
RBF = gaussian_process.kernels.RBF(0.1, length_scale_bounds=(1e-10, 1e10))
#noise = gaussian_process.kernels.WhiteKernel (0.025, noise_level_bounds=(1e-03, 1.0))

# define the kernel
#K = sigma2 * RBF + noise
K = sigma2 * RBF

GP = gaussian_process.GaussianProcessRegressor(kernel=K, alpha=0.0025**2, n_restarts_optimizer=100)

t0 = time()
GP.fit(X, y)
GP.get_params()
GP.log_marginal_likelihood()
GP.kernel_.theta
print"GP Learning Time:", round(time()-t0, 3), "s"

# save the learned GP
joblib.dump(GP, 'GPLearnedCalibrationModel.pkl') 
############################
### GP Prediction
############################
t0 = time()
minX = np.min(X,axis=0)
maxX = np.max(X,axis=0)
xPred1 = np.linspace(minX[0], maxX[0], numGrid)
xPred2 = np.linspace(minX[1], maxX[1], numGrid)
U, V  = np.meshgrid(xPred1, xPred2)
u = U.flatten()
v = V.flatten()
u = np.reshape(u, (-1,1))
v = np.reshape(v, (-1,1))
xPred = np.hstack((u,v))
yPred, sigma = GP.predict(xPred, return_std=True)
print"GP Predicting Time:", round(time()-t0, 3), "s"

# restore the centroid
yPred += centroid
y += centroid
############################
### Plot the GP prediction
############################
#fig = plt.figure()
#plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
#         np.concatenate([y_pred[:,0:1] - 1.9600 * sigma,
#                        (y_pred[:,0:1] + 1.9600 * sigma)[::-1]]),
#         alpha=.2, fc='y', ec='None')
#plt.plot(t[indexBegin:indexEnd], trajectory[indexBegin:indexEnd,0:1], 'r.', label=u'Hector SLAM')
#plt.plot(x_pred, y_pred[:,0:1], 'b-', label=u'GP Smoothed')
#plt.xlabel('$Time [s]$')
#plt.ylabel('$X [m]$')
#plt.legend(loc='upper left')
#plt.title("Initial ${\Theta}$: %s\nOptimum ${\Theta}$: %s\nLog-Marginal-Likelihood: %s"
#          % (K, GP.kernel_,
#             GP.log_marginal_likelihood(GP.kernel_.theta)))
#plt.show()
#
#fig = plt.figure()
#plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
#         np.concatenate([y_pred[:,1:2] - 1.9600 * sigma,
#                        (y_pred[:,1:2] + 1.9600 * sigma)[::-1]]),
#         alpha=.2, fc='y', ec='None')
#plt.plot(t[indexBegin:indexEnd], trajectory[indexBegin:indexEnd,1:2], 'r.', label=u'Hector SLAM')
#plt.plot(x_pred, y_pred[:,1:2], 'b-', label=u'GP Smoothed')
#plt.xlabel('$Time [s]$')
#plt.ylabel('$Y [m]$')
#plt.legend(loc='upper left')
#plt.title("Initial ${\Theta}$: %s\nOptimum ${\Theta}$: %s\nLog-Marginal-Likelihood: %s"
#          % (K, GP.kernel_,
#             GP.log_marginal_likelihood(GP.kernel_.theta)))
#plt.show()
#
#fig = plt.figure()
#plt.plot(trajectory[indexBegin:indexEnd,0:1],trajectory[indexBegin:indexEnd,1:2], 'r-', label=u'Hector SLAM')
#plt.plot(y_pred[:,0:1], y_pred[:,1:2], 'b-', label=u'GP Smoothed')
#plt.xlabel('$X [m]$')
#plt.ylabel('$Y [m]$')
#plt.legend(loc='lower left')
#plt.show()

xResidual = np.reshape(yPred[:,0],(numGrid,numGrid))
yResidual = np.reshape(yPred[:,1],(numGrid,numGrid))

fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(U, V, xResidual, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('x-residuals')
## Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.scatter(X[:,0], X[:,1], y[:,0], c='g', marker='*')

plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(U, V, yResidual, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('y-residuals')
## Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.scatter(X[:,0], X[:,1], y[:,1], c='g', marker='*')

plt.show()

############################
### Output GP Smoothed Trajectory
############################
#t0 = time()
#data = np.genfromtxt(inputFilename, delimiter=',', dtype=None)
#for n in range(1,len(data)):
#    data[n][4] = str(y_pred[n-1,0])
#    data[n][5] = str(y_pred[n-1,1])
#
#np.savetxt(outputFilename, data, '%s', delimiter=',', newline='\n')
#print"Done output smoothed trajectory:", round(time()-t0, 3), "s"