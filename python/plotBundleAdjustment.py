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
from matplotlib import pyplot as plt
import random

##################################
### User defined parameters
##################################   
inputFilename  = '/home/jckchow/BundleAdjustment/build/residuals.jck'
outputDirectory = '/home/jckchow/BundleAdjustment/build/'


##################################
### Read the data
##################################  
data = np.genfromtxt(inputFilename, delimiter=' ', skip_header=0, usecols = (0,1,2,3,4,5,6,7,8, 9, 10))

pointID         = data[:,0].astype(int);
frameID         = data[:,1].astype(int);
sensorID        = data[:,2].astype(int);
xImg            = data[:,3];
yImg            = data[:,4];
xResidual       = data[:,5];
yResidual       = data[:,6];
xRedundancy     = data[:,7];
yRedundancy     = data[:,8];
xResidualStdDev = data[:,9];
yResidualStdDev = data[:,10];


sensorsUnique = np.unique(sensorID);
stationsUnique = np.unique(frameID);

##################################
### Plot Data
################################## 

# plot each sensor separately
for iter in range(0,len(sensorsUnique)): # iterate and calibrate each sensor
    
    
    cameraID = sensorsUnique[iter]; #currently sensor ID
    indexImage = np.argwhere(sensorID == cameraID);
    
    print ("  Processing sensor: ", cameraID)
    
    
    # plot the residuals
    random.seed( 0 );
    fig = plt.figure(figsize=(8.0, 5.0))
    for i in range(0,len(stationsUnique)):
        I = np.argwhere(frameID == stationsUnique[i])
#        fig = plt.scatter(pointID[I], xResidual[I], s=2, c=np.atleast_2d(np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)])))
        fig = plt.plot(pointID[I], xResidual[I], color=(np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)])))
    fig = plt.title('Image Measurement Errors for Sensor ' + str(cameraID))
    fig = plt.xlabel('Point ID')
    fig = plt.ylabel('x Image Residuals')
    fig = plt.savefig(outputDirectory + str('xResiduals'), dpi=100, format="tif")
#    plt.show()

    random.seed( 1 );    
    plt.figure()
    for i in range(0,len(stationsUnique)):
        I = np.argwhere(frameID == stationsUnique[i])
        plt.scatter(pointID[I], yResidual[I], s=2, c=np.atleast_2d(np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)])))
    plt.title('Image Measurement Errors for Sensor ' + str(cameraID))
    plt.xlabel('Point ID')
    plt.ylabel('y Image Residuals')
#    plt.show()
    
    # plot the redundancy numbers
    plt.figure()
    for i in range(0,len(stationsUnique)):
        I = np.argwhere(frameID == stationsUnique[i])
        plt.scatter(pointID[I], xRedundancy[I])
    plt.title('Redundancy Numbers for Sensor ' + str(cameraID))
    plt.xlabel('Point ID')
    plt.ylabel('x Redundancy Number')
#    plt.show()
        
    plt.figure()
    for i in range(0,len(stationsUnique)):
        I = np.argwhere(frameID == stationsUnique[i])
        plt.scatter(pointID[I], yRedundancy[I])
    plt.title('Redundancy Numbers for Sensor ' + str(cameraID))
    plt.xlabel('Point ID')
    plt.ylabel('y Redundancy Number')
#    plt.show()
    
    # plot the standard deviation of residuals
    plt.figure()
    for i in range(0,len(stationsUnique)):
        I = np.argwhere(frameID == stationsUnique[i])
        plt.scatter(pointID[I], xResidualStdDev[I])
    plt.title('Std Dev of Residuals for Sensor ' + str(cameraID))
    plt.xlabel('Point ID')
    plt.ylabel('x Residual Std Dev')
#    plt.show()
    
    plt.figure()
    for i in range(0,len(stationsUnique)):
        I = np.argwhere(frameID == stationsUnique[i])
        plt.scatter(pointID[I], yResidualStdDev[I])
    plt.title('Std Dev of Residuals for Sensor ' + str(cameraID))
    plt.xlabel('Point ID')
    plt.ylabel('y Residual Std Dev')
#    plt.show()


    print("Statistics of Residuals")    
    print("   Average x Residuals: " + str(np.mean(xResidual)))
    print("   Min x Residuals: " + str(np.min(xResidual)))
    print("   Max x Residuals: " + str(np.max(xResidual)))
    print("   Average y Residuals: " + str(np.mean(yResidual)))
    print("   Min y Residuals: " + str(np.min(yResidual)))
    print("   Max y Residuals: " + str(np.max(yResidual)))
    
    print("Statistics of Redundancy Numbers")    
    print("   Average x Redundancy: " + str(np.mean(xRedundancy)))
    print("   Min x Redundancy: " + str(np.min(xRedundancy)))
    print("   Max x Redundancy: " + str(np.max(xRedundancy)))
    print("   Average y Redundancy: " + str(np.mean(yRedundancy)))
    print("   Min y Redundancy: " + str(np.min(yRedundancy)))
    print("   Max y Redundancy: " + str(np.max(yRedundancy)))

    
    plt.show()
