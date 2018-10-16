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

##########################################
### Functions 
##########################################
def integrateAbsAngles(R, opk):

    #integrate the absolute angles
    opk1 = np.fabs(opk)
    
    ## rotation from map to sensor1
    deltaR = np.zeros((3,3))
    deltaR[0,0] = np.cos(opk1[1]) * np.cos(opk1[2]);
    deltaR[0,1] = np.cos(opk1[0]) * np.sin(opk1[2]) + np.sin(opk1[0]) * np.sin(opk1[1]) * np.cos(opk1[2]);
    deltaR[0,2] = np.sin(opk1[0]) * np.sin(opk1[2]) - np.cos(opk1[0]) * np.sin(opk1[1]) * np.cos(opk1[2]);
    
    deltaR[1,0] = -np.cos(opk1[1]) * np.sin(opk1[2]);
    deltaR[1,1] = np.cos(opk1[0]) * np.cos(opk1[2]) - np.sin(opk1[0]) * np.sin(opk1[1]) * np.sin(opk1[2]);
    deltaR[1,2] = np.sin(opk1[0]) * np.cos(opk1[2]) + np.cos(opk1[0]) * np.sin(opk1[1]) * np.sin(opk1[2]);
    
    deltaR[2,0] = np.sin(opk1[1]);
    deltaR[2,1] = -np.sin(opk1[0]) * np.cos(opk1[1]);
    deltaR[2,2] = np.cos(opk1[0]) * np.cos(opk1[1]);
    
    # deltaR_1to2
    M = np.matmul(deltaR, R)
    
    return M
    
    
def calculateChangeAngles(opk1, opk2):
    
    ## rotation from map to sensor1
    R1 = np.zeros((3,3))
    R1[0,0] = np.cos(opk1[1]) * np.cos(opk1[2]);
    R1[0,1] = np.cos(opk1[0]) * np.sin(opk1[2]) + np.sin(opk1[0]) * np.sin(opk1[1]) * np.cos(opk1[2]);
    R1[0,2] = np.sin(opk1[0]) * np.sin(opk1[2]) - np.cos(opk1[0]) * np.sin(opk1[1]) * np.cos(opk1[2]);
    
    R1[1,0] = -np.cos(opk1[1]) * np.sin(opk1[2]);
    R1[1,1] = np.cos(opk1[0]) * np.cos(opk1[2]) - np.sin(opk1[0]) * np.sin(opk1[1]) * np.sin(opk1[2]);
    R1[1,2] = np.sin(opk1[0]) * np.cos(opk1[2]) + np.cos(opk1[0]) * np.sin(opk1[1]) * np.sin(opk1[2]);
    
    R1[2,0] = np.sin(opk1[1]);
    R1[2,1] = -np.sin(opk1[0]) * np.cos(opk1[1]);
    R1[2,2] = np.cos(opk1[0]) * np.cos(opk1[1]);
    

    ## rotation from map to sensor2
    R2 = np.zeros((3,3))
    R2[0,0] = np.cos(opk2[1]) * np.cos(opk2[2]);
    R2[0,1] = np.cos(opk2[0]) * np.sin(opk2[2]) + np.sin(opk2[0]) * np.sin(opk2[1]) * np.cos(opk2[2]);
    R2[0,2] = np.sin(opk2[0]) * np.sin(opk2[2]) - np.cos(opk2[0]) * np.sin(opk2[1]) * np.cos(opk2[2]);
    
    R2[1,0] = -np.cos(opk2[1]) * np.sin(opk2[2]);
    R2[1,1] = np.cos(opk2[0]) * np.cos(opk2[2]) - np.sin(opk2[0]) * np.sin(opk2[1]) * np.sin(opk2[2]);
    R2[1,2] = np.sin(opk2[0]) * np.cos(opk2[2]) + np.cos(opk2[0]) * np.sin(opk2[1]) * np.sin(opk2[2]);
    
    R2[2,0] = np.sin(opk2[1]);
    R2[2,1] = -np.sin(opk2[0]) * np.cos(opk2[1]);
    R2[2,2] = np.cos(opk2[0]) * np.cos(opk2[1]);
    
    # deltaR_1to2
    deltaR_1to2 = np.matmul(R2, R1.transpose())
    
    deltaOPK12 = np.zeros(3)
    deltaOPK12[0] = np.arctan2(-deltaR_1to2[2,1],deltaR_1to2[2,2])
    deltaOPK12[1] = np.arcsin(deltaR_1to2[2,0])
    deltaOPK12[2] = np.arctan2(-deltaR_1to2[1,0],deltaR_1to2[0,0])

    # rotation angles from 1 to 2
    return deltaOPK12

##########################################
### user defined parameteres
##########################################

#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/NewResults/TrainAB/EOP.jck'


#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/ExternalControl/Before_AB/EOP.jck'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/ExternalControl/TrainAB_IOP_TestAB/EOP.jck'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/ExternalControl/TrainAB_TestAB/EOP.jck'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/ExternalControl/TrainAB_TestAB_Separate/EOP.jck'
    
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/ExternalControl/Before_A/EOP.jck'    
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/ExternalControl/TrainA_TestA/EOP.jck'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/ExternalControl/TrainAB_TestA/EOP.jck'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/ExternalControl/TrainA_IOP_TestA/EOP.jck'
    
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/ExternalControl/Before_B/EOP.jck'  
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/ExternalControl/TrainB_TestB/EOP.jck'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/ExternalControl/TrainAB_TestB/EOP.jck'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/ExternalControl/TrainB_IOP_TestB_moreIter/EOP.jck'

#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/InternalControl/Before_AB/EOP.jck'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/InternalControl/Before_A/EOP.jck'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/NewResults/TrainA/EOP.jck'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/InternalControl/Before_B/EOP.jck'
#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/NewResults/TrainB2/EOP.jck'

#eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/InternalControl/Before_AB_IOP/EOP.jck'
##eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/IOP/Train_AB/EOP.jck'
##eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/InternalControl/Before_A_IOP/EOP.jck'
##eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/IOP/Train_A/EOP.jck'
##eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/InternalControl/Before_B_IOP/EOP.jck'
##eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/IOP/Train_B_moreIter/EOP.jck'
#eopTruthFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1Truth.eop'
    
eopFilename = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingResults/Training150_photoROP_IOP_linearSmoothing200/EOP.jck'
eopTruthFilename = '/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROP.eop'

numSamples = 75

##########################################
### Process eop data
##########################################
eop = np.genfromtxt(eopFilename, delimiter=' ', skip_header=0, usecols = (0,1,2,3,4,5,6,7,8,9,10,11,12))
eopTruth = np.genfromtxt(eopTruthFilename, delimiter=' ', skip_header=0, usecols = (0,1,2,3,4,5,6))

ID  = eop[0:numSamples,(0)].astype(int)
XYZ = eop[0:numSamples,(4,5,6)]
OPK = eop[0:numSamples,(1,2,3)] * np.pi/180.0

IDTrue  = eopTruth[:,(0)].astype(int)
XYZTrue = eopTruth[:,(1,2,3)]
OPKTrue = eopTruth[:,(4,5,6)] * np.pi/180.0

# computer error in camera position
print 'Start computing errors in camera position...'
diffXYZ = np.zeros((len(ID),3))
for n in range(0,len(ID)):
    index = np.argwhere(ID[n] == IDTrue)
#    print XYZTrue[index,:]
#    print XYZ[n,:]
    diffXYZ[n,:] = XYZ[n,:] - XYZTrue[index,:]

print '  RMSE Xo, Yo, Zo: ', np.sqrt( np.mean(diffXYZ**2,axis=0) )

# computer error in camera orientation
print 'Start computing errors in camera orientation...'
diffR = np.identity(3)
omega = np.zeros((len(ID)))
phi   = np.zeros((len(ID)))
kappa = np.zeros((len(ID)))
theta = np.zeros((len(ID)))
vectorOPK = np.zeros((len(ID),3))
for n in range(0,len(ID)):
    index = np.argwhere(ID[n] == IDTrue)
    diffOPK =  calculateChangeAngles(OPK[n,:],OPKTrue[index,:].flatten())
    
    vectorOPK[n,(0,1,2)] = np.fabs(diffOPK)
    
#    print diffOPK * 180.0 / np.pi
    diffOPK /= len(ID) # averaging
    diffR = integrateAbsAngles(diffR, diffOPK)
    
    theta[n] = np.arccos(0.5 * (diffR[0,0]+diffR[1,1]+diffR[2,2]) - 0.5)
    
    omega[n] = np.arctan2(-diffR[2,1],diffR[2,2])
    phi[n]   = np.arcsin(diffR[2,0])
    kappa[n] = np.arctan2(-diffR[1,0],diffR[0,0])
    
deltaOPK = np.zeros(3)
deltaOPK[0] = np.arctan2(-diffR[2,1],diffR[2,2])
deltaOPK[1] = np.arcsin(diffR[2,0])
deltaOPK[2] = np.arctan2(-diffR[1,0],diffR[0,0])

print '  Integrated absolute omega, phi, kappa [deg]: ', deltaOPK * 180.0 / np.pi
#print '  Integrated then averaged absolute omega, phi, kappa [deg]: ', (deltaOPK / len(ID) ) * 180.0 / np.pi
print '  Average absolute omega, phi, kappa [deg]: ', deltaOPK * 180.0 / np.pi / len(ID)
print '  Average absolute vector omega, phi, kappa [deg]: ', np.mean(vectorOPK,axis = 0) * 180.0 / np.pi

plt.figure()
plt.plot(range(0,len(ID)), omega * 180.0 / np.pi, color = 'darkorange')
plt.title('Omega')
plt.ylabel('Degrees')

plt.figure()
plt.plot(range(0,len(ID)), phi * 180.0 / np.pi, color = 'cyan')
plt.title('Phi')
plt.ylabel('Degrees')

plt.figure()
plt.plot(range(0,len(ID)), kappa * 180.0 / np.pi, color = 'violet')
plt.title('Kappa')
plt.ylabel('Degrees')

plt.figure()
plt.plot(range(0,len(ID)), theta * 180.0 / np.pi, color = 'magenta')
plt.title('Axis-Angle')
plt.ylabel('Theta [deg]')