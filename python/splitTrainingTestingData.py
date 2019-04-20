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

##################################
### User defined parameters
##################################   
### First paper
#inputPho = '/home/jckchow/BundleAdjustment/xrayData1/xray1.pho'
#inputEop = '/home/jckchow/BundleAdjustment/xrayData1/xray1.eop'
#
#outputEopTraining = '/home/jckchow/BundleAdjustment/xrayData1/xray1Training.eop'
#outputEopTesting = '/home/jckchow/BundleAdjustment/xrayData1/xray1Testing.eop'
#
#outputPhoTraining = '/home/jckchow/BundleAdjustment/xrayData1/xray1Training.pho'
#outputPhoTesting = '/home/jckchow/BundleAdjustment/xrayData1/xray1Testing.pho'
#
#chooseIncrement = 10

### Second paper
#inputPho = '/home/jckchow/BundleAdjustment/xrayData1/xray1.pho'
#inputEop = '/home/jckchow/BundleAdjustment/xrayData1/xray1.eop'
#
#outputEopTraining = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/xray1Training.eop'
#outputEopTesting = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/xray1Testing.eop'
#
#outputPhoTraining = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/xray1Training.pho'
#outputPhoTesting = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/xray1Testing.pho'

#####
## Paper 2: X-ray
#####

#inputPho = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training150.pho'
#inputEop = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training120.eop'
#
#outputEopTraining = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training.eop'
#outputEopTesting = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Testing.eop'
#
#outputPhoTraining = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training.pho'
#outputPhoTesting = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Testing.pho'

inputPho = '/home/jckchow/BundleAdjustment/xrayData1/xray1.pho'
inputEop = '/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROP.eop'

outputEopTraining = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingROP.eop'
outputEopTesting = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TestingROP.eop'

outputPhoTraining = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingROP.pho'
outputPhoTesting = '/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TestingROP.pho'


chooseIncrement = 2

#####
## Omni Paper 1 TC 1: Nikon
#####
#inputPho = '/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/nikonLess.pho'
#inputEop = '/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/nikonRobust.eop'
#
#outputEopTraining = '/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/TrainingTesting/nikonTraining.eop'
#outputEopTesting = '/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/TrainingTesting/nikonTesting.eop'
#
#outputPhoTraining = '/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/TrainingTesting/nikonTraining.pho'
#outputPhoTesting = '/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/TrainingTesting/nikonTesting.pho'

#### go pro
#inputPho = '/home/jckchow/BundleAdjustment/omnidirectionalCamera/gopro/gopro.pho'
#inputEop = '/home/jckchow/BundleAdjustment/omnidirectionalCamera/gopro/gopro.eop'
#
#outputEopTraining = '/home/jckchow/BundleAdjustment/omnidirectionalCamera/gopro/TrainingTesting/goproTraining.eop'
#outputEopTesting = '/home/jckchow/BundleAdjustment/omnidirectionalCamera/gopro/TrainingTesting/goproTesting.eop'
#
#outputPhoTraining = '/home/jckchow/BundleAdjustment/omnidirectionalCamera/gopro/TrainingTesting/goproTraining.pho'
#outputPhoTesting = '/home/jckchow/BundleAdjustment/omnidirectionalCamera/gopro/TrainingTesting/goproTesting.pho'

#chooseIncrement = 2

##########################################
### read and split the data
##########################################
t0 = time()
pho = np.genfromtxt(inputPho, delimiter=' ', skip_header=0, usecols = (0,1,2,3,4,5,6,7))
eop = np.genfromtxt(inputEop, delimiter=' ', skip_header=0, usecols = (0,1,2,3,4,5,6,7))

print "Number of original stations: ", len(eop)

# ID of images to be used for testing. The remaining will be used for training.
testingIndex  = range(0, len(eop), chooseIncrement) # using 10% of the photos for testing
trainingIndex = range(0, len(eop), 1)
trainingIndex = np.setdiff1d(trainingIndex, testingIndex)

print "  Number of testing stations: ", len(testingIndex)
print "  Number of training stations: ", len(trainingIndex)


eopTesting = eop[testingIndex,:]
eopTraining = eop[trainingIndex,:]

eopTestingID = eopTesting[:,0]
eopTrainingID = eopTraining[:,0]

phoTesting = []
for n in range(0,len(eopTestingID)):
    index = np.argwhere(pho[:,1] == eopTestingID[n])
    phoTesting.extend(pho[index,:])
phoTesting = np.asarray(phoTesting)
phoTesting = np.reshape(phoTesting, (-1,8))

print "Number of testing samples: ", len(phoTesting)

phoTraining = []
for n in range(0,len(eopTrainingID)):
    index = np.argwhere(pho[:,1] == eopTrainingID[n])
    phoTraining.extend(pho[index,:])
phoTraining = np.asarray(phoTraining)
phoTraining = np.reshape(phoTraining, (-1,8))

print "Number of training samples: ", len(phoTraining)

############################
### Output training and testing data to file  corrections
############################
np.savetxt(outputEopTesting, eopTesting, '%i %i %f %f %f %f %f %f', delimiter=' ', newline='\n')
np.savetxt(outputEopTraining, eopTraining, '%i %i %f %f %f %f %f %f', delimiter=' ', newline='\n')

np.savetxt(outputPhoTesting, phoTesting, '%i %i %f %f %f %f %f %f', delimiter=' ', newline='\n')
np.savetxt(outputPhoTraining, phoTraining, '%i %i %f %f %f %f %f %f', delimiter=' ', newline='\n')

print"Done outputting training and testing data:", round(time()-t0, 3), "s"