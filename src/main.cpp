////////////////////////////////////////////////////////////////////////////
///
///   COPYRIGHT (C) 2017 Vusion Technologies, all rights reserved.
///
///   No part of this software may be reproduced or modified in any
///   form or by any means - electronic, mechanical, photocopying,
///   recording, or otherwise - without the prior written consent of
///   Vusion Technologies
///
///   Author: Jacky Chow
///   Date: December 23, 2017
///   Description: Photogrammetric Bundle Adjustment
///      - To build $ cd ~/BundleAdjustment/build
///      - $ make
///      - $ /bundleAdjustment
///
////////////////////////////////////////////////////////////////////////////

#include "ceres/ceres.h"
#include "ceres/cost_function.h"
#include "ceres/cubic_interpolation.h"
#include "glog/logging.h"
#include "Python.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>
#include <Eigen/Sparse>

//#include <pcl/point_types.h>
//#include <pcl/filters/voxel_grid.h>

// Define constants
#define PI 3.141592653589793238462643383279502884197169399
#define NUMITERATION 1 // Set it to anything greater than 1 to do ML. Otherwise, set it to 1 to do non-machine learning bundle adjustment
#define DEBUGMODE 0
#define ROPMODE 0 // Turn on absolute boresight and leverarm constraints. 1 for true, 0 for false
#define WEIGHTEDROPMODE 0 // weighted boresight and leverarm constraints. 1 for true, 0 for false
#define INITIALIZEAP 0 // if true, we will backproject good object space to calculate the initial APs in machine learning pipeline. Will need good resection and object space to do this.

#define COMPUTECX 1 // Compute covariance matrix of unknowns Cx, 1 is true, 0 is false
#define COMPUTECV 0 // Compute covariance matrix of residuals Cv, 1 is true, 0 is false. If we need Cv, we must also calculate Cx
// if (COMPUTECV)
//     #define COMPUTECX 1

#define PLOTRESULTS 1 // plots the outputs using python

// #define APSCALE 1000.0 // arbitrary scale for x_bar and y_bar, makes the inversion of matrix more stable for the AP
#define APSCALE 1.0 // arbitrary scale for x_bar and y_bar, makes the inversion of matrix more stable for the AP

#define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/Data/Dcs28mm.pho"
#define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/Data/Dcs28mmTemp.pho" 
#define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/Data/Dcs28mm.iop"
#define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/Data/Dcs28mm.eop"
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/Data/Dcs28mm.xyz"
#define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/Data/Dcs28mmLowWeight.xyz"
#define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/Data/Dcs28mmTruth.xyz"
#define INPUTROPFILENAME ""

// // #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1Training.pho"
// // #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Training270Testing30/After/xray1TrainingCalibrated.pho"
// // #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Training270Testing30/After/xray1TrainingCalibrated_separate.pho"
// // #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/NewResults/xray1Training_CalibratedAB.pho"
// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/NewResults/xray1Training_CalibratedSeparate.pho"
// // #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/IOP/xray1TrainingAB_CalibratedAB_IOP.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/xrayData1/xray1TrainingTemp.pho" 
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.iop"
// // #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/IOP/Train_AB/iop.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1Training.eop"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.xyz"
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthLowWeight.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1Truth.xyz" // only use for QC
// #define INPUTROPFILENAME ""

// // #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TrainingA.pho"
// // #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Training270Testing30/After_A/xray1TrainingCalibrated.pho"
// // #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/NewResults/xray1TrainingA_CalibratedA.pho"
// // #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/NewResults/xray1TrainingA_CalibratedAB.pho"
// // #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/IOP/xray1TrainingA_CalibratedA_IOP.pho"
// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/IOP/xray1TrainingA_CalibratedAB_IOP.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/xrayData1/xray1TrainingTemp.pho" 
// // #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1A.iop"
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/IOP/Train_AB/iopA.iop"
// // #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/IOP/Train_A/iop.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TrainingA.eop"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.xyz"
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthLowWeight.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1Truth.xyz" // only use for QC
// #define INPUTROPFILENAME ""

// // #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TrainingB.pho"
// // #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Training270Testing30/After_B/xray1TrainingCalibrated.pho"
// // #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/NewResults/xray1TrainingB_CalibratedAB.pho"
// // #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/NewResults/xray1TrainingB_CalibratedB.pho"
// // #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/IOP/xray1TrainingB_CalibratedB_moreIter_IOP.pho"
// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/IOP/xray1TrainingB_CalibratedAB_IOP.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/xrayData1/xray1TrainingTemp.pho" 
// // #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/IOP/Train_B_moreIter/iop.iop"
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/IOP/Train_AB/iopB.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TrainingB.eop"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.xyz"
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthLowWeight.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1Truth.xyz" // only use for QC
// #define INPUTROPFILENAME ""

// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1Testing.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/xrayData1/xray1TestingTemp.pho" 
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1Testing.eop"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.xyz"
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthLowWeight.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1Truth.xyz" // only use for QC
// #define INPUTROPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.rop"
// // #define INPUTROPFILENAME ""

// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TestingA.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/xrayData1/xray1TestingTemp.pho" 
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1A.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TestingA.eop"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.xyz"
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthLowWeight.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1Truth.xyz" // only use for QC
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarmLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarm.xyz" // only use for QC
// #define INPUTROPFILENAME ""

// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TestingB.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/xrayData1/xray1TestingTemp.pho" 
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1B.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TestingB.eop"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.xyz"
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthLowWeight.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1Truth.xyz" // only use for QC
// #define INPUTROPFILENAME ""

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// 
/// Paper 2: 150 Training, 150 Testing
/// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////
/// Sensor A
////////

// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training30A.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho" 
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1A.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training30A.eop"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.xyz"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1Truth.xyz" // only use for QC
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarmLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarm.xyz" // only use for QC
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROPLowWeight.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROP.xyz" // only use for QC
// #define INPUTROPFILENAME ""

// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training60A.pho"
// // #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training150A_continue.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho" 
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1A.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training60A.eop"
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROPLowWeight.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROP.xyz" // only use for QC
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarmLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarm.xyz" // only use for QC
// #define INPUTROPFILENAME ""

// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training90A.pho"
// // #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training150A_continue.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho" 
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1A.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training90A.eop"
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROPLowWeight.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROP.xyz" // only use for QC
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarmLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarm.xyz" // only use for QC
// #define INPUTROPFILENAME ""

// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training120A.pho"
// // #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training150A_continue.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho" 
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1A.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training120A.eop"
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROPLowWeight.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROP.xyz" // only use for QC
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarmLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarm.xyz" // only use for QC
// #define INPUTROPFILENAME ""

// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training150A.pho"
// // #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training150A_continue.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho" 
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1A.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training150A.eop"
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROPLowWeight.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROP.xyz" // only use for QC
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarmLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarm.xyz" // only use for QC
// #define INPUTROPFILENAME ""

////////
/// Sensor B
////////

// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training30B.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho" 
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1B.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training30B.eop"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.xyz"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1Truth.xyz" // only use for QC
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarmLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarm.xyz" // only use for QC
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROPLowWeight.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROP.xyz" // only use for QC
// #define INPUTROPFILENAME ""

// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training60B.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho" 
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1B.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training60B.eop"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.xyz"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1Truth.xyz" // only use for QC
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarmLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarm.xyz" // only use for QC
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROPLowWeight.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROP.xyz" // only use for QC
// #define INPUTROPFILENAME ""

// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training90B.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho" 
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1B.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training90B.eop"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.xyz"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1Truth.xyz" // only use for QC
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarmLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarm.xyz" // only use for QC
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROPLowWeight.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROP.xyz" // only use for QC
// #define INPUTROPFILENAME ""

// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training120B.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho" 
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1B.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training120B.eop"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.xyz"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1Truth.xyz" // only use for QC
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarmLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarm.xyz" // only use for QC
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROPLowWeight.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROP.xyz" // only use for QC
// #define INPUTROPFILENAME ""

// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training150B.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho" 
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1B.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training150B.eop"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.xyz"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1Truth.xyz" // only use for QC
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarmLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarm.xyz" // only use for QC
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROPLowWeight.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROP.xyz" // only use for QC
// #define INPUTROPFILENAME ""

///////////////////////////
// Testing on 150
///////////////////////////
// // #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TestingResults/xray1TestingA.pho"
// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TestingResults/xray1Training150AB_photoROP_old.pho"
// // #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TestingResults/Output/xray1TestingA_Training150A_photoROP_linearSmoothing_robust.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho"
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1A.iop"
// // #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TestingResults/xray1ATesting.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TestingResults/xray1TestingA.eop"
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROPLowWeight.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROP.xyz" // only use for QC
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarmLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarm.xyz" // only use for QC
// #define INPUTROPFILENAME ""

// // #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TestingResults/xray1TestingB.pho"
// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TestingResults/Output/xray1TestingB_Training150B_photoROP_IOP_linearSmoothing200.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho"
// // #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1A.iop"
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TestingResults/xray1BTesting.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TestingResults/xray1TestingB.eop"
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROPLowWeight.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROP.xyz" // only use for QC
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarmLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarm.xyz" // only use for QC
// #define INPUTROPFILENAME ""

// // #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TestingResults/xray1Testing.pho"
// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TestingResults/Output/xray1Testing_Training30_photoROP_IOP.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho"
// // #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1A.iop"
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TestingResults/xray1Testing.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TestingResults/xray1Testing.eop"
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROPLowWeight.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROP.xyz" // only use for QC
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarmLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarm.xyz" // only use for QC
// #define INPUTROPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.rop"

////////
/// Sensor A + B together
////////

// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training30.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho" 
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training30.eop"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.xyz"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1Truth.xyz" // only use for QC
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarmLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarm.xyz" // only use for QC
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROPLowWeight.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROP.xyz" // only use for QC
// // #define INPUTROPFILENAME ""
// #define INPUTROPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.rop"

// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training60.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho" 
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training60.eop"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.xyz"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1Truth.xyz" // only use for QC
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarmLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarm.xyz" // only use for QC
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROPLowWeight.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROP.xyz" // only use for QC
// // #define INPUTROPFILENAME ""
// #define INPUTROPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.rop"

// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training90.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho" 
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training90.eop"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.xyz"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1Truth.xyz" // only use for QC
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarmLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarm.xyz" // only use for QC
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROPLowWeight.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROP.xyz" // only use for QC
// // #define INPUTROPFILENAME ""
// #define INPUTROPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.rop"

// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training120.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho" 
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training120.eop"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.xyz"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1Truth.xyz" // only use for QC
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarmLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarm.xyz" // only use for QC
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROPLowWeight.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROP.xyz" // only use for QC
// // #define INPUTROPFILENAME ""
// #define INPUTROPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.rop"

// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training150.pho"
// // #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training150_continue.pho"
// // #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training150AB_photoROP.pho" // pre-calibrated each sensor individually
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1TrainingTemp.pho" 
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.iop"
// // #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training150.eop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/Data_Train150_Test150/TrainingSubset/xray1Training150_ROP.eop"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.xyz"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1Truth.xyz" // only use for QC
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarmLowWeight.xyz"
// // #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/faroarm.xyz" // only use for QC
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROPLowWeight.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1TruthROP.xyz" // only use for QC
// // #define INPUTROPFILENAME ""
// #define INPUTROPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.rop"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// 
/// Paper 1 ISPRS TC 1: Omnidirectional camera
/// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/nikonLess.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/nikonLessTemp.pho" 
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/nikon.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/nikonRobust.eop"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/nikon.xyz"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/nikonTruth.xyz"
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/nikonLowWeight.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/nikonTruth.xyz" // only use for QC
// #define INPUTROPFILENAME ""

// // for training Nikon
// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/TrainingTesting/nikonTraining.pho"
// // #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/TrainingTesting/nikonTrainingDeleteMe.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/TrainingTesting/nikonTrainingTemp.pho"
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/nikon.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/TrainingTesting/nikonTraining.eop"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/nikon.xyz"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/nikonTruth.xyz"
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/nikon.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon/nikonTruth.xyz" // only use for QC
// #define INPUTROPFILENAME ""

// // for training Go Pro Hero 3 Silver Edition
// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/gopro/TrainingTesting/goproTraining.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/omnidirectionalCamera/gopro/TrainingTesting/goproTrainingTemp.pho"
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/gopro/gopro.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/gopro/TrainingTesting/goproTraining.eop"
// // #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/gopro/goproTruth.xyz"
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/gopro/goproLowWeight.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/gopro/goproTruth.xyz" // only use for QC
// #define INPUTROPFILENAME ""


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// 
/// Paper 2 Omnidirectional Camera Journal Paper
/// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// // //for all Nikon
// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon_2020_03_23/nikon_screened.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon_2020_03_23/nikonTemp.pho"
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon_2020_03_23/nikon_updated.iop"
// // #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon_2020_03_23/nikon_stereographic.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon_2020_03_23/nikon_updated.eop"
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon_2020_03_23/nikonLowWeight_centred.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon_2020_03_23/nikonTruth.xyz" // only use for QC
// #define INPUTROPFILENAME ""

// // // Nikon Training Data
// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon_2020_03_23/TrainingTesting/nikonTraining.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon_2020_03_23/TrainingTesting/nikonTrainingTemp.pho"
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon_2020_03_23/nikon_updated.iop"
// // #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon_2020_03_23/nikon_stereographic.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon_2020_03_23/TrainingTesting/nikonTraining.eop"
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon_2020_03_23/TrainingTesting/nikonTraining.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon_2020_03_23/TrainingTesting/nikonTruthTraining.xyz" // only use for QC
// #define INPUTROPFILENAME ""

// // Nikon Testing Data
// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon_2020_03_23/TrainingTesting/nikonTesting.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon_2020_03_23/TrainingTesting/nikonTestingTemp.pho"
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon_2020_03_23/nikon_updated.iop"
// // #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon_2020_03_23/nikon_stereographic.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon_2020_03_23/TrainingTesting/nikonTesting.eop"
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon_2020_03_23/TrainingTesting/nikonTesting.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/nikon_2020_03_23/TrainingTesting/nikonTruthTesting.xyz" // only use for QC
// #define INPUTROPFILENAME ""

// // //for all goPro
// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/gopro_2020_04_01/gopro_screened.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/omnidirectionalCamera/gopro_2020_04_01/goproTemp.pho"
// // #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/gopro_2020_04_01/gopro.iop"
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/gopro_2020_04_01/gopro_stereographic.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/gopro_2020_04_01/gopro.eop"
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/gopro_2020_04_01/gopro.xyz"
// #define INPUTXYZTRUTHFILENAME "/home/jckchow/BundleAdjustment/omnidirectionalCamera/gopro_2020_04_01/goproTruth.xyz" // only use for QC
// #define INPUTROPFILENAME ""

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// 
/// Functions
/// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// function for calculating the median of vector
double calcMedian(std::vector<double> scores)
{
  double median;
  size_t size = scores.size();

  sort(scores.begin(), scores.end());

  if (size  % 2 == 0)
  {
      median = (scores[size / 2 - 1] + scores[size / 2]) / 2;
  }
  else 
  {
      median = scores[size / 2];
  }

  return median;
}

// function for calculating the standard deviation of vector
double calcStdDev(std::vector<double> v)
{
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();

    std::vector<double> diff(v.size());
    std::transform(v.begin(), v.end(), diff.begin(),
                std::bind2nd(std::minus<double>(), mean));
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / v.size());

    return stdev;
}

// function for randomly selecting a point between min and max
int rangeRandomNumber (int min, int max){
    int n = max - min + 1;
    int remainder = RAND_MAX % n;
    int x;
    do{
        x = rand();
    }while (x >= RAND_MAX - remainder);
    return min + x % n;
}

// Calculate incidence angle relative to optical axis
double incidenceAngle (const double* EOP, const double* XYZ)
{
  // rotation from map to sensor
  double r11 = cos(EOP[1]) * cos(EOP[2]);
  double r12 = cos(EOP[0]) * sin(EOP[2]) + sin(EOP[0]) * sin(EOP[1]) * cos(EOP[2]);
  double r13 = sin(EOP[0]) * sin(EOP[2]) - cos(EOP[0]) * sin(EOP[1]) * cos(EOP[2]);

  double r21 = -cos(EOP[1]) * sin(EOP[2]);
  double r22 = cos(EOP[0]) * cos(EOP[2]) - sin(EOP[0]) * sin(EOP[1]) * sin(EOP[2]);
  double r23 = sin(EOP[0]) * cos(EOP[2]) + cos(EOP[0]) * sin(EOP[1]) * sin(EOP[2]);

  double r31 = sin(EOP[1]);
  double r32 = -sin(EOP[0]) * cos(EOP[1]);
  double r33 = cos(EOP[0]) * cos(EOP[1]);

  // rigid body transformation
  // Object space coordinates oordinates in sensor frame
  double Xs = r11 * ( XYZ[0] - EOP[3] ) + r12 * ( XYZ[1] - EOP[4] ) + r13 * ( XYZ[2] - EOP[5] );
  double Ys = r21 * ( XYZ[0] - EOP[3] ) + r22 * ( XYZ[1] - EOP[4] ) + r23 * ( XYZ[2] - EOP[5] );
  double Zs = r31 * ( XYZ[0] - EOP[3] ) + r32 * ( XYZ[1] - EOP[4] ) + r33 * ( XYZ[2] - EOP[5] );

  return ( atan2(sqrt(Xs*Xs+Ys*Ys) , -Zs) );
}

double refractionAngle(const double x, const double y, const double* IOP, const double* AP)
{
  // camera correction model AP = a1, a2, k1, k2, k3, p1, p2, ...
  double x_bar = (x - IOP[0]) / APSCALE; // arbitrary scale for stability
  double y_bar = (y - IOP[1]) / APSCALE; // arbitrary scale for stability
  double r = sqrt(x_bar*x_bar + y_bar*y_bar); 

  double delta_x = x_bar*(AP[2]*r*r+AP[3]*r*r*r*r+AP[4]*r*r*r*r*r*r) + AP[5]*(r*r+(2.0)*x_bar*x_bar)+(2.0)*AP[6]*x_bar*y_bar + AP[0]*x_bar+AP[1]*y_bar;
  double delta_y = y_bar*(AP[2]*r*r+AP[3]*r*r*r*r+AP[4]*r*r*r*r*r*r) + AP[6]*(r*r+(2.0)*y_bar*y_bar)+(2.0)*AP[5]*x_bar*y_bar;

  double x_corr = x - IOP[0] - delta_x;
  double y_corr = y - IOP[1] - delta_y;

  return ( atan2(sqrt(x_corr*x_corr+y_corr*y_corr) , IOP[2]) );
}

// Calculates the Akaike information criterion
double calculateAIC(double numObs, double mse, double numUnk)
{
	double aic = numObs * log(mse) + 2.0 * numUnk;
	return (aic);
}

// Calculates the Bayesian information criterion
double calculateBIC(double numObs, double mse, double numUnk)
{
	double bic = numObs * log(mse) + numUnk * log(numObs);
	return (bic);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Pseudo observation of a constant
/// Input:    y      - Some constant value
///           weight - 1 / StdDev
/// Unknowns: x      - some unknown parameter in the adjustment
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct constantConstraint {
  
  constantConstraint(double y, double weight)
        : y_(y), weight_(weight)  {}

  template <typename T>
  // unknown parameters followed by the output residual
  bool operator()(const T* const x, T* residual) const {

  residual[0] = x[0] - T(y_);
  residual[0] *= T(weight_);
  return true;
  }

 private:
  // Observations for a sample.
  const double y_;
  const double weight_;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Pseudo observation of XYZ
/// Input:    X       - Some constant value
///           Y
///           Z
///           XStdDev - weight to constrain it
///           YStdDev
///           ZStdDev
/// Unknowns: XYZ     - the object space coordinate
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct constrainPoint {
  
  constrainPoint(double X, double Y, double Z, double XStdDev, double YStdDev, double ZStdDev)
        : X_(X), Y_(Y), Z_(Z), XStdDev_(XStdDev), YStdDev_(YStdDev), ZStdDev_(ZStdDev){}

  template <typename T>
  // unknown parameters followed by the output residual
  bool operator()(const T* const XYZ, T* residual) const {

  residual[0] = XYZ[0] - T(X_);
  residual[1] = XYZ[1] - T(Y_);
  residual[2] = XYZ[2] - T(Z_);
  residual[0] /= T(XStdDev_);
  residual[1] /= T(YStdDev_);
  residual[2] /= T(ZStdDev_);

  return true;
  }

 private:
  // what we want to constraint the point to
  const double X_;
  const double Y_;
  const double Z_;
  const double XStdDev_;
  const double YStdDev_;
  const double ZStdDev_;

};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Pseudo observation of AP
/// Input:    a1       - Some constant value
///           a2
///           k1
///           k2
///           k3
///           p1
///           p2
///           a1StdDev - weight to constrain it
///           a2StdDev
///           k1StdDev
///           k2StdDev
///           k3StdDev
///           p1StdDev
///           p2StdDev
/// Unknowns: AP     - the object space coordinate
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct constrainAP {
  
  constrainAP(double a1, double a2, double k1, double k2, double k3, double p1, double p2, double a1StdDev, double a2StdDev, double k1StdDev, double k2StdDev, double k3StdDev, double p1StdDev, double p2StdDev)
        : a1_(a1), a2_(a2), k1_(k1), k2_(k2), k3_(k3), p1_(p1), p2_(p2), a1StdDev_(a1StdDev), a2StdDev_(a2StdDev), k1StdDev_(k1StdDev), k2StdDev_(k2StdDev), k3StdDev_(k3StdDev), p1StdDev_(p1StdDev), p2StdDev_(p2StdDev){}

  template <typename T>
  // unknown parameters followed by the output residual
  bool operator()(const T* const AP, T* residual) const {

  residual[0] = AP[0] - T(a1_);
  residual[1] = AP[1] - T(a2_);
  residual[2] = AP[2] - T(k1_);
  residual[3] = AP[3] - T(k2_);
  residual[4] = AP[4] - T(k3_);
  residual[5] = AP[5] - T(p1_);
  residual[6] = AP[6] - T(p2_);

  residual[0] /= T(a1StdDev_);
  residual[1] /= T(a2StdDev_);
  residual[2] /= T(k1StdDev_);
  residual[3] /= T(k2StdDev_);
  residual[4] /= T(k3StdDev_);
  residual[5] /= T(p1StdDev_);
  residual[6] /= T(p2StdDev_);

  return true;
  }

 private:
  // what we want to constraint the point to
  const double a1_;
  const double a2_;
  const double k1_;
  const double k2_;
  const double k3_;
  const double p1_;
  const double p2_;

  const double a1StdDev_;
  const double a2StdDev_;
  const double k1StdDev_;
  const double k2StdDev_;
  const double k3StdDev_;
  const double p1StdDev_;
  const double p2StdDev_;

};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Collinearity Equation
/// Input:    x       - x observation
///           y       - y observation
///           xStdDev - noise
///           yStdDev - noise
/// Unknowns: x      - some unknown parameter in the adjustment
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct collinearity {
  
  collinearity(double x, double y, double xStdDev, double yStdDev, double xp, double yp)
        : x_(x), y_(y), xStdDev_(xStdDev), yStdDev_(yStdDev), xp_(xp), yp_(yp)  {}

  template <typename T>
  // unknown parameters followed by the output residual
  bool operator()(const T* const EOP, const T* const XYZ, const T* const IOP, const T* const AP, T* residual) const {

  // rotation from map to sensor
  T r11 = cos(EOP[1]) * cos(EOP[2]);
  T r12 = cos(EOP[0]) * sin(EOP[2]) + sin(EOP[0]) * sin(EOP[1]) * cos(EOP[2]);
  T r13 = sin(EOP[0]) * sin(EOP[2]) - cos(EOP[0]) * sin(EOP[1]) * cos(EOP[2]);

  T r21 = -cos(EOP[1]) * sin(EOP[2]);
  T r22 = cos(EOP[0]) * cos(EOP[2]) - sin(EOP[0]) * sin(EOP[1]) * sin(EOP[2]);
  T r23 = sin(EOP[0]) * cos(EOP[2]) + cos(EOP[0]) * sin(EOP[1]) * sin(EOP[2]);

  T r31 = sin(EOP[1]);
  T r32 = -sin(EOP[0]) * cos(EOP[1]);
  T r33 = cos(EOP[0]) * cos(EOP[1]);

  // rigid body transformation
  T XTemp = r11 * ( XYZ[0] - EOP[3] ) + r12 * ( XYZ[1] - EOP[4] ) + r13 * ( XYZ[2] - EOP[5] );
  T YTemp = r21 * ( XYZ[0] - EOP[3] ) + r22 * ( XYZ[1] - EOP[4] ) + r23 * ( XYZ[2] - EOP[5] );
  T ZTemp = r31 * ( XYZ[0] - EOP[3] ) + r32 * ( XYZ[1] - EOP[4] ) + r33 * ( XYZ[2] - EOP[5] );

//   std::cout<<"XYZ[0], XYZ[1], XYZ[2]: "<<XYZ[0]<<", "<<XYZ[1]<<", "<<XYZ[2]<<std::endl;
//   std::cout<<"EOP[3], EOP[4], EOP[5]: "<<EOP[3]<<", "<<EOP[4]<<", "<<EOP[5]<<std::endl;
//   std::cout<<"XTemp, YTemp, ZTemp: "<<XTemp<<", "<<YTemp<<", "<<ZTemp<<std::endl;

//   sleep(10000000000);        

  // collinearity condition
  T x = -IOP[2] * XTemp / ZTemp;
  T y = -IOP[2] * YTemp / ZTemp;

//   std::cout<<"x, y: "<<x+T(xp_)<<", "<<y+T(yp_)<<std::endl;
//   std::cout<<"x_obs, y_obs: "<<T(x_)<<", "<<T(y_)<<std::endl;


  // camera correction model AP = a1, a2, k1, k2, k3, p1, p2, ...
  T x_bar = (T(x_) - IOP[0]) / APSCALE; // arbitrary scale for stability
  T y_bar = (T(y_) - IOP[1]) / APSCALE; // arbitrary scale for stability
//   T x_bar = (T(x_) - T(xp_)) / APSCALE; // arbitrary scale for stability
//   T y_bar = (T(y_) - T(yp_)) / APSCALE; // arbitrary scale for stability
  T r = sqrt(x_bar*x_bar + y_bar*y_bar); 

//   T delta_x = x_bar*(AP[2]*pow(r,2.0)+AP[3]*pow(r,4.0)+AP[4]*pow(r,6.0)) + AP[5]*(pow(r,2.0)+T(2.0)*pow(x_bar,2.0))+T(2.0)*AP[6]*x_bar*y_bar + AP[0]*x_bar+AP[1]*y_bar;
//   T delta_y = y_bar*(AP[2]*pow(r,2.0)+AP[3]*pow(r,4.0)+AP[4]*pow(r,6.0)) + AP[6]*(pow(r,2.0)+T(2.0)*pow(y_bar,2.0))+T(2.0)*AP[5]*x_bar*y_bar;

  T delta_x = x_bar*(AP[2]*r*r+AP[3]*r*r*r*r+AP[4]*r*r*r*r*r*r) + AP[5]*(r*r+T(2.0)*x_bar*x_bar)+T(2.0)*AP[6]*x_bar*y_bar + AP[0]*x_bar+AP[1]*y_bar;
  T delta_y = y_bar*(AP[2]*r*r+AP[3]*r*r*r*r+AP[4]*r*r*r*r*r*r) + AP[6]*(r*r+T(2.0)*y_bar*y_bar)+T(2.0)*AP[5]*x_bar*y_bar;


  T x_true = x + IOP[0] + delta_x;
  T y_true = y + IOP[1] + delta_y;

  // actual cost function
  residual[0] = x_true - T(x_); // x-residual
  residual[1] = y_true - T(y_); // y-residual    

//   std::cout<<"x_diff, y_diff: "<<residual[0]<<", "<<residual[1]<<std::endl;

  residual[0] /= T(xStdDev_);
  residual[1] /= T(yStdDev_);

  return true;
  }

 private:
  const double x_;
  const double y_;
  const double xStdDev_;
  const double yStdDev_;
  const double xp_;
  const double yp_;

};


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Collinearity Equation with Stereographic Projection
/// Input:    x       - x observation
///           y       - y observation
///           xStdDev - noise
///           yStdDev - noise
/// Unknowns: x      - some unknown parameter in the adjustment
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct collinearityStereographic {
  
  collinearityStereographic(double x, double y, double xStdDev, double yStdDev, double xp, double yp)
        : x_(x), y_(y), xStdDev_(xStdDev), yStdDev_(yStdDev), xp_(xp), yp_(yp)  {}

  template <typename T>
  // unknown parameters followed by the output residual
  bool operator()(const T* const EOP, const T* const XYZ, const T* const IOP, const T* const AP, T* residual) const {

  // rotation from map to sensor
  T r11 = cos(EOP[1]) * cos(EOP[2]);
  T r12 = cos(EOP[0]) * sin(EOP[2]) + sin(EOP[0]) * sin(EOP[1]) * cos(EOP[2]);
  T r13 = sin(EOP[0]) * sin(EOP[2]) - cos(EOP[0]) * sin(EOP[1]) * cos(EOP[2]);

  T r21 = -cos(EOP[1]) * sin(EOP[2]);
  T r22 = cos(EOP[0]) * cos(EOP[2]) - sin(EOP[0]) * sin(EOP[1]) * sin(EOP[2]);
  T r23 = sin(EOP[0]) * cos(EOP[2]) + cos(EOP[0]) * sin(EOP[1]) * sin(EOP[2]);

  T r31 = sin(EOP[1]);
  T r32 = -sin(EOP[0]) * cos(EOP[1]);
  T r33 = cos(EOP[0]) * cos(EOP[1]);

  // rigid body transformation
  // Object space coordinates oordinates in sensor frame
  T Xs = r11 * ( XYZ[0] - EOP[3] ) + r12 * ( XYZ[1] - EOP[4] ) + r13 * ( XYZ[2] - EOP[5] );
  T Ys = r21 * ( XYZ[0] - EOP[3] ) + r22 * ( XYZ[1] - EOP[4] ) + r23 * ( XYZ[2] - EOP[5] );
  T Zs = r31 * ( XYZ[0] - EOP[3] ) + r32 * ( XYZ[1] - EOP[4] ) + r33 * ( XYZ[2] - EOP[5] );


  // Project coordinates onto a circle with radius equal to the focal length
  T d = sqrt(Xs*Xs + Ys*Ys + Zs*Zs);

  // stereographic projection of point on sphere onto image place
  // C = c + radius --> approx it as 2 * c, which is what happens if we assume the image plane is tangent to the sphere
  T x = ( T(2.0)*IOP[2]/(d - Zs) )* Xs;
  T y = ( T(2.0)*IOP[2]/(d - Zs) )* Ys;

//   std::cout<<"x, y: "<<x+T(xp_)<<", "<<y+T(yp_)<<std::endl;
//   std::cout<<"x_obs, y_obs: "<<T(x_)<<", "<<T(y_)<<std::endl;

  // camera correction model AP = a1, a2, k1, k2, k3, p1, p2, ...
  T x_bar = (T(x_) - IOP[0]) / APSCALE; // arbitrary scale for stability
  T y_bar = (T(y_) - IOP[1]) / APSCALE; // arbitrary scale for stability
//   T x_bar = (T(x_) - T(xp_)) / APSCALE; // arbitrary scale for numerical stability
//   T y_bar = (T(y_) - T(yp_)) / APSCALE; // arbitrary scale for numerical stability
  T r = sqrt(x_bar*x_bar + y_bar*y_bar);

//   T delta_x = x_bar*(AP[2]*pow(r,2.0)+AP[3]*pow(r,4.0)+AP[4]*pow(r,6.0)) + AP[5]*(pow(r,2.0)+T(2.0)*pow(x_bar,2.0))+T(2.0)*AP[6]*x_bar*y_bar + AP[0]*x_bar+AP[1]*y_bar;
//   T delta_y = y_bar*(AP[2]*pow(r,2.0)+AP[3]*pow(r,4.0)+AP[4]*pow(r,6.0)) + AP[6]*(pow(r,2.0)+T(2.0)*pow(y_bar,2.0))+T(2.0)*AP[5]*x_bar*y_bar;

  T delta_x = x_bar*(AP[2]*r*r+AP[3]*r*r*r*r+AP[4]*r*r*r*r*r*r) + AP[5]*(r*r+T(2.0)*x_bar*x_bar)+T(2.0)*AP[6]*x_bar*y_bar + AP[0]*x_bar+AP[1]*y_bar;
  T delta_y = y_bar*(AP[2]*r*r+AP[3]*r*r*r*r+AP[4]*r*r*r*r*r*r) + AP[6]*(r*r+T(2.0)*y_bar*y_bar)+T(2.0)*AP[5]*x_bar*y_bar;


  T x_true = x + IOP[0] + delta_x;
  T y_true = y + IOP[1] + delta_y;

  // actual cost function
  residual[0] = x_true - T(x_); // x-residual
  residual[1] = y_true - T(y_); // y-residual    

//   std::cout<<"x_diff, y_diff: "<<residual[0]<<", "<<residual[1]<<std::endl;

  residual[0] /= T(xStdDev_);
  residual[1] /= T(yStdDev_);

  return true;
  }

 private:
  const double x_;
  const double y_;
  const double xStdDev_;
  const double yStdDev_;
  const double xp_;
  const double yp_;

};


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Fish-eye lens camera with Equidistant Projection
/// Input:    x       - x observation
///           y       - y observation
///           xStdDev - noise
///           yStdDev - noise
/// Unknowns: x      - some unknown parameter in the adjustment
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct fisheyeEquidistant {
  
  fisheyeEquidistant(double x, double y, double xStdDev, double yStdDev, double xp, double yp)
        : x_(x), y_(y), xStdDev_(xStdDev), yStdDev_(yStdDev), xp_(xp), yp_(yp)  {}

  template <typename T>
  // unknown parameters followed by the output residual
  bool operator()(const T* const EOP, const T* const XYZ, const T* const IOP, const T* const AP, T* residual) const {

  // rotation from map to sensor
  T r11 = cos(EOP[1]) * cos(EOP[2]);
  T r12 = cos(EOP[0]) * sin(EOP[2]) + sin(EOP[0]) * sin(EOP[1]) * cos(EOP[2]);
  T r13 = sin(EOP[0]) * sin(EOP[2]) - cos(EOP[0]) * sin(EOP[1]) * cos(EOP[2]);

  T r21 = -cos(EOP[1]) * sin(EOP[2]);
  T r22 = cos(EOP[0]) * cos(EOP[2]) - sin(EOP[0]) * sin(EOP[1]) * sin(EOP[2]);
  T r23 = sin(EOP[0]) * cos(EOP[2]) + cos(EOP[0]) * sin(EOP[1]) * sin(EOP[2]);

  T r31 = sin(EOP[1]);
  T r32 = -sin(EOP[0]) * cos(EOP[1]);
  T r33 = cos(EOP[0]) * cos(EOP[1]);

  // rigid body transformation
  // Object space coordinates oordinates in sensor frame
  T Xs = r11 * ( XYZ[0] - EOP[3] ) + r12 * ( XYZ[1] - EOP[4] ) + r13 * ( XYZ[2] - EOP[5] );
  T Ys = r21 * ( XYZ[0] - EOP[3] ) + r22 * ( XYZ[1] - EOP[4] ) + r23 * ( XYZ[2] - EOP[5] );
  T Zs = r31 * ( XYZ[0] - EOP[3] ) + r32 * ( XYZ[1] - EOP[4] ) + r33 * ( XYZ[2] - EOP[5] );

  // ISPRS "Validation of geometric models for fisheye lenses" journal paper
  T x = IOP[2]*Xs*atan2(sqrt(Xs*Xs+Ys*Ys), -Zs) / sqrt(Xs*Xs+Ys*Ys);
  T y = IOP[2]*Ys*atan2(sqrt(Xs*Xs+Ys*Ys), -Zs) / sqrt(Xs*Xs+Ys*Ys);

  // camera correction model AP = a1, a2, k1, k2, k3, p1, p2, ...
  T x_bar = (T(x_) - IOP[0]) / APSCALE; // arbitrary scale for stability
  T y_bar = (T(y_) - IOP[1]) / APSCALE; // arbitrary scale for stability

  T r = sqrt(x_bar*x_bar + y_bar*y_bar);

  T delta_x = x_bar*(AP[2]*r*r+AP[3]*r*r*r*r+AP[4]*r*r*r*r*r*r) + AP[5]*(r*r+T(2.0)*x_bar*x_bar)+T(2.0)*AP[6]*x_bar*y_bar + AP[0]*x_bar+AP[1]*y_bar;
  T delta_y = y_bar*(AP[2]*r*r+AP[3]*r*r*r*r+AP[4]*r*r*r*r*r*r) + AP[6]*(r*r+T(2.0)*y_bar*y_bar)+T(2.0)*AP[5]*x_bar*y_bar;


  T x_true = x + IOP[0] + delta_x;
  T y_true = y + IOP[1] + delta_y;

  // actual cost function
  residual[0] = x_true - T(x_); // x-residual
  residual[1] = y_true - T(y_); // y-residual    

  residual[0] /= T(xStdDev_);
  residual[1] /= T(yStdDev_);

  return true;
  }

 private:
  const double x_;
  const double y_;
  const double xStdDev_;
  const double yStdDev_;
  const double xp_;
  const double yp_;

};



/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Fish-eye lens camera with Stereographic Projection
/// Input:    x       - x observation
///           y       - y observation
///           xStdDev - noise
///           yStdDev - noise
/// Unknowns: x      - some unknown parameter in the adjustment
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct fisheyeStereographic {
  
  fisheyeStereographic(double x, double y, double xStdDev, double yStdDev, double xp, double yp)
        : x_(x), y_(y), xStdDev_(xStdDev), yStdDev_(yStdDev), xp_(xp), yp_(yp)  {}

  template <typename T>
  // unknown parameters followed by the output residual
  bool operator()(const T* const EOP, const T* const XYZ, const T* const IOP, const T* const AP, T* residual) const {

  // rotation from map to sensor
  T r11 = cos(EOP[1]) * cos(EOP[2]);
  T r12 = cos(EOP[0]) * sin(EOP[2]) + sin(EOP[0]) * sin(EOP[1]) * cos(EOP[2]);
  T r13 = sin(EOP[0]) * sin(EOP[2]) - cos(EOP[0]) * sin(EOP[1]) * cos(EOP[2]);

  T r21 = -cos(EOP[1]) * sin(EOP[2]);
  T r22 = cos(EOP[0]) * cos(EOP[2]) - sin(EOP[0]) * sin(EOP[1]) * sin(EOP[2]);
  T r23 = sin(EOP[0]) * cos(EOP[2]) + cos(EOP[0]) * sin(EOP[1]) * sin(EOP[2]);

  T r31 = sin(EOP[1]);
  T r32 = -sin(EOP[0]) * cos(EOP[1]);
  T r33 = cos(EOP[0]) * cos(EOP[1]);

  // rigid body transformation
  // Object space coordinates oordinates in sensor frame
  T Xs = r11 * ( XYZ[0] - EOP[3] ) + r12 * ( XYZ[1] - EOP[4] ) + r13 * ( XYZ[2] - EOP[5] );
  T Ys = r21 * ( XYZ[0] - EOP[3] ) + r22 * ( XYZ[1] - EOP[4] ) + r23 * ( XYZ[2] - EOP[5] );
  T Zs = r31 * ( XYZ[0] - EOP[3] ) + r32 * ( XYZ[1] - EOP[4] ) + r33 * ( XYZ[2] - EOP[5] );

  // ISPRS "Validation of geometric models for fisheye lenses" journal paper
  T x = IOP[2]*Xs*tan( T(0.5)*atan2(sqrt(Xs*Xs+Ys*Ys), -Zs) / sqrt(Xs*Xs+Ys*Ys) );
  T y = IOP[2]*Ys*tan( T(0.5)*atan2(sqrt(Xs*Xs+Ys*Ys), -Zs) / sqrt(Xs*Xs+Ys*Ys) );

  // camera correction model AP = a1, a2, k1, k2, k3, p1, p2, ...
  T x_bar = (T(x_) - IOP[0]) / APSCALE; // arbitrary scale for stability
  T y_bar = (T(y_) - IOP[1]) / APSCALE; // arbitrary scale for stability

  T r = sqrt(x_bar*x_bar + y_bar*y_bar);

  T delta_x = x_bar*(AP[2]*r*r+AP[3]*r*r*r*r+AP[4]*r*r*r*r*r*r) + AP[5]*(r*r+T(2.0)*x_bar*x_bar)+T(2.0)*AP[6]*x_bar*y_bar + AP[0]*x_bar+AP[1]*y_bar;
  T delta_y = y_bar*(AP[2]*r*r+AP[3]*r*r*r*r+AP[4]*r*r*r*r*r*r) + AP[6]*(r*r+T(2.0)*y_bar*y_bar)+T(2.0)*AP[5]*x_bar*y_bar;


  T x_true = x + IOP[0] + delta_x;
  T y_true = y + IOP[1] + delta_y;

  // actual cost function
  residual[0] = x_true - T(x_); // x-residual
  residual[1] = y_true - T(y_); // y-residual    

  residual[0] /= T(xStdDev_);
  residual[1] /= T(yStdDev_);

  return true;
  }

 private:
  const double x_;
  const double y_;
  const double xStdDev_;
  const double yStdDev_;
  const double xp_;
  const double yp_;

};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Collinearity Equation With Machine Learned Parameters as unknowns
/// Input:    x       - x observation
///           y       - y observation
///           xStdDev - noise
///           yStdDev - noise
/// Unknowns: x      - some unknown parameter in the adjustment
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct collinearityMachineLearned {
  
  collinearityMachineLearned(double x, double y, double xStdDev, double yStdDev, double xp, double yp)
        : x_(x), y_(y), xStdDev_(xStdDev), yStdDev_(yStdDev), xp_(xp), yp_(yp)  {}

  template <typename T>
  // unknown parameters followed by the output residual
  bool operator()(const T* const EOP, const T* const XYZ, const T* const IOP, const T* const AP, const T* const MLP, T* residual) const {

  // rotation from map to sensor
  T r11 = cos(EOP[1]) * cos(EOP[2]);
  T r12 = cos(EOP[0]) * sin(EOP[2]) + sin(EOP[0]) * sin(EOP[1]) * cos(EOP[2]);
  T r13 = sin(EOP[0]) * sin(EOP[2]) - cos(EOP[0]) * sin(EOP[1]) * cos(EOP[2]);

  T r21 = -cos(EOP[1]) * sin(EOP[2]);
  T r22 = cos(EOP[0]) * cos(EOP[2]) - sin(EOP[0]) * sin(EOP[1]) * sin(EOP[2]);
  T r23 = sin(EOP[0]) * cos(EOP[2]) + cos(EOP[0]) * sin(EOP[1]) * sin(EOP[2]);

  T r31 = sin(EOP[1]);
  T r32 = -sin(EOP[0]) * cos(EOP[1]);
  T r33 = cos(EOP[0]) * cos(EOP[1]);

  // rigid body transformation
  T XTemp = r11 * ( XYZ[0] - EOP[3] ) + r12 * ( XYZ[1] - EOP[4] ) + r13 * ( XYZ[2] - EOP[5] );
  T YTemp = r21 * ( XYZ[0] - EOP[3] ) + r22 * ( XYZ[1] - EOP[4] ) + r23 * ( XYZ[2] - EOP[5] );
  T ZTemp = r31 * ( XYZ[0] - EOP[3] ) + r32 * ( XYZ[1] - EOP[4] ) + r33 * ( XYZ[2] - EOP[5] );

//   std::cout<<"XYZ[0], XYZ[1], XYZ[2]: "<<XYZ[0]<<", "<<XYZ[1]<<", "<<XYZ[2]<<std::endl;
//   std::cout<<"EOP[3], EOP[4], EOP[5]: "<<EOP[3]<<", "<<EOP[4]<<", "<<EOP[5]<<std::endl;
//   std::cout<<"XTemp, YTemp, ZTemp: "<<XTemp<<", "<<YTemp<<", "<<ZTemp<<std::endl;

//   sleep(10000000000);        

  // collinearity condition
  T x = -IOP[2] * XTemp / ZTemp;
  T y = -IOP[2] * YTemp / ZTemp;

//   std::cout<<"x, y: "<<x+T(xp_)<<", "<<y+T(yp_)<<std::endl;
//   std::cout<<"x_obs, y_obs: "<<T(x_)<<", "<<T(y_)<<std::endl;


  // camera correction model AP = a1, a2, k1, k2, k3, p1, p2, ...
  T x_bar = (T(x_) - IOP[0]) / APSCALE; // arbitrary scale for stability
  T y_bar = (T(y_) - IOP[1]) / APSCALE; // arbitrary scale for stability
//   T x_bar = (T(x_) - T(xp_)) / APSCALE; // arbitrary scale for numerical stability
//   T y_bar = (T(y_) - T(yp_)) / APSCALE; // arbitrary scale for numerical stability
  T r = sqrt(x_bar*x_bar + y_bar*y_bar);

//   T delta_x = x_bar*(AP[2]*pow(r,2.0)+AP[3]*pow(r,4.0)+AP[4]*pow(r,6.0)) + AP[5]*(pow(r,2.0)+T(2.0)*pow(x_bar,2.0))+T(2.0)*AP[6]*x_bar*y_bar + AP[0]*x_bar+AP[1]*y_bar;
//   T delta_y = y_bar*(AP[2]*pow(r,2.0)+AP[3]*pow(r,4.0)+AP[4]*pow(r,6.0)) + AP[6]*(pow(r,2.0)+T(2.0)*pow(y_bar,2.0))+T(2.0)*AP[5]*x_bar*y_bar;

  T delta_x = x_bar*(AP[2]*r*r+AP[3]*r*r*r*r+AP[4]*r*r*r*r*r*r) + AP[5]*(r*r+T(2.0)*x_bar*x_bar)+T(2.0)*AP[6]*x_bar*y_bar + AP[0]*x_bar+AP[1]*y_bar;
  T delta_y = y_bar*(AP[2]*r*r+AP[3]*r*r*r*r+AP[4]*r*r*r*r*r*r) + AP[6]*(r*r+T(2.0)*y_bar*y_bar)+T(2.0)*AP[5]*x_bar*y_bar;


  T x_true = x + IOP[0] + delta_x - MLP[0]; // MLP is the machine learned parameters
  T y_true = y + IOP[1] + delta_y - MLP[1];

  // actual cost function
  residual[0] = x_true - T(x_); // x-residual
  residual[1] = y_true - T(y_); // y-residual    

//   std::cout<<"x_diff, y_diff: "<<residual[0]<<", "<<residual[1]<<std::endl;

  residual[0] /= T(xStdDev_);
  residual[1] /= T(yStdDev_);

  return true;
  }

 private:
  const double x_;
  const double y_;
  const double xStdDev_;
  const double yStdDev_;
  const double xp_;
  const double yp_;

};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Collinearity Equation With Machine Learned Parameters as constants
/// Input:    x       - x observation
///           y       - y observation
///           xStdDev - noise
///           yStdDev - noise
/// Unknowns: x      - some unknown parameter in the adjustment
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct collinearityMachineLearnedSimple {
  
  collinearityMachineLearnedSimple(double x, double y, double xStdDev, double yStdDev, double xp, double yp, double xMLP, double yMLP)
        : x_(x), y_(y), xStdDev_(xStdDev), yStdDev_(yStdDev), xp_(xp), yp_(yp), xMLP_(xMLP), yMLP_(yMLP)  {}

  template <typename T>
  // unknown parameters followed by the output residual
  bool operator()(const T* const EOP, const T* const XYZ, const T* const IOP, const T* const AP, T* residual) const {

  // rotation from map to sensor
  T r11 = cos(EOP[1]) * cos(EOP[2]);
  T r12 = cos(EOP[0]) * sin(EOP[2]) + sin(EOP[0]) * sin(EOP[1]) * cos(EOP[2]);
  T r13 = sin(EOP[0]) * sin(EOP[2]) - cos(EOP[0]) * sin(EOP[1]) * cos(EOP[2]);

  T r21 = -cos(EOP[1]) * sin(EOP[2]);
  T r22 = cos(EOP[0]) * cos(EOP[2]) - sin(EOP[0]) * sin(EOP[1]) * sin(EOP[2]);
  T r23 = sin(EOP[0]) * cos(EOP[2]) + cos(EOP[0]) * sin(EOP[1]) * sin(EOP[2]);

  T r31 = sin(EOP[1]);
  T r32 = -sin(EOP[0]) * cos(EOP[1]);
  T r33 = cos(EOP[0]) * cos(EOP[1]);

  // rigid body transformation
  T XTemp = r11 * ( XYZ[0] - EOP[3] ) + r12 * ( XYZ[1] - EOP[4] ) + r13 * ( XYZ[2] - EOP[5] );
  T YTemp = r21 * ( XYZ[0] - EOP[3] ) + r22 * ( XYZ[1] - EOP[4] ) + r23 * ( XYZ[2] - EOP[5] );
  T ZTemp = r31 * ( XYZ[0] - EOP[3] ) + r32 * ( XYZ[1] - EOP[4] ) + r33 * ( XYZ[2] - EOP[5] );

//   std::cout<<"XYZ[0], XYZ[1], XYZ[2]: "<<XYZ[0]<<", "<<XYZ[1]<<", "<<XYZ[2]<<std::endl;
//   std::cout<<"EOP[3], EOP[4], EOP[5]: "<<EOP[3]<<", "<<EOP[4]<<", "<<EOP[5]<<std::endl;
//   std::cout<<"XTemp, YTemp, ZTemp: "<<XTemp<<", "<<YTemp<<", "<<ZTemp<<std::endl;

//   sleep(10000000000);        

  // collinearity condition
  T x = -IOP[2] * XTemp / ZTemp;
  T y = -IOP[2] * YTemp / ZTemp;

//   std::cout<<"x, y: "<<x+xp_<<", "<<y+yp_<<std::endl;
//   std::cout<<"x_obs, y_obs: "<<x_<<", "<<y_<<std::endl;


  // camera correction model AP = a1, a2, k1, k2, k3, p1, p2, ...
  T x_bar = (T(x_) - IOP[0]) / APSCALE; // arbitrary scale for stability
  T y_bar = (T(y_) - IOP[1]) / APSCALE; // arbitrary scale for stability
//   T x_bar = (T(x_) - T(xp_)) / APSCALE; // arbitrary scale for numerical stability
//   T y_bar = (T(y_) - T(yp_)) / APSCALE; // arbitrary scale for numerical stability
  T r = sqrt(x_bar*x_bar + y_bar*y_bar);

//   T delta_x = x_bar*(AP[2]*pow(r,2.0)+AP[3]*pow(r,4.0)+AP[4]*pow(r,6.0)) + AP[5]*(pow(r,2.0)+T(2.0)*pow(x_bar,2.0))+T(2.0)*AP[6]*x_bar*y_bar + AP[0]*x_bar+AP[1]*y_bar;
//   T delta_y = y_bar*(AP[2]*pow(r,2.0)+AP[3]*pow(r,4.0)+AP[4]*pow(r,6.0)) + AP[6]*(pow(r,2.0)+T(2.0)*pow(y_bar,2.0))+T(2.0)*AP[5]*x_bar*y_bar;

  T delta_x = x_bar*(AP[2]*r*r+AP[3]*r*r*r*r+AP[4]*r*r*r*r*r*r) + AP[5]*(r*r+T(2.0)*x_bar*x_bar)+T(2.0)*AP[6]*x_bar*y_bar + AP[0]*x_bar+AP[1]*y_bar;
  T delta_y = y_bar*(AP[2]*r*r+AP[3]*r*r*r*r+AP[4]*r*r*r*r*r*r) + AP[6]*(r*r+T(2.0)*y_bar*y_bar)+T(2.0)*AP[5]*x_bar*y_bar;


  T x_true = x + IOP[0] + delta_x - T(xMLP_); // MLP is the machine learned parameters
  T y_true = y + IOP[1] + delta_y - T(yMLP_);

  // actual cost function
  residual[0] = x_true - T(x_); // x-residual
  residual[1] = y_true - T(y_); // y-residual    

//   std::cout<<"x_diff, y_diff: "<<residual[0]<<", "<<residual[1]<<std::endl;

  residual[0] /= T(xStdDev_);
  residual[1] /= T(yStdDev_);

  return true;
  }

 private:
  const double x_;
  const double y_;
  const double xStdDev_;
  const double yStdDev_;
  const double xp_;
  const double yp_;
  const double xMLP_;
  const double yMLP_;
};


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Stereographic projection collinearity Equation With Machine Learned Parameters as constants
/// Input:    x       - x observation
///           y       - y observation
///           xStdDev - noise
///           yStdDev - noise
///           xp      - principal point location (not just offset)
///           yp      - principal point location (not just offset)
/// Unknowns: x       - some unknown parameter in the adjustment
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct collinearityStereographicMachineLearnedSimple {
  
  collinearityStereographicMachineLearnedSimple(double x, double y, double xStdDev, double yStdDev, double xp, double yp, double xMLP, double yMLP)
        : x_(x), y_(y), xStdDev_(xStdDev), yStdDev_(yStdDev), xp_(xp), yp_(yp), xMLP_(xMLP), yMLP_(yMLP)  {}

  template <typename T>
  // unknown parameters followed by the output residual
  bool operator()(const T* const EOP, const T* const XYZ, const T* const IOP, const T* const AP, T* residual) const {

  // rotation from map to sensor
  T r11 = cos(EOP[1]) * cos(EOP[2]);
  T r12 = cos(EOP[0]) * sin(EOP[2]) + sin(EOP[0]) * sin(EOP[1]) * cos(EOP[2]);
  T r13 = sin(EOP[0]) * sin(EOP[2]) - cos(EOP[0]) * sin(EOP[1]) * cos(EOP[2]);

  T r21 = -cos(EOP[1]) * sin(EOP[2]);
  T r22 = cos(EOP[0]) * cos(EOP[2]) - sin(EOP[0]) * sin(EOP[1]) * sin(EOP[2]);
  T r23 = sin(EOP[0]) * cos(EOP[2]) + cos(EOP[0]) * sin(EOP[1]) * sin(EOP[2]);

  T r31 = sin(EOP[1]);
  T r32 = -sin(EOP[0]) * cos(EOP[1]);
  T r33 = cos(EOP[0]) * cos(EOP[1]);

  // rigid body transformation to get XYZ in sensor frame
  T Xs = r11 * ( XYZ[0] - EOP[3] ) + r12 * ( XYZ[1] - EOP[4] ) + r13 * ( XYZ[2] - EOP[5] );
  T Ys = r21 * ( XYZ[0] - EOP[3] ) + r22 * ( XYZ[1] - EOP[4] ) + r23 * ( XYZ[2] - EOP[5] );
  T Zs = r31 * ( XYZ[0] - EOP[3] ) + r32 * ( XYZ[1] - EOP[4] ) + r33 * ( XYZ[2] - EOP[5] );

  // Project coordinates onto a circle with same radius. as radius approaches zero, we get the conventional collinearity equations back
  T d = sqrt(Xs*Xs + Ys*Ys + Zs*Zs);

  // stereographic projection of point on sphere onto image place
  // C = 2*c if we assume image plane is tangent to the circle
  T x = (T(2.0)*IOP[2])/(d - Zs) * Xs;
  T y = (T(2.0)*IOP[2])/(d - Zs) * Ys;

  // camera correction model AP = a1, a2, k1, k2, k3, p1, p2, ...
  T x_bar = (T(x_) - IOP[0]) / APSCALE; // arbitrary scale for stability
  T y_bar = (T(y_) - IOP[1]) / APSCALE; // arbitrary scale for stability
//   T x_bar = (T(x_) - T(xp_)) / APSCALE; // arbitrary scale for numerical stability
//   T y_bar = (T(y_) - T(yp_)) / APSCALE; // arbitrary scale for numerical stability
  T r = sqrt(x_bar*x_bar + y_bar*y_bar);

//   T delta_x = x_bar*(AP[2]*pow(r,2.0)+AP[3]*pow(r,4.0)+AP[4]*pow(r,6.0)) + AP[5]*(pow(r,2.0)+T(2.0)*pow(x_bar,2.0))+T(2.0)*AP[6]*x_bar*y_bar + AP[0]*x_bar+AP[1]*y_bar;
//   T delta_y = y_bar*(AP[2]*pow(r,2.0)+AP[3]*pow(r,4.0)+AP[4]*pow(r,6.0)) + AP[6]*(pow(r,2.0)+T(2.0)*pow(y_bar,2.0))+T(2.0)*AP[5]*x_bar*y_bar;

  T delta_x = x_bar*(AP[2]*r*r+AP[3]*r*r*r*r+AP[4]*r*r*r*r*r*r) + AP[5]*(r*r+T(2.0)*x_bar*x_bar)+T(2.0)*AP[6]*x_bar*y_bar + AP[0]*x_bar+AP[1]*y_bar;
  T delta_y = y_bar*(AP[2]*r*r+AP[3]*r*r*r*r+AP[4]*r*r*r*r*r*r) + AP[6]*(r*r+T(2.0)*y_bar*y_bar)+T(2.0)*AP[5]*x_bar*y_bar;


  T x_true = x + IOP[0] + delta_x - T(xMLP_); // MLP is the machine learned parameters
  T y_true = y + IOP[1] + delta_y - T(yMLP_);

  // actual cost function
  residual[0] = x_true - T(x_); // x-residual
  residual[1] = y_true - T(y_); // y-residual    

//   std::cout<<"x_diff, y_diff: "<<residual[0]<<", "<<residual[1]<<std::endl;

  residual[0] /= T(xStdDev_);
  residual[1] /= T(yStdDev_);

  return true;
  }

 private:
  const double x_;
  const double y_;
  const double xStdDev_;
  const double yStdDev_;
  const double xp_;
  const double yp_;
  const double xMLP_;
  const double yMLP_;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Collinearity Equation With Machine Learned Parameters as constants and ROP
/// Input:    x       - x observation
///           y       - y observation
///           xStdDev - noise
///           yStdDev - noise
/// Unknowns: x      - some unknown parameter in the adjustment
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct collinearityMachineLearnedROP {
  
  collinearityMachineLearnedROP(double x, double y, double xStdDev, double yStdDev, double xp, double yp, double xMLP, double yMLP)
        : x_(x), y_(y), xStdDev_(xStdDev), yStdDev_(yStdDev), xp_(xp), yp_(yp), xMLP_(xMLP), yMLP_(yMLP)  {}

  template <typename T>
  // unknown parameters followed by the output residual
  bool operator()(const T* const EOP, const T* const ROP, const T* const XYZ, const T* const IOP, const T* const AP, T* residual) const {

  // rotation from map to sensor1
  T r11 = cos(EOP[1]) * cos(EOP[2]);
  T r12 = cos(EOP[0]) * sin(EOP[2]) + sin(EOP[0]) * sin(EOP[1]) * cos(EOP[2]);
  T r13 = sin(EOP[0]) * sin(EOP[2]) - cos(EOP[0]) * sin(EOP[1]) * cos(EOP[2]);

  T r21 = -cos(EOP[1]) * sin(EOP[2]);
  T r22 = cos(EOP[0]) * cos(EOP[2]) - sin(EOP[0]) * sin(EOP[1]) * sin(EOP[2]);
  T r23 = sin(EOP[0]) * cos(EOP[2]) + cos(EOP[0]) * sin(EOP[1]) * sin(EOP[2]);

  T r31 = sin(EOP[1]);
  T r32 = -sin(EOP[0]) * cos(EOP[1]);
  T r33 = cos(EOP[0]) * cos(EOP[1]);

  // rotation from sensor 1 to sensor 2
  T m11 = cos(ROP[1]) * cos(ROP[2]);
  T m12 = cos(ROP[0]) * sin(ROP[2]) + sin(ROP[0]) * sin(ROP[1]) * cos(ROP[2]);
  T m13 = sin(ROP[0]) * sin(ROP[2]) - cos(ROP[0]) * sin(ROP[1]) * cos(ROP[2]);

  T m21 = -cos(ROP[1]) * sin(ROP[2]);
  T m22 = cos(ROP[0]) * cos(ROP[2]) - sin(ROP[0]) * sin(ROP[1]) * sin(ROP[2]);
  T m23 = sin(ROP[0]) * cos(ROP[2]) + cos(ROP[0]) * sin(ROP[1]) * sin(ROP[2]);

  T m31 = sin(ROP[1]);
  T m32 = -sin(ROP[0]) * cos(ROP[1]);
  T m33 = cos(ROP[0]) * cos(ROP[1]); 

  T a11 = m11*r11 + m12*r21 + m13*r31;
  T a12 = m11*r12 + m12*r22 + m13*r32;
  T a13 = m11*r13 + m12*r23 + m13*r33;

  T a21 = m21*r11 + m22*r21 + m23*r31;
  T a22 = m21*r12 + m22*r22 + m23*r32;
  T a23 = m21*r13 + m22*r23 + m23*r33;

  T a31 = m31*r11 + m32*r21 + m33*r31;
  T a32 = m31*r12 + m32*r22 + m33*r32;
  T a33 = m31*r13 + m32*r23 + m33*r33;

  T hx = r11*ROP[3] + r21*ROP[4] + r31*ROP[5];
  T hy = r12*ROP[3] + r22*ROP[4] + r32*ROP[5];
  T hz = r13*ROP[3] + r23*ROP[4] + r33*ROP[5];


  // rigid body transformation
  T XTemp = a11 * ( XYZ[0] - EOP[3] - hx ) + a12 * ( XYZ[1] - EOP[4] - hy ) + a13 * ( XYZ[2] - EOP[5] - hz );
  T YTemp = a21 * ( XYZ[0] - EOP[3] - hx ) + a22 * ( XYZ[1] - EOP[4] - hy ) + a23 * ( XYZ[2] - EOP[5] - hz );
  T ZTemp = a31 * ( XYZ[0] - EOP[3] - hx ) + a32 * ( XYZ[1] - EOP[4] - hy ) + a33 * ( XYZ[2] - EOP[5] - hz );

//   std::cout<<"XYZ[0], XYZ[1], XYZ[2]: "<<XYZ[0]<<", "<<XYZ[1]<<", "<<XYZ[2]<<std::endl;
//   std::cout<<"EOP[3], EOP[4], EOP[5]: "<<EOP[3]<<", "<<EOP[4]<<", "<<EOP[5]<<std::endl;
//   std::cout<<"XTemp, YTemp, ZTemp: "<<XTemp<<", "<<YTemp<<", "<<ZTemp<<std::endl;

//   sleep(10000000000);        

  // collinearity condition
  T x = -IOP[2] * XTemp / ZTemp;
  T y = -IOP[2] * YTemp / ZTemp;

//   std::cout<<"x, y: "<<x+T(xp_)<<", "<<y+T(yp_)<<std::endl;
//   std::cout<<"x_obs, y_obs: "<<T(x_)<<", "<<T(y_)<<std::endl;


  // camera correction model AP = a1, a2, k1, k2, k3, p1, p2, ...
  T x_bar = (T(x_) - IOP[0]) / APSCALE; // arbitrary scale for stability
  T y_bar = (T(y_) - IOP[1]) / APSCALE; // arbitrary scale for stability
//   T x_bar = (T(x_) - T(xp_)) / APSCALE; // arbitrary scale for numerical stability
//   T y_bar = (T(y_) - T(yp_)) / APSCALE; // arbitrary scale for numerical stability
  T r = sqrt(x_bar*x_bar + y_bar*y_bar);

//   T delta_x = x_bar*(AP[2]*pow(r,2.0)+AP[3]*pow(r,4.0)+AP[4]*pow(r,6.0)) + AP[5]*(pow(r,2.0)+T(2.0)*pow(x_bar,2.0))+T(2.0)*AP[6]*x_bar*y_bar + AP[0]*x_bar+AP[1]*y_bar;
//   T delta_y = y_bar*(AP[2]*pow(r,2.0)+AP[3]*pow(r,4.0)+AP[4]*pow(r,6.0)) + AP[6]*(pow(r,2.0)+T(2.0)*pow(y_bar,2.0))+T(2.0)*AP[5]*x_bar*y_bar;

  T delta_x = x_bar*(AP[2]*r*r+AP[3]*r*r*r*r+AP[4]*r*r*r*r*r*r) + AP[5]*(r*r+T(2.0)*x_bar*x_bar)+T(2.0)*AP[6]*x_bar*y_bar + AP[0]*x_bar+AP[1]*y_bar;
  T delta_y = y_bar*(AP[2]*r*r+AP[3]*r*r*r*r+AP[4]*r*r*r*r*r*r) + AP[6]*(r*r+T(2.0)*y_bar*y_bar)+T(2.0)*AP[5]*x_bar*y_bar;


  T x_true = x + IOP[0] + delta_x - T(xMLP_); // MLP is the machine learned parameters
  T y_true = y + IOP[1] + delta_y - T(yMLP_);

  // actual cost function
  residual[0] = x_true - T(x_); // x-residual
  residual[1] = y_true - T(y_); // y-residual    

//   std::cout<<"x_diff, y_diff: "<<residual[0]<<", "<<residual[1]<<std::endl;

  residual[0] /= T(xStdDev_);
  residual[1] /= T(yStdDev_);

  return true;
  }

 private:
  const double x_;
  const double y_;
  const double xStdDev_;
  const double yStdDev_;
  const double xp_;
  const double yp_;
  const double xMLP_;
  const double yMLP_;
};


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Modified Omnidirectional Collinearity Equation With Machine Learned Parameters as constants
/// Input:    x       - x observation
///           y       - y observation
///           xStdDev - noise
///           yStdDev - noise
/// Unknowns: x      - some unknown parameter in the adjustment
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct omniCollinearityMachineLearnedSimple {
  
  omniCollinearityMachineLearnedSimple(double x, double y, double xStdDev, double yStdDev, double xp, double yp, double xMLP, double yMLP)
        : x_(x), y_(y), xStdDev_(xStdDev), yStdDev_(yStdDev), xp_(xp), yp_(yp), xMLP_(xMLP), yMLP_(yMLP)  {}

  template <typename T>
  // unknown parameters followed by the output residual
  bool operator()(const T* const EOP, const T* const XYZ, const T* const IOP, const T* const AP, T* residual) const {

  // rotation from map to sensor
  T r11 = cos(EOP[1]) * cos(EOP[2]);
  T r12 = cos(EOP[0]) * sin(EOP[2]) + sin(EOP[0]) * sin(EOP[1]) * cos(EOP[2]);
  T r13 = sin(EOP[0]) * sin(EOP[2]) - cos(EOP[0]) * sin(EOP[1]) * cos(EOP[2]);

  T r21 = -cos(EOP[1]) * sin(EOP[2]);
  T r22 = cos(EOP[0]) * cos(EOP[2]) - sin(EOP[0]) * sin(EOP[1]) * sin(EOP[2]);
  T r23 = sin(EOP[0]) * cos(EOP[2]) + cos(EOP[0]) * sin(EOP[1]) * sin(EOP[2]);

  T r31 = sin(EOP[1]);
  T r32 = -sin(EOP[0]) * cos(EOP[1]);
  T r33 = cos(EOP[0]) * cos(EOP[1]);

  // rigid body transformation
  T XTemp = r11 * ( XYZ[0] - EOP[3] ) + r12 * ( XYZ[1] - EOP[4] ) + r13 * ( XYZ[2] - EOP[5] );
  T YTemp = r21 * ( XYZ[0] - EOP[3] ) + r22 * ( XYZ[1] - EOP[4] ) + r23 * ( XYZ[2] - EOP[5] );
  T ZTemp = r31 * ( XYZ[0] - EOP[3] ) + r32 * ( XYZ[1] - EOP[4] ) + r33 * ( XYZ[2] - EOP[5] );

//   std::cout<<"XYZ[0], XYZ[1], XYZ[2]: "<<XYZ[0]<<", "<<XYZ[1]<<", "<<XYZ[2]<<std::endl;
//   std::cout<<"EOP[3], EOP[4], EOP[5]: "<<EOP[3]<<", "<<EOP[4]<<", "<<EOP[5]<<std::endl;
//   std::cout<<"XTemp, YTemp, ZTemp: "<<XTemp<<", "<<YTemp<<", "<<ZTemp<<std::endl;

//   sleep(10000000000);        

  // modified omnidirectional collinearity condition dividing x and y
//   T x = IOP[2] * atan2(XTemp , -ZTemp);
//   T y = IOP[2] * atan2(YTemp , -ZTemp);
// modified omnidirectional collinearity condition using spatial angle
  T x = IOP[2] * atan2(sqrt(XTemp*XTemp+YTemp*YTemp),-ZTemp) / sqrt((YTemp/XTemp)*(YTemp/XTemp) + 1.0);
  T y = IOP[2] * atan2(sqrt(XTemp*XTemp+YTemp*YTemp),-ZTemp) / sqrt((XTemp/YTemp)*(XTemp/YTemp) + 1.0);

//   std::cout<<"x, y: "<<x+xp_<<", "<<y+yp_<<std::endl;
//   std::cout<<"x_obs, y_obs: "<<x_<<", "<<y_<<std::endl;


  // camera correction model AP = a1, a2, k1, k2, k3, p1, p2, ...
  T x_bar = (T(x_) - IOP[0]) / APSCALE; // arbitrary scale for stability
  T y_bar = (T(y_) - IOP[1]) / APSCALE; // arbitrary scale for stability
//   T x_bar = (T(x_) - T(xp_)) / APSCALE; // arbitrary scale for numerical stability
//   T y_bar = (T(y_) - T(yp_)) / APSCALE; // arbitrary scale for numerical stability
  T r = sqrt(x_bar*x_bar + y_bar*y_bar);

//   T delta_x = x_bar*(AP[2]*pow(r,2.0)+AP[3]*pow(r,4.0)+AP[4]*pow(r,6.0)) + AP[5]*(pow(r,2.0)+T(2.0)*pow(x_bar,2.0))+T(2.0)*AP[6]*x_bar*y_bar + AP[0]*x_bar+AP[1]*y_bar;
//   T delta_y = y_bar*(AP[2]*pow(r,2.0)+AP[3]*pow(r,4.0)+AP[4]*pow(r,6.0)) + AP[6]*(pow(r,2.0)+T(2.0)*pow(y_bar,2.0))+T(2.0)*AP[5]*x_bar*y_bar;

  T delta_x = x_bar*(AP[2]*r*r+AP[3]*r*r*r*r+AP[4]*r*r*r*r*r*r) + AP[5]*(r*r+T(2.0)*x_bar*x_bar)+T(2.0)*AP[6]*x_bar*y_bar + AP[0]*x_bar+AP[1]*y_bar;
  T delta_y = y_bar*(AP[2]*r*r+AP[3]*r*r*r*r+AP[4]*r*r*r*r*r*r) + AP[6]*(r*r+T(2.0)*y_bar*y_bar)+T(2.0)*AP[5]*x_bar*y_bar;


  T x_true = x + IOP[0] + delta_x - T(xMLP_); // MLP is the machine learned parameters
  T y_true = y + IOP[1] + delta_y - T(yMLP_);

  // actual cost function
  residual[0] = x_true - T(x_); // x-residual
  residual[1] = y_true - T(y_); // y-residual    

//   std::cout<<"x_diff, y_diff: "<<residual[0]<<", "<<residual[1]<<std::endl;

  residual[0] /= T(xStdDev_);
  residual[1] /= T(yStdDev_);

  return true;
  }

 private:
  const double x_;
  const double y_;
  const double xStdDev_;
  const double yStdDev_;
  const double xp_;
  const double yp_;
  const double xMLP_;
  const double yMLP_;
};


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// ROP constraint
/// Input:    omegaStdDev
///           phiStdDev
///           kappaStdDev
///           XoStdDev
///           YoStdDev
///           ZoStdDev
/// Unknowns: EOP1     - EOP of master
///           EOP2     - EOP of slave
///           ROP      - boresight angles followed by the leverarm
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct ropConstraint {
  
  ropConstraint(double omegaStdDev, double phiStdDev, double kappaStdDev, double XoStdDev, double YoStdDev, double ZoStdDev)
        : omegaStdDev_(omegaStdDev), phiStdDev_(phiStdDev), kappaStdDev_(kappaStdDev), XoStdDev_(XoStdDev), YoStdDev_(YoStdDev), ZoStdDev_(ZoStdDev) {}

  template <typename T>
  // unknown parameters followed by the output residual
  bool operator()(const T* const EOP1, const T* const EOP2, const T* const ROP, T* residual) const {

  // rotation from map to sensor 1
  T a11 = cos(EOP1[1]) * cos(EOP1[2]);
  T a12 = cos(EOP1[0]) * sin(EOP1[2]) + sin(EOP1[0]) * sin(EOP1[1]) * cos(EOP1[2]);
  T a13 = sin(EOP1[0]) * sin(EOP1[2]) - cos(EOP1[0]) * sin(EOP1[1]) * cos(EOP1[2]);

  T a21 = -cos(EOP1[1]) * sin(EOP1[2]);
  T a22 = cos(EOP1[0]) * cos(EOP1[2]) - sin(EOP1[0]) * sin(EOP1[1]) * sin(EOP1[2]);
  T a23 = sin(EOP1[0]) * cos(EOP1[2]) + cos(EOP1[0]) * sin(EOP1[1]) * sin(EOP1[2]);

  T a31 = sin(EOP1[1]);
  T a32 = -sin(EOP1[0]) * cos(EOP1[1]);
  T a33 = cos(EOP1[0]) * cos(EOP1[1]); 

  // rotation from map to sensor 2
  T b11 = cos(EOP2[1]) * cos(EOP2[2]);
  T b12 = cos(EOP2[0]) * sin(EOP2[2]) + sin(EOP2[0]) * sin(EOP2[1]) * cos(EOP2[2]);
  T b13 = sin(EOP2[0]) * sin(EOP2[2]) - cos(EOP2[0]) * sin(EOP2[1]) * cos(EOP2[2]);

  T b21 = -cos(EOP2[1]) * sin(EOP2[2]);
  T b22 = cos(EOP2[0]) * cos(EOP2[2]) - sin(EOP2[0]) * sin(EOP2[1]) * sin(EOP2[2]);
  T b23 = sin(EOP2[0]) * cos(EOP2[2]) + cos(EOP2[0]) * sin(EOP2[1]) * sin(EOP2[2]);

  T b31 = sin(EOP2[1]);
  T b32 = -sin(EOP2[0]) * cos(EOP2[1]);
  T b33 = cos(EOP2[0]) * cos(EOP2[1]); 

  // rotation from sensor 2 to sensor 1
  T r11 = cos(ROP[1]) * cos(ROP[2]);
  T r21 = cos(ROP[0]) * sin(ROP[2]) + sin(ROP[0]) * sin(ROP[1]) * cos(ROP[2]);
  T r31 = sin(ROP[0]) * sin(ROP[2]) - cos(ROP[0]) * sin(ROP[1]) * cos(ROP[2]);

  T r12 = -cos(ROP[1]) * sin(ROP[2]);
  T r22 = cos(ROP[0]) * cos(ROP[2]) - sin(ROP[0]) * sin(ROP[1]) * sin(ROP[2]);
  T r32 = sin(ROP[0]) * cos(ROP[2]) + cos(ROP[0]) * sin(ROP[1]) * sin(ROP[2]);

  T r13 = sin(ROP[1]);
  T r23 = -sin(ROP[0]) * cos(ROP[1]);
  T r33 = cos(ROP[0]) * cos(ROP[1]); 

  // R_1To2 = R_mTo2 * R_1Tom 
  T m11 = b11 * a11 + b12 * a12 + b13 * a13;
  T m12 = b11 * a21 + b12 * a22 + b13 * a23;
  T m13 = b11 * a31 + b12 * a32 + b13 * a33;

  T m21 = b21 * a11 + b22 * a12 + b23 * a13;
  T m22 = b21 * a21 + b22 * a22 + b23 * a23;
  T m23 = b21 * a31 + b22 * a32 + b23 * a33;

  T m31 = b31 * a11 + b32 * a12 + b33 * a13;
  T m32 = b31 * a21 + b32 * a22 + b33 * a23;
  T m33 = b31 * a31 + b32 * a32 + b33 * a33;

    T Tx = EOP2[3] - EOP1[3];
    T Ty = EOP2[4] - EOP1[4];
    T Tz = EOP2[5] - EOP1[5];

    // I = boresight_2To1 * R_1To2
    T deltaR32 = r31*m12 + r32*m22 + r33*m32;
    T deltaR33 = r31*m13 + r32*m23 + r33*m33;
    T deltaR31 = r31*m11 + r32*m21 + r33*m31;
    T deltaR21 = r21*m11 + r22*m21 + r23*m31;
    T deltaR11 = r11*m11 + r12*m21 + r13*m31;

    // T deltaR22 = r21*m12 + r22*m22 + r23*m32;

    T deltaOmega = atan2(-deltaR32, deltaR33);
    T deltaPhi   = asin(deltaR31);
    T deltaKappa = atan2(-deltaR21, deltaR11);

    // defined in the coordinate frame of sensor 1
    T bx = a11*Tx + a12*Ty + a13*Tz;
    T by = a21*Tx + a22*Ty + a23*Tz;
    T bz = a31*Tx + a32*Ty + a33*Tz;

  // actual cost function
  residual[0] = deltaOmega; // delta omega
  residual[1] = deltaPhi; // delta phi 
  residual[2] = deltaKappa; // delta kappa 
//   residual[0] = deltaR11 - 1.0;
//   residual[1] = deltaR22 - 1.0;
//   residual[2] = deltaR33 - 1.0;
  residual[3] = bx - ROP[3]; // delta Xo 
  residual[4] = by - ROP[4]; // delta Yo 
  residual[5] = bz - ROP[5]; // delta Zo 

  residual[0] /= T(omegaStdDev_);
  residual[1] /= T(phiStdDev_);
  residual[2] /= T(kappaStdDev_);
  residual[3] /= T(XoStdDev_);
  residual[4] /= T(YoStdDev_);
  residual[5] /= T(ZoStdDev_);


  return true;
  }

 private:
  const double omegaStdDev_;
  const double phiStdDev_;
  const double kappaStdDev_;
  const double XoStdDev_;
  const double YoStdDev_;
  const double ZoStdDev_;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// ROP constraint
/// Input:    omegaStdDev
///           phiStdDev
///           kappaStdDev
///           XoStdDev
///           YoStdDev
///           ZoStdDev
/// Unknowns: EOP1     - EOP of master
///           EOP2     - EOP of slave
///           ROP      - boresight angles followed by the leverarm
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct ropConstraintVector {
  
  ropConstraintVector(double XoStdDev, double YoStdDev, double ZoStdDev)
        : XoStdDev_(XoStdDev), YoStdDev_(YoStdDev), ZoStdDev_(ZoStdDev) {}

  template <typename T>
  // unknown parameters followed by the output residual
  bool operator()(const T* const EOP1, const T* const EOP2, const T* const ROP, T* residual) const {

  // rotation from map to sensor 1
  T a11 = cos(EOP1[1]) * cos(EOP1[2]);
  T a12 = cos(EOP1[0]) * sin(EOP1[2]) + sin(EOP1[0]) * sin(EOP1[1]) * cos(EOP1[2]);
  T a13 = sin(EOP1[0]) * sin(EOP1[2]) - cos(EOP1[0]) * sin(EOP1[1]) * cos(EOP1[2]);

  T a21 = -cos(EOP1[1]) * sin(EOP1[2]);
  T a22 = cos(EOP1[0]) * cos(EOP1[2]) - sin(EOP1[0]) * sin(EOP1[1]) * sin(EOP1[2]);
  T a23 = sin(EOP1[0]) * cos(EOP1[2]) + cos(EOP1[0]) * sin(EOP1[1]) * sin(EOP1[2]);

  T a31 = sin(EOP1[1]);
  T a32 = -sin(EOP1[0]) * cos(EOP1[1]);
  T a33 = cos(EOP1[0]) * cos(EOP1[1]); 

  // rotation from map to sensor 2
  T b11 = cos(EOP2[1]) * cos(EOP2[2]);
  T b12 = cos(EOP2[0]) * sin(EOP2[2]) + sin(EOP2[0]) * sin(EOP2[1]) * cos(EOP2[2]);
  T b13 = sin(EOP2[0]) * sin(EOP2[2]) - cos(EOP2[0]) * sin(EOP2[1]) * cos(EOP2[2]);

  T b21 = -cos(EOP2[1]) * sin(EOP2[2]);
  T b22 = cos(EOP2[0]) * cos(EOP2[2]) - sin(EOP2[0]) * sin(EOP2[1]) * sin(EOP2[2]);
  T b23 = sin(EOP2[0]) * cos(EOP2[2]) + cos(EOP2[0]) * sin(EOP2[1]) * sin(EOP2[2]);

  T b31 = sin(EOP2[1]);
  T b32 = -sin(EOP2[0]) * cos(EOP2[1]);
  T b33 = cos(EOP2[0]) * cos(EOP2[1]); 

  // rotation from sensor 2 to sensor 1
  T r11 = cos(ROP[1]) * cos(ROP[2]);
  T r12 = cos(ROP[0]) * sin(ROP[2]) + sin(ROP[0]) * sin(ROP[1]) * cos(ROP[2]);
  T r13 = sin(ROP[0]) * sin(ROP[2]) - cos(ROP[0]) * sin(ROP[1]) * cos(ROP[2]);

  T r21 = -cos(ROP[1]) * sin(ROP[2]);
  T r22 = cos(ROP[0]) * cos(ROP[2]) - sin(ROP[0]) * sin(ROP[1]) * sin(ROP[2]);
  T r23 = sin(ROP[0]) * cos(ROP[2]) + cos(ROP[0]) * sin(ROP[1]) * sin(ROP[2]);

  T r31 = sin(ROP[1]);
  T r32 = -sin(ROP[0]) * cos(ROP[1]);
  T r33 = cos(ROP[0]) * cos(ROP[1]); 

    // Method 2: Summation of vectors in a triangle is zero
    T pX = b11 * EOP2[3] + b12 * EOP2[4] + b13 * EOP2[5];
    T pY = b21 * EOP2[3] + b22 * EOP2[4] + b23 * EOP2[5];
    T pZ = b31 * EOP2[3] + b32 * EOP2[4] + b33 * EOP2[5];

    T rX = r11 * pX + r12 * pY + r13 * pZ;
    T rY = r21 * pX + r22 * pY + r23 * pZ;
    T rZ = r31 * pX + r32 * pY + r33 * pZ;

    T qX = a11 * EOP1[3] + a12 * EOP1[4] + a13 * EOP1[5] + ROP[3];
    T qY = a21 * EOP1[3] + a22 * EOP1[4] + a23 * EOP1[5] + ROP[4];
    T qZ = a31 * EOP1[3] + a32 * EOP1[4] + a33 * EOP1[5] + ROP[5];

  // actual cost function
//   residual[0] = rX - qX;
//   residual[1] = rY - qY;
//   residual[2] = rZ - qZ;


//   residual[0] /= T(XoStdDev_);
//   residual[1] /= T(YoStdDev_);
//   residual[2] /= T(ZoStdDev_);

  residual[0] = sqrt( (rX - qX)*(rX - qX) + (rY - qY)*(rY - qY) + (rZ - qZ)*(rZ - qZ) );


  residual[0] /= T(XoStdDev_);


  return true;
  }

 private:
  const double XoStdDev_;
  const double YoStdDev_;
  const double ZoStdDev_;
};

/////////////////////////
/// MAIN CERES-SOLVER ///
/////////////////////////
int main(int argc, char** argv) {
    Py_Initialize();
    google::InitGoogleLogging(argv[0]);

    PyRun_SimpleString("import matplotlib.pyplot as plt");
    PyRun_SimpleString("import numpy as np");
    PyRun_SimpleString("import time as TIME");
    PyRun_SimpleString("totalTime = TIME.process_time()");        
     
    std::ifstream inp;      
    std::vector<double> leastSquaresCost;
    std::vector<double> leastSquaresRedundancy;
    std::vector<double> machineLearnedCost;
    std::vector<double> machineLearnedRedundancy;
    bool doML = true;
    //////////////////////////////////////
    /// Read in the data from files
    //////////////////////////////////////
    for (int iterNum = 0; iterNum < NUMITERATION; iterNum++)
    {

        std::cout<<"---------------------------------------- Global Iteration: " << iterNum+1<<"/"<< NUMITERATION <<"----------------------------------------"<<std::endl;
        PyRun_SimpleString("t0 = TIME.process_time()");        
        PyRun_SimpleString("print( 'Start reading data' )");   
        // Reading *.pho file
        PyRun_SimpleString("print( '  Start reading image observations' )");  
        std::cout<<"  Input image filename: "<<INPUTIMAGEFILENAME<<std::endl;
        if (iterNum == 0)
            inp.open(INPUTIMAGEFILENAME);
        else
            inp.open(INPUTIMAGEFILENAMETEMP);
        std::vector<int> imageTarget, imageStation;
        std::vector<double> imageX, imageY, imageXStdDev, imageYStdDev, imageXCorr, imageYCorr;
        std::vector<std::vector<double> > MLP;
        while (true) 
        {
            int c1, c2;
            double c3, c4, c5, c6, c7, c8;
            inp  >> c1 >> c2 >> c3 >> c4 >> c5 >> c6 >> c7 >> c8;

            imageTarget.push_back(c1);
            imageStation.push_back(c2);
            imageX.push_back(c3);
            imageY.push_back(c4);
            imageXStdDev.push_back(c5);
            imageYStdDev.push_back(c6);
            imageXCorr.push_back(c7);
            imageYCorr.push_back(c8);

            std::vector<double> tempMLP;
            tempMLP.resize(2);
            tempMLP[0] = c7; 
            tempMLP[1] = c8;
            MLP.push_back(tempMLP);

            if( inp.eof() ) 
                break;
        }
        
        imageTarget.pop_back();
        imageStation.pop_back();
        imageX.pop_back();
        imageY.pop_back();
        imageXStdDev.pop_back();
        imageYStdDev.pop_back();
        imageXCorr.pop_back();
        imageYCorr.pop_back();
        MLP.pop_back();
        inp.close();
        std::cout << "    Number of observations read: "<< imageX.size() << std::endl;
        std::vector<int> imageFrameID;
        imageFrameID = imageStation;
        std::sort (imageFrameID.begin(), imageFrameID.end()); // must sort before the following unique function works
        imageFrameID.erase(std::unique(imageFrameID.begin(), imageFrameID.end()), imageFrameID.end());
        std::cout << "    Number of unique frames read: "<< imageFrameID.size() << std::endl;
        // for(int i = 0; i < imageFrameID.size(); i++)
        //     std::cout<<imageFrameID[i]<<std::endl;
        std::vector<int> imageTargetID;
        imageTargetID = imageTarget;
        std::sort (imageTargetID.begin(), imageTargetID.end()); // must sort before the following unique function works
        imageTargetID.erase(std::unique(imageTargetID.begin(), imageTargetID.end()), imageTargetID.end());
        std::cout << "    Number of unique targets read: "<< imageTargetID.size() << std::endl;
        // for(int i = 0; i < imageTargetID.size(); i++)
        //     std::cout<<imageTargetID[i]<<std::endl;

        std::cout << "    Approximate redundancy (2*img - 6*EOP - 3*XYZ): "<< 2*imageX.size() - 6*imageFrameID.size() - 3*imageTargetID.size() << std::endl;

        if (imageX.size() >=5)
        {
            std::cout << "    First 5 lines of input image file "<< std::endl;
            for (int i = 0; i < 5; i++)
            {
                std::cout<<" \t " <<imageTarget[i]<<" \t "<<imageStation[i]<<" \t "<<imageX[i]<<" \t "<<imageY[i]<<" \t "<<imageXStdDev[i]<<" \t "<<imageYStdDev[i]<<" \t "<<imageXCorr[i]<<" \t "<<imageYCorr[i]<<std::endl;
            }
        }

        /// write a temporary *.pho file for communicating with python
        if (iterNum == 0) // only do this for first iteration where we copy the file
        {
            std::cout<<"  Copying the *.pho file..."<<std::endl;
            FILE *fout = fopen(INPUTIMAGEFILENAMETEMP, "w");
            for(int i = 0; i < imageTarget.size(); ++i)
            {
                fprintf(fout, "%i %i %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf\n", imageTarget[i], imageStation[i], imageX[i], imageY[i], imageXStdDev[i], imageYStdDev[i], imageXCorr[i], imageYCorr[i]);
            }
            fclose(fout);
        }

        // Reading *.eop file
        PyRun_SimpleString("print( '  Start reading EOPs' )");          
        std::cout<<"  Input EOP filename: "<<INPUTEOPFILENAME<<std::endl;
        inp.open(INPUTEOPFILENAME);
        std::vector<int> eopStation, eopCamera;
        std::vector<double> eopXo, eopYo, eopZo, eopOmega, eopPhi, eopKappa;
        std::vector<std::vector<double> > EOP;
        while (true) 
        {
            int c1, c2;
            double c3, c4, c5, c6, c7, c8;
            inp  >> c1 >> c2 >> c3 >> c4 >> c5 >> c6 >> c7 >> c8;

            //convert angles to radians
            c6 *= PI / 180.0;
            c7 *= PI / 180.0;
            c8 *= PI / 180.0;


            eopStation.push_back(c1);
            eopCamera.push_back(c2);
            eopXo.push_back(c3);
            eopYo.push_back(c4);
            eopZo.push_back(c5);
            eopOmega.push_back(c6);
            eopPhi.push_back(c7);
            eopKappa.push_back(c8);

            std::vector<double> tempEOP;
            tempEOP.resize(6);
            tempEOP[0] = c6; // omega
            tempEOP[1] = c7; // phi
            tempEOP[2] = c8; // kappa
            tempEOP[3] = c3; // Xo
            tempEOP[4] = c4; // Yo
            tempEOP[5] = c5; // Zo
            EOP.push_back(tempEOP);

            if( inp.eof() ) 
                break;
        }
        
        eopStation.pop_back();
        eopCamera.pop_back();
        eopXo.pop_back();
        eopYo.pop_back();
        eopZo.pop_back();
        eopOmega.pop_back();
        eopPhi.pop_back();
        eopKappa.pop_back();
        EOP.pop_back();

        inp.close();
        std::cout << "    Number of EOPs read: "<< eopStation.size() << std::endl;
        std::vector<int> eopCameraID;
        eopCameraID = eopCamera;
        std::sort (eopCameraID.begin(), eopCameraID.end()); // must sort before the following unique function works
        eopCameraID.erase(std::unique(eopCameraID.begin(), eopCameraID.end()), eopCameraID.end());
        std::cout << "    Number of cameras read: "<< eopCameraID.size() << std::endl;

        if (eopStation.size() >=2)
        {
            std::cout << "    First 2 lines of EOP file "<< std::endl;
            for (int i = 0; i < 2; i++)
            {
                std::cout<<" \t " <<eopStation[i]<<" \t "<<eopCamera[i]<<" \t "<<eopXo[i]<<" \t "<<eopYo[i]<<" \t "<<eopZo[i]<<" \t "<<eopOmega[i]*180.0/PI<<" \t "<<eopPhi[i]*180.0/PI<<" \t "<<eopKappa[i]*180.0/PI<<std::endl;
            }
        }

        std::vector<std::vector<double> > ROP;
        std::vector<std::vector<int> >ropID;
        std::vector<int> ropMaster;
        std::vector<int> ropSlave;
        std::vector<double> ropXo, ropYo, ropZo, ropOmega, ropPhi, ropKappa;
        if(ROPMODE || WEIGHTEDROPMODE)
        {
            // Checking for ROP constraints
            PyRun_SimpleString("print( '  Start reading ROPs' )");          
            std::cout<<"  Input ROP filename: "<<INPUTROPFILENAME<<std::endl;
            inp.open(INPUTROPFILENAME);

            while (true) 
            {
                int c1, c2, c3;
                inp  >> c1 >> c2 >> c3;

                ropMaster.push_back(c1);
                ropSlave.push_back(c2);

                std::vector<int> temp;
                temp.resize(3);
                temp[0] = c1; // reference camera
                temp[1] = c2; // slave camera
                temp[2] = c3; // offset in ID to get slave to master camera
                ropID.push_back(temp);

                if( inp.eof() ) 
                    break;
            }
            
            ropMaster.pop_back();
            ropSlave.pop_back();
            ropID.pop_back();

            inp.close();
            std::cout << "    Number of ROPs read: "<< ropID.size() << std::endl;
            std::cout << "      ROP ID: " <<std::endl;
            for (int i = 0; i < ropID.size(); i++)
            {
                std::cout<<" \t " <<ropID[i][0]<<" --> "<<ropID[i][1]<<" = "<<ropID[i][2]<<std::endl;
            }

            // establish what the initial ROP should be
            for(int i = 0; i < ropID.size(); i++)
            {
                // we store them temporarily to get the median value and use that initial values
                std::vector<double> listOmega;
                std::vector<double> listPhi;
                std::vector<double> listKappa;
                std::vector<double> listXo;
                std::vector<double> listYo;
                std::vector<double> listZo;

                for(int n = 0; n < eopStation.size(); n++) // loop through all EOPS
                {
                    // std::cout<<eopCamera[n]<<" ?= "<<ropID[i][0]<<std::endl;
                    if(eopCamera[n] == ropID[i][0]) // find the right master camera
                    {
                        for(int m = 0; m < eopStation.size(); m++) // find matching slave EOP based on ROP ID
                        {
                            if( eopCamera[m] == ropID[i][1] && (eopStation[n] == eopStation[m] - ropID[i][2]) ) // find the right slave camera
                            {
                                // we found the matching EOPs
                                // n is the index of the reference
                                // m is the index of the slave
                                Eigen::MatrixXd R1(3,3);
                                Eigen::MatrixXd R2(3,3);
                                Eigen::MatrixXd T(3,1);
                                double Tx, Ty, Tz;

                                R1(0,0) = cos(eopPhi[n]) * cos(eopKappa[n]);
                                R1(0,1) = cos(eopOmega[n]) * sin(eopKappa[n]) + sin(eopOmega[n]) * sin(eopPhi[n]) * cos(eopKappa[n]);
                                R1(0,2) = sin(eopOmega[n]) * sin(eopKappa[n]) - cos(eopOmega[n]) * sin(eopPhi[n]) * cos(eopKappa[n]);

                                R1(1,0) = -cos(eopPhi[n]) * sin(eopKappa[n]);
                                R1(1,1) = cos(eopOmega[n]) * cos(eopKappa[n]) - sin(eopOmega[n]) * sin(eopPhi[n]) * sin(eopKappa[n]);
                                R1(1,2) = sin(eopOmega[n]) * cos(eopKappa[n]) + cos(eopOmega[n]) * sin(eopPhi[n]) * sin(eopKappa[n]);

                                R1(2,0) = sin(eopPhi[n]);
                                R1(2,1) = -sin(eopOmega[n]) * cos(eopPhi[n]);
                                R1(2,2) = cos(eopOmega[n]) * cos(eopPhi[n]);

                                R2(0,0) = cos(eopPhi[m]) * cos(eopKappa[m]);
                                R2(0,1) = cos(eopOmega[m]) * sin(eopKappa[m]) + sin(eopOmega[m]) * sin(eopPhi[m]) * cos(eopKappa[m]);
                                R2(0,2) = sin(eopOmega[m]) * sin(eopKappa[m]) - cos(eopOmega[m]) * sin(eopPhi[m]) * cos(eopKappa[m]);

                                R2(1,0) = -cos(eopPhi[m]) * sin(eopKappa[m]);
                                R2(1,1) = cos(eopOmega[m]) * cos(eopKappa[m]) - sin(eopOmega[m]) * sin(eopPhi[m]) * sin(eopKappa[m]);
                                R2(1,2) = sin(eopOmega[m]) * cos(eopKappa[m]) + cos(eopOmega[m]) * sin(eopPhi[m]) * sin(eopKappa[m]);

                                R2(2,0) = sin(eopPhi[m]);
                                R2(2,1) = -sin(eopOmega[m]) * cos(eopPhi[m]);
                                R2(2,2) = cos(eopOmega[m]) * cos(eopPhi[m]);

                                Tx = eopXo[m] - eopXo[n]; // leverarm in the frame of master, a vector pointing to the slave from the master
                                Ty = eopYo[m] - eopYo[n];
                                Tz = eopZo[m] - eopZo[n];

                                T(0,0) = Tx;
                                T(1,0) = Ty;
                                T(2,0) = Tz;

                                // deltaR_1_to_2 = R_m_to_2 * R_1_to_m
                                Eigen::MatrixXd deltaR = R2 * R1.transpose();

                                double deltaOmega = atan2(-deltaR(2,1), deltaR(2,2));
                                double deltaPhi   = asin (deltaR(2,0));
                                double deltaKappa = atan2(-deltaR(1,0), deltaR(0,0));

                                Eigen::MatrixXd b = R1 * T;

                                // manually calculate the boresight and leverarm
                                double deltaR21 = R1(2,0)*R2(1,0) + R1(2,1)*R2(1,1) + R1(2,2)*R2(1,2);
                                double deltaR22 = R1(2,0)*R2(2,0) + R1(2,1)*R2(2,1) + R1(2,2)*R2(2,2);
                                double deltaR20 = R1(2,0)*R2(0,0) + R1(2,1)*R2(0,1) + R1(2,2)*R2(0,2);
                                double deltaR10 = R1(1,0)*R2(0,0) + R1(1,1)*R2(0,1) + R1(1,2)*R2(0,2);
                                double deltaR00 = R1(0,0)*R2(0,0) + R1(0,1)*R2(0,1) + R1(0,2)*R2(0,2);

                                double deltaOmega2 = atan2(-deltaR21, deltaR22);
                                double deltaPhi2   = asin(deltaR20);
                                double deltaKappa2 = atan2(-deltaR10, deltaR00);
                                double bx = R1(0,0)*Tx + R1(0,1)*Ty + R1(0,2)*Tz;
                                double by = R1(1,0)*Tx + R1(1,1)*Ty + R1(1,2)*Tz;
                                double bz = R1(2,0)*Tx + R1(2,1)*Ty + R1(2,2)*Tz;

                                listOmega.push_back(deltaOmega);
                                listPhi.push_back(deltaPhi);
                                listKappa.push_back(deltaKappa);
                                listXo.push_back(b(0,0));
                                listYo.push_back(b(1,0));
                                listZo.push_back(b(2,0));

                                // std::cout<< deltaOmega2 * 180.0/PI<<", "<<deltaPhi2 * 180.0/PI<<", "<<deltaKappa2 * 180.0/PI<<", "<<bx<<", "<<by<<", "<<bz<<std::endl;
                                // std::cout<< deltaOmega * 180.0/PI<<", "<<deltaPhi * 180.0/PI<<", "<<deltaKappa * 180.0/PI<<", " <<b(0,0) <<", "<< b(1,0)<<", "<<b(2,0)<< ", distance = "<< sqrt( b(0,0)*b(0,0) + b(1,0)*b(1,0) + b(2,0)*b(2,0) )<<std::endl;

                                /////////////////////////////////////////////////////////
                                // check my own math
                                std::vector<double> ROP(3);
                                ROP[0] = deltaOmega2;
                                ROP[1] = deltaPhi2;
                                ROP[2] = deltaKappa2;
                                // rotation from sensor 2 to sensor 1
                                double r11 = cos(ROP[1]) * cos(ROP[2]);
                                double r12 = cos(ROP[0]) * sin(ROP[2]) + sin(ROP[0]) * sin(ROP[1]) * cos(ROP[2]);
                                double r13 = sin(ROP[0]) * sin(ROP[2]) - cos(ROP[0]) * sin(ROP[1]) * cos(ROP[2]);

                                double r21 = -cos(ROP[1]) * sin(ROP[2]);
                                double r22 = cos(ROP[0]) * cos(ROP[2]) - sin(ROP[0]) * sin(ROP[1]) * sin(ROP[2]);
                                double r23 = sin(ROP[0]) * cos(ROP[2]) + cos(ROP[0]) * sin(ROP[1]) * sin(ROP[2]);

                                double r31 = sin(ROP[1]);
                                double r32 = -sin(ROP[0]) * cos(ROP[1]);
                                double r33 = cos(ROP[0]) * cos(ROP[1]); 

                                // R_1To2 = R_mTo2 * R_1Tom 
                                double a11 = R1(0,0);
                                double a12 = R1(0,1);
                                double a13 = R1(0,2);
                                double a21 = R1(1,0);
                                double a22 = R1(1,1);
                                double a23 = R1(1,2);
                                double a31 = R1(2,0);
                                double a32 = R1(2,1);
                                double a33 = R1(2,2);

                                double b11 = R2(0,0);
                                double b12 = R2(0,1);
                                double b13 = R2(0,2);
                                double b21 = R2(1,0);
                                double b22 = R2(1,1);
                                double b23 = R2(1,2);
                                double b31 = R2(2,0);
                                double b32 = R2(2,1);
                                double b33 = R2(2,2);

                                double m11 = b11 * a11 + b12 * a12 + b13 * a13;
                                double m12 = b11 * a21 + b12 * a22 + b13 * a23;
                                double m13 = b11 * a31 + b12 * a32 + b13 * a33;

                                double m21 = b21 * a11 + b22 * a12 + b23 * a13;
                                double m22 = b21 * a21 + b22 * a22 + b23 * a23;
                                double m23 = b21 * a31 + b22 * a32 + b23 * a33;

                                double m31 = b31 * a11 + b32 * a12 + b33 * a13;
                                double m32 = b31 * a21 + b32 * a22 + b33 * a23;
                                double m33 = b31 * a31 + b32 * a32 + b33 * a33;

                                    // I = boresight_2To1 * R_1To2
                                    double dR32 = r31*m12 + r32*m22 + r33*m32;
                                    double dR33 = r31*m13 + r32*m23 + r33*m33;
                                    double dR31 = r31*m11 + r32*m21 + r33*m31;
                                    double dR21 = r21*m11 + r22*m21 + r23*m31;
                                    double dR11 = r11*m11 + r12*m21 + r13*m31;

                                    double deltaOmega3 = atan2(-dR32, dR33);
                                    double deltaPhi3   = asin(dR31);
                                    double deltaKappa3 = atan2(-dR21, dR11);

                                    // std::cout<<"Should be EXACTLY to zero: "<<deltaOmega3 * 180.0/PI <<", "<<deltaPhi3 * 180.0/PI <<", "<<deltaKappa3 * 180.0/PI <<std::endl;
                                // Method 2: Summation of vectors
                                double pX = b11 * eopXo[m] + b12 * eopYo[m] + b13 * eopZo[m];
                                double pY = b21 * eopXo[m] + b22 * eopYo[m] + b23 * eopZo[m];
                                double pZ = b31 * eopXo[m] + b32 * eopYo[m] + b33 * eopZo[m];

                                double qX = a11 * eopXo[n] + a12 * eopYo[n] + a13 * eopZo[n] + bx;
                                double qY = a21 * eopXo[n] + a22 * eopYo[n] + a23 * eopZo[n] + by;
                                double qZ = a31 * eopXo[n] + a32 * eopYo[n] + a33 * eopZo[n] + bz;

                                double rX = r11 * pX + r12 * pY + r13 * pZ;
                                double rY = r21 * pX + r22 * pY + r23 * pZ;
                                double rZ = r31 * pX + r32 * pY + r33 * pZ;
                             
                                //std::cout<<"r should equal q: "<<rX<<", "<<rY<<", "<<rZ<<" = "<<qX<<", "<<qY<<", "<<qZ<<std::endl;

                            }
                        }
                    }
                }
                double tempOmega = calcMedian(listOmega);
                double tempPhi = calcMedian(listPhi);
                double tempKappa = calcMedian(listKappa);
                double tempXo = calcMedian(listXo);
                double tempYo = calcMedian(listYo);
                double tempZo = calcMedian(listZo);
                // double tempOmega = (listOmega[1]);
                // double tempPhi = (listPhi[1]);
                // double tempKappa = (listKappa[1]);
                // double tempXo = (listXo[1]);
                // double tempYo = (listYo[1]);
                // double tempZo = (listZo[1]);

                std::cout<<"      Median boresight and leverarm (from master to slave): "<<std::endl;
                std::cout<<"        "<<tempOmega * 180.0/PI<<", "<< tempPhi * 180.0/PI << ", " << tempKappa * 180.0/PI << ", " << tempXo << ", " << tempYo << ", " << tempZo <<". Distance: "<< sqrt(tempXo*tempXo+tempYo*tempYo+tempZo*tempZo) << std::endl;
                std::cout<<"      Mean boresight and leverarm (from master to slave):" <<std::endl;
                std::cout<<"        "<<std::accumulate( listOmega.begin(), listOmega.end(), 0.0)/listOmega.size() * 180.0/PI<<", "<< std::accumulate( listPhi.begin(), listPhi.end(), 0.0)/listPhi.size() * 180.0/PI << ", " << std::accumulate( listKappa.begin(), listKappa.end(), 0.0)/listKappa.size() * 180.0/PI << ", " << std::accumulate( listXo.begin(), listXo.end(), 0.0)/listXo.size() <<", " << std::accumulate( listYo.begin(), listYo.end(), 0.0)/listYo.size() << ", " << std::accumulate( listZo.begin(), listZo.end(), 0.0)/listZo.size() << ". Distance: " << sqrt( pow(std::accumulate( listXo.begin(), listXo.end(), 0.0)/listXo.size(),2.0)+pow(std::accumulate( listYo.begin(), listYo.end(), 0.0)/listYo.size(),2.0)+pow(std::accumulate( listZo.begin(), listZo.end(), 0.0)/listZo.size(),2.0) )<<std::endl;

                double tempOmegaStdDev = calcStdDev(listOmega);
                double tempPhiStdDev = calcStdDev(listPhi);
                double tempKappaStdDev = calcStdDev(listKappa);
                double tempXoStdDev = calcStdDev(listXo);
                double tempYoStdDev = calcStdDev(listYo);
                double tempZoStdDev = calcStdDev(listZo);

                std::cout<<"      Std. Dev. boresight [deg] and leverarm:" <<std::endl;
                std::cout<<"        "<<tempOmegaStdDev * 180.0/PI<<", "<< tempPhiStdDev * 180.0/PI << ", " << tempKappaStdDev * 180.0/PI << ", " << tempXoStdDev << ", " << tempYoStdDev << ", " << tempZoStdDev << std::endl;

                ropOmega.push_back(tempOmega);
                ropPhi.push_back(tempPhi);
                ropKappa.push_back(tempKappa);
                ropXo.push_back(tempXo);
                ropYo.push_back(tempYo);
                ropZo.push_back(tempZo);

                std::vector<double> tempROP;
                tempROP.resize(6);
                tempROP[0] = tempOmega;
                tempROP[1] = tempPhi;
                tempROP[2] = tempKappa;
                tempROP[3] = tempXo;
                tempROP[4] = tempYo;
                tempROP[5] = tempZo;

                ///////////
                /// Ad-hoc fix: For typing in the constant ROP instead of approximating it from the EOP
                //////////
                tempROP[0] = PI/180.0 * -7.231491;
                tempROP[1] = PI/180.0 *   -83.505478;
                tempROP[2] = PI/180.0 *   -6.160191;
                tempROP[3] =            -1547.322726;
                tempROP[4] =             16.001595;
                tempROP[5] =           -1303.126412;

                ROP.push_back(tempROP);
            }
        }

        // Reading *.iop file
        PyRun_SimpleString("print( '  Start reading IOPs' )");
        std::cout<<"  Input IOP filename: "<<INPUTIOPFILENAME<<std::endl;
        inp.open(INPUTIOPFILENAME);
        std::vector<int> iopCamera, iopAxis;
        std::vector<double> iopXMin, iopYMin, iopXMax, iopYMax, iopXp, iopYp, iopC, iopA1, iopA2, iopK1, iopK2, iopK3, iopP1, iopP2;
        std::vector<std::vector<double> > IOP, AP;
        while (true) 
        {
            int c1, c2;
            double c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16;
            inp  >> c1 >> c2 >> c3 >> c4 >> c5 >> c6 >> c7 >> c8 >> c9 >> c10 >> c11 >> c12 >> c13 >> c14 >> c15 >> c16;

            iopCamera.push_back(c1);
            iopAxis.push_back(c2);
            iopXMin.push_back(c3);
            iopYMin.push_back(c4);
            iopXMax.push_back(c5);
            iopYMax.push_back(c6);
            iopXp.push_back(c7);
            iopYp.push_back(c8);
            iopC.push_back(c9);
            iopA1.push_back(c10);
            iopA2.push_back(c11);
            iopK1.push_back(c12);
            iopK2.push_back(c13);
            iopK3.push_back(c14);
            iopP1.push_back(c15);
            iopP2.push_back(c16);

            std::vector<double> tempIOP;
            tempIOP.resize(3);
            tempIOP[0] = c7;
            tempIOP[1] = c8;
            tempIOP[2] = c9;
            IOP.push_back(tempIOP);

            std::vector<double> tempAP;
            tempAP.resize(7);
            tempAP[0] = c10; //a1
            tempAP[1] = c11; //a2
            tempAP[2] = c12;
            tempAP[3] = c13;
            tempAP[4] = c14;
            tempAP[5] = c15;
            tempAP[6] = c16;
            AP.push_back(tempAP);

            if( inp.eof() ) 
                break;
        }
        
        iopCamera.pop_back();
        iopAxis.pop_back();
        iopXMin.pop_back();
        iopYMin.pop_back();
        iopXMax.pop_back();
        iopYMax.pop_back();
        iopXp.pop_back();
        iopYp.pop_back();
        iopC.pop_back();
        iopA1.pop_back();
        iopA2.pop_back();
        iopK1.pop_back();
        iopK2.pop_back();
        iopK3.pop_back();
        iopP1.pop_back();
        iopP2.pop_back();
        IOP.pop_back();
        AP.pop_back();

        inp.close();
        std::cout << "    Number of IOPs read: "<< iopCamera.size() << std::endl;
        if(true)
        {
            std::cout << "      IOP: " <<std::endl;
            for (int i = 0; i < iopCamera.size(); i++)
            {
                std::cout<<" \t " <<iopCamera[i]<<" \t "<<iopAxis[i]<<" \t "<<iopXMin[i]<<" \t "<<iopYMin[i]<<" \t "<<iopXMax[i]<<" \t "<<iopYMax[i]<<" \t "<<iopXp[i]<<" \t "<<iopYp[i]<<" \t "<<iopC[i]<<std::endl;
            }
        }

        // Reading *.xyz file
        PyRun_SimpleString("print( '  Start reading XYZ' )");  
        std::cout<<"  Input XYZ filename: "<<INPUTXYZFILENAME<<std::endl;
        inp.open(INPUTXYZFILENAME);
        std::vector<int> xyzTarget;
        std::vector<double> xyzX, xyzY, xyzZ, xyzXStdDev, xyzYStdDev, xyzZStdDev;
        std::vector<std::vector<double> > XYZ;
        while (true) 
        {
            int c1;
            double c2, c3, c4, c5, c6, c7;
            inp  >> c1 >> c2 >> c3 >> c4 >> c5 >> c6 >> c7;

            xyzTarget.push_back(c1);
            xyzX.push_back(c2);
            xyzY.push_back(c3);
            xyzZ.push_back(c4);
            xyzXStdDev.push_back(c5);
            xyzYStdDev.push_back(c6);
            xyzZStdDev.push_back(c7);

            std::vector<double> tempXYZ;
            tempXYZ.resize(3);
            tempXYZ[0] = c2; // X
            tempXYZ[1] = c3; // Y
            tempXYZ[2] = c4; // Z
            XYZ.push_back(tempXYZ);

            if( inp.eof() ) 
                break;
        }
        
        xyzTarget.pop_back();
        xyzX.pop_back();
        xyzY.pop_back();
        xyzZ.pop_back();
        xyzXStdDev.pop_back();
        xyzYStdDev.pop_back();
        xyzZStdDev.pop_back();
        XYZ.pop_back();

        inp.close();

        std::cout << "    Number of XYZ read: "<< xyzTarget.size() << std::endl;
        std::vector<int> xyzTargetID;
        xyzTargetID = xyzTarget;
        std::sort (xyzTargetID.begin(), xyzTargetID.end()); // must sort before the following unique function works
        xyzTargetID.erase(std::unique(xyzTargetID.begin(), xyzTargetID.end()), xyzTargetID.end());
        std::cout << "    Number of unique Targets read: "<< xyzTargetID.size() << std::endl;
        if(DEBUGMODE)
        {
            std::cout << "      Data: " <<std::endl;
            for (int i = 0; i < xyzTarget.size(); i++)
            {
                std::cout<<xyzTarget[i]<<" \t "<<xyzX[i]<<" \t "<<xyzY[i]<<" \t "<<xyzZ[i]<<" \t "<<xyzXStdDev[i]<<" \t "<<xyzYStdDev[i]<<" \t "<<xyzZStdDev[i]<<std::endl;
            }
        }

        if (xyzTarget.size() >=5)
        {
            std::cout << "    First 5 lines of XYZ file "<< std::endl;
            for (int i = 0; i < 5; i++)
            {
                std::cout<<" \t " <<xyzTarget[i]<<" \t " <<xyzX[i]<<" \t "<<xyzY[i]<<" \t "<<xyzZ[i]<<" \t "<<xyzXStdDev[i]<<" \t "<<xyzYStdDev[i]<<" \t "<<xyzZStdDev[i]<<std::endl;
            }
        }

        PyRun_SimpleString("print( 'Done reading data:', round(TIME.process_time()-t0, 3), 's' )");


        // //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // /// RANSAC Resection
        // //////////////////////////////////////////////////////////////////////////////////////////////////////////////   
        // PyRun_SimpleString("print 'RANSAC Single Photo Resection...' ");
    
        // ceres::Problem problemResection;

        // // define the parameters in the order we want
        // // EOP             
        // for(int n = 0; n < EOP.size(); n++) 
        //     problemResection.AddParameterBlock(&EOP[n][0], 6);  

        // ceres::LossFunction* loss = NULL;
        // loss = new ceres::HuberLoss(1.0);
        // // loss = new ceres::CauchyLoss(0.5);

        // // Conventional collinearity condition, no machine learning
        // for(int n = 0; n < imageX.size(); n++) // loop through all observations
        // {
        //      std::vector<int>::iterator it;
        //      it = std::find(xyzTarget.begin(), xyzTarget.end(), imageTarget[n]);
        //      int indexPoint = std::distance(xyzTarget.begin(),it);
        //      // std::cout<<"indexPoint: "<<indexPoint<<", ID: "<< imageTarget[n]<<std::endl;

        //      it = std::find(eopStation.begin(), eopStation.end(), imageStation[n]);
        //      int indexPose = std::distance(eopStation.begin(),it);
        //      // std::cout<<"index: "<<indexPose<<", ID: "<< imageStation[n]<<std::endl;

        //     //  std::cout<<"EOP: "<< EOP[indexPose][3] <<", " << EOP[indexPose][4] <<", " << EOP[indexPose][5]  <<std::endl;
        //     //  std::cout<<"XYZ: "<< XYZ[indexPoint][0] <<", " << XYZ[indexPoint][1] <<", " << XYZ[indexPoint][2]  <<std::endl;

        //     ceres::CostFunction* cost_function =
        //         new ceres::AutoDiffCostFunction<singlePhotoResection, 2, 6>(
        //             new singlePhotoResection(imageX[n],imageY[n],imageXStdDev[n], imageYStdDev[n],iopXp[indexSensor],iopYp[indexSensor], xyzX[indexPoint], xyzY[indexPoint], xyzZ[indexPoint]));
        //     problem.AddResidualBlock(cost_function, loss, &EOP[indexPose][0]);  

        // }

        // PyRun_SimpleString("print 'RANSAC Single Photo Resection DONE' ");


        // //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // /// Initialize the unknowns
        // //////////////////////////////////////////////////////////////////////////////////////////////////////////////   
        // std::vector<double> X;
        // X.push_back(1.0);
        // X.push_back(10.0);
        // X.push_back(100.0);


        // //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // /// Set up cost functions
        // //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        PyRun_SimpleString("t0 = TIME.process_time()");        
        PyRun_SimpleString("print( 'Start building Ceres-Solver cost functions' )");
    
        std::vector<int> sensorReferenceID; // for use when outting the residuals
        std::vector<int> pointReferenceID; // for use when outting the residuals
        std::vector<int> frameReferenceID;  // for use when outting the residuals
        sensorReferenceID.resize(imageX.size());
        pointReferenceID.resize(imageX.size());
        frameReferenceID.resize(imageX.size());
        std::vector<double> variances;
        ceres::Problem problem;

        // define the parameters in the order we want
        // EOP      
        // XYZ          
        // IOP                
        // AP                    
        // MLP
        // ROP
        for(int n = 0; n < EOP.size(); n++) 
            problem.AddParameterBlock(&EOP[n][0], 6);  
        for(int n = 0; n < XYZ.size(); n++) 
            problem.AddParameterBlock(&XYZ[n][0], 3);  
        for(int n = 0; n < IOP.size(); n++) 
            problem.AddParameterBlock(&IOP[n][0], 3);
        for(int n = 0; n < AP.size(); n++) 
            problem.AddParameterBlock(&AP[n][0], 7);   
        // for(int n = 0; n < MLP.size(); n++) 
        //     problem.AddParameterBlock(&MLP[n][0], 2);  
        if(ROPMODE || WEIGHTEDROPMODE)
        {
            for(int n = 0; n < ROP.size(); n++) 
                problem.AddParameterBlock(&ROP[n][0], 6);  
        }

        ceres::LossFunction* loss = NULL; // default to normal Gaussian
        // loss = new ceres::HuberLoss(1.0);

        // ceres::LossFunction* loss2 = NULL;
        // loss = new ceres::CauchyLoss(0.5);


        // Conventional collinearity condition, no machine learning
        if (true)
        {
            std::cout<<"   RUNNING CONVENTIONAL COLLINEARITY EQUATIONS..."<<std::endl;
            for(int n = 0; n < imageX.size(); n++) // loop through all observations
            {
                std::vector<int>::iterator it;
                it = std::find(xyzTarget.begin(), xyzTarget.end(), imageTarget[n]);
                int indexPoint = std::distance(xyzTarget.begin(),it);
                //  std::cout<<"indexPoint: "<<indexPoint<<", ID: "<< imageTarget[n]<<std::endl;

                it = std::find(eopStation.begin(), eopStation.end(), imageStation[n]);
                int indexPose = std::distance(eopStation.begin(),it);
                //  std::cout<<"index: "<<indexPose<<", ID: "<< imageStation[n]<<std::endl;

                it = std::find(iopCamera.begin(), iopCamera.end(), eopCamera[indexPose]);
                int indexSensor = std::distance(iopCamera.begin(),it);
                // std::cout<<"index: "<<indexSensor<<", ID: "<< eopCamera[indexPose]<<std::endl;   

                //  std::cout<<"EOP: "<< EOP[indexPose][3] <<", " << EOP[indexPose][4] <<", " << EOP[indexPose][5]  <<std::endl;
                //  std::cout<<"XYZ: "<< XYZ[indexPoint][0] <<", " << XYZ[indexPoint][1] <<", " << XYZ[indexPoint][2]  <<std::endl;

                // for book keeping and output later
                sensorReferenceID[n] = iopCamera[indexSensor]; // which sensor was used
                pointReferenceID[n]  = xyzTarget[indexPoint];  // ID of the target point this observation corresponds to
                frameReferenceID[n]  = eopStation[indexPose];

                // imageXStdDev[n] *= 10000.0; // for debugging use only
                // imageYStdDev[n] *= 10000.0;

                ceres::CostFunction* cost_function =
                    new ceres::AutoDiffCostFunction<collinearity, 2, 6, 3, 3, 7>(
                        new collinearity(imageX[n],imageY[n],imageXStdDev[n], imageYStdDev[n],iopXp[indexSensor],iopYp[indexSensor]));
                problem.AddResidualBlock(cost_function, loss, &EOP[indexPose][0], &XYZ[indexPoint][0], &IOP[indexSensor][0], &AP[indexSensor][0]);  

                problem.SetParameterBlockConstant(&IOP[indexSensor][0]);
                problem.SetParameterBlockConstant(&AP[indexSensor][0]);
                // problem.SetParameterBlockConstant(&XYZ[indexPoint][0]);

                variances.push_back(imageXStdDev[n]*imageXStdDev[n]);
                variances.push_back(imageYStdDev[n]*imageYStdDev[n]);
            }
                // no machine learning so turn off the Python learning script at the end
                doML = false;
        }

        // Stereographic collinearity condition, no machine learning
        if (false)
        {
            std::cout<<"   RUNNING STEREOGRAPHIC PROJECTION COLLINEARITY EQUATIONS..."<<std::endl;

            for(int n = 0; n < imageX.size(); n++) // loop through all observations
            {
                std::vector<int>::iterator it;
                it = std::find(xyzTarget.begin(), xyzTarget.end(), imageTarget[n]);
                int indexPoint = std::distance(xyzTarget.begin(),it);
                //  std::cout<<"indexPoint: "<<indexPoint<<", ID: "<< imageTarget[n]<<std::endl;

                it = std::find(eopStation.begin(), eopStation.end(), imageStation[n]);
                int indexPose = std::distance(eopStation.begin(),it);
                //  std::cout<<"index: "<<indexPose<<", ID: "<< imageStation[n]<<std::endl;

                it = std::find(iopCamera.begin(), iopCamera.end(), eopCamera[indexPose]);
                int indexSensor = std::distance(iopCamera.begin(),it);
                // std::cout<<"index: "<<indexSensor<<", ID: "<< eopCamera[indexPose]<<std::endl;   

                //  std::cout<<"EOP: "<< EOP[indexPose][3] <<", " << EOP[indexPose][4] <<", " << EOP[indexPose][5]  <<std::endl;
                //  std::cout<<"XYZ: "<< XYZ[indexPoint][0] <<", " << XYZ[indexPoint][1] <<", " << XYZ[indexPoint][2]  <<std::endl;

                // for book keeping
                sensorReferenceID[n] = iopCamera[indexSensor]; // which sensor was used
                pointReferenceID[n]  = xyzTarget[indexPoint];  // ID of the target point this observation corresponds to
                frameReferenceID[n]  = eopStation[indexPose];

                // imageXStdDev[n] *= 10000.0;
                // imageYStdDev[n] *= 10000.0;

                ceres::CostFunction* cost_function =
                    new ceres::AutoDiffCostFunction<collinearityStereographic, 2, 6, 3, 3, 7>(
                        new collinearityStereographic(imageX[n],imageY[n],imageXStdDev[n], imageYStdDev[n],iopXp[indexSensor],iopYp[indexSensor]));
                problem.AddResidualBlock(cost_function, loss, &EOP[indexPose][0], &XYZ[indexPoint][0], &IOP[indexSensor][0], &AP[indexSensor][0]);  

                // ceres::CostFunction* cost_function =
                //     new ceres::AutoDiffCostFunction<fisheyeEquidistant, 2, 6, 3, 3, 7>(
                //         new fisheyeEquidistant(imageX[n],imageY[n],imageXStdDev[n], imageYStdDev[n],iopXp[indexSensor],iopYp[indexSensor]));
                // problem.AddResidualBlock(cost_function, loss, &EOP[indexPose][0], &XYZ[indexPoint][0], &IOP[indexSensor][0], &AP[indexSensor][0]);  

                // ceres::CostFunction* cost_function =
                //     new ceres::AutoDiffCostFunction<fisheyeStereographic, 2, 6, 3, 3, 7>(
                //         new fisheyeStereographic(imageX[n],imageY[n],imageXStdDev[n], imageYStdDev[n],iopXp[indexSensor],iopYp[indexSensor]));
                // problem.AddResidualBlock(cost_function, loss, &EOP[indexPose][0], &XYZ[indexPoint][0], &IOP[indexSensor][0], &AP[indexSensor][0]);  

                problem.SetParameterLowerBound(&IOP[indexSensor][0], 2, 0.0);

                // problem.SetParameterBlockConstant(&IOP[indexSensor][0]);
                problem.SetParameterBlockConstant(&AP[indexSensor][0]);
                // problem.SetParameterBlockConstant(&XYZ[indexPoint][0]);

                variances.push_back(imageXStdDev[n]*imageXStdDev[n]);
                variances.push_back(imageYStdDev[n]*imageYStdDev[n]);
            }
            // no machine learning so turn off the Python learning script at the end
            doML = false;
        }

        if (INITIALIZEAP && iterNum == 0)
        {
            std::cout<<"START: Initializing the APs by backprojecting the initial XYZ using the approximate EOP"<<std::endl;
            // Collinearity condition with machine learned parameters and ROP
            for(int n = 0; n < imageX.size(); n++) // loop through all observations
            {
                std::vector<int>::iterator it;
                it = std::find(xyzTarget.begin(), xyzTarget.end(), imageTarget[n]);
                int indexPoint = std::distance(xyzTarget.begin(),it);
                // std::cout<<"indexPoint: "<<indexPoint<<", ID: "<< imageTarget[n]<<std::endl;

                it = std::find(eopStation.begin(), eopStation.end(), imageStation[n]);
                int indexPose = std::distance(eopStation.begin(),it);
                // std::cout<<"indexPose: "<<indexPose<<", ID: "<< imageStation[n]<<std::endl;

                it = std::find(iopCamera.begin(), iopCamera.end(), eopCamera[indexPose]);
                int indexSensor = std::distance(iopCamera.begin(),it);
                // std::cout<<"indexSensor: "<<indexSensor<<", ID: "<< eopCamera[indexPose]<<std::endl; 

                it = std::find(ropSlave.begin(), ropSlave.end(), iopCamera[indexSensor]);
                int indexROPSlave = std::distance(ropSlave.begin(),it);
  
                if (ROPMODE && it!=ropSlave.end() && iopCamera[indexSensor] == *it) // is a slave in ROP constraint
                {
                    it = std::find(eopStation.begin(), eopStation.end(), imageStation[n] - ropID[indexROPSlave][2]);
                    int indexPoseMaster = std::distance(eopStation.begin(),it);
                    // std::cout<<"indexPoseMaster: "<<indexPoseMaster<<", ID: "<< imageStation[n] - ropID[indexROPSlave][2]<<std::endl;
                    // std::cout<<"indexROP: "<< indexROPSlave<<std::endl;        

                    ceres::CostFunction* cost_function =
                        new ceres::AutoDiffCostFunction<collinearityMachineLearnedROP, 2, 6, 6, 3, 3, 7>(
                            new collinearityMachineLearnedROP(imageX[n],imageY[n],imageXStdDev[n], imageYStdDev[n],iopXp[indexSensor],iopYp[indexSensor], imageXCorr[n], imageYCorr[n]));
                    problem.AddResidualBlock(cost_function, loss, &EOP[indexPoseMaster][0], &ROP[indexROPSlave][0], &XYZ[indexPoint][0], &IOP[indexSensor][0], &AP[indexSensor][0]);  
                
                }
                // else if(eopCamera[indexPose] != ropSlave[indexROPSlave]) // not a slave in ROP constraint
                else
                {
                    // ceres::CostFunction* cost_function =
                    //     new ceres::AutoDiffCostFunction<omniCollinearityMachineLearnedSimple, 2, 6, 3, 3, 7>(
                    //         new omniCollinearityMachineLearnedSimple(imageX[n],imageY[n],imageXStdDev[n], imageYStdDev[n],iopXp[indexSensor],iopYp[indexSensor], imageXCorr[n], imageYCorr[n]));
                    // problem.AddResidualBlock(cost_function, loss, &EOP[indexPose][0], &XYZ[indexPoint][0], &IOP[indexSensor][0], &AP[indexSensor][0]); 


                    // rotation from map to sensor
                    double r11 = cos(EOP[indexPose][1]) * cos(EOP[indexPose][2]);
                    double r12 = cos(EOP[indexPose][0]) * sin(EOP[indexPose][2]) + sin(EOP[indexPose][0]) * sin(EOP[indexPose][1]) * cos(EOP[indexPose][2]);
                    double r13 = sin(EOP[indexPose][0]) * sin(EOP[indexPose][2]) - cos(EOP[indexPose][0]) * sin(EOP[indexPose][1]) * cos(EOP[indexPose][2]);

                    double r21 = -cos(EOP[indexPose][1]) * sin(EOP[indexPose][2]);
                    double r22 = cos(EOP[indexPose][0]) * cos(EOP[indexPose][2]) - sin(EOP[indexPose][0]) * sin(EOP[indexPose][1]) * sin(EOP[indexPose][2]);
                    double r23 = sin(EOP[indexPose][0]) * cos(EOP[indexPose][2]) + cos(EOP[indexPose][0]) * sin(EOP[indexPose][1]) * sin(EOP[indexPose][2]);

                    double r31 = sin(EOP[indexPose][1]);
                    double r32 = -sin(EOP[indexPose][0]) * cos(EOP[indexPose][1]);
                    double r33 = cos(EOP[indexPose][0]) * cos(EOP[indexPose][1]);

                    // rigid body transformation
                    double XTemp = r11 * ( XYZ[indexPoint][0] - EOP[indexPose][3] ) + r12 * ( XYZ[indexPoint][1] - EOP[indexPose][4] ) + r13 * ( XYZ[indexPoint][2] - EOP[indexPose][5] );
                    double YTemp = r21 * ( XYZ[indexPoint][0] - EOP[indexPose][3] ) + r22 * ( XYZ[indexPoint][1] - EOP[indexPose][4] ) + r23 * ( XYZ[indexPoint][2] - EOP[indexPose][5] );
                    double ZTemp = r31 * ( XYZ[indexPoint][0] - EOP[indexPose][3] ) + r32 * ( XYZ[indexPoint][1] - EOP[indexPose][4] ) + r33 * ( XYZ[indexPoint][2] - EOP[indexPose][5] );

                    // modified omnidirectional collinearity condition dividing x and y
                    //   double x = IOP[indexSensor][2] * atan2(XTemp , -ZTemp);
                    //   double y = IOP[indexSensor][2] * atan2(YTemp , -ZTemp);
                    // modified omnidirectional collinearity condition using spatial angle
                    double x = IOP[indexSensor][2] * atan2(sqrt(XTemp*XTemp+YTemp*YTemp),-ZTemp) / sqrt((YTemp/XTemp)*(YTemp/XTemp) + 1.0);
                    double y = IOP[indexSensor][2] * atan2(sqrt(XTemp*XTemp+YTemp*YTemp),-ZTemp) / sqrt((XTemp/YTemp)*(XTemp/YTemp) + 1.0);

                    // camera correction model AP = a1, a2, k1, k2, k3, p1, p2, ...
                    double x_bar = imageX[n] - iopXp[indexSensor];
                    double y_bar = imageY[n] - iopYp[indexSensor];
                    double r = sqrt(x_bar*x_bar + y_bar*y_bar);

                    double delta_x = x_bar*(AP[indexSensor][2]*r*r+AP[indexSensor][3]*r*r*r*r+AP[indexSensor][4]*r*r*r*r*r*r) + AP[indexSensor][5]*(r*r+(2.0)*x_bar*x_bar)+(2.0)*AP[indexSensor][6]*x_bar*y_bar + AP[indexSensor][0]*x_bar+AP[indexSensor][1]*y_bar;
                    double delta_y = y_bar*(AP[indexSensor][2]*r*r+AP[indexSensor][3]*r*r*r*r+AP[indexSensor][4]*r*r*r*r*r*r) + AP[indexSensor][6]*(r*r+(2.0)*y_bar*y_bar)+(2.0)*AP[indexSensor][5]*x_bar*y_bar;

                    // double x_true = x + IOP[indexSensor][0] + delta_x - imageXCorr[n]; // imageXCorr probably is zero is most case if the input file is zero
                    // double y_true = y + IOP[indexSensor][1] + delta_y - imageYCorr[n];

                    // // actual cost function
                    // residual[0] = x_true - T(x_); // x-residual = reprojected - observed
                    // residual[1] = y_true - T(y_); // y-residual 

                    imageXCorr[n] = round(x + IOP[indexSensor][0] + delta_x - imageX[n]);
                    imageYCorr[n] = round(y + IOP[indexSensor][1] + delta_y - imageY[n]);

                }
            }
        std::cout<<"   Done: Initializing the APs by backprojecting the initial XYZ using the approximate EOP"<<std::endl;
        }


        // if(ROPMODE)
        // {
        //     std::cout<<"  Doing absolute ROP constraint..."<<std::endl;
        // }
        //
        //
        // // Collinearity condition with machine learned parameters and ROP
        // for(int n = 0; n < imageX.size(); n++) // loop through all observations
        // {
        //     std::vector<int>::iterator it;
        //     it = std::find(xyzTarget.begin(), xyzTarget.end(), imageTarget[n]);
        //     int indexPoint = std::distance(xyzTarget.begin(),it);
        //     // std::cout<<"indexPoint: "<<indexPoint<<", ID: "<< imageTarget[n]<<std::endl;

        //     it = std::find(eopStation.begin(), eopStation.end(), imageStation[n]);
        //     int indexPose = std::distance(eopStation.begin(),it);
        //     // std::cout<<"indexPose: "<<indexPose<<", ID: "<< imageStation[n]<<std::endl;

        //     it = std::find(iopCamera.begin(), iopCamera.end(), eopCamera[indexPose]);
        //     int indexSensor = std::distance(iopCamera.begin(),it);
        //     // std::cout<<"indexSensor: "<<indexSensor<<", ID: "<< eopCamera[indexPose]<<std::endl; 

        //     it = std::find(ropSlave.begin(), ropSlave.end(), iopCamera[indexSensor]);
        //     int indexROPSlave = std::distance(ropSlave.begin(),it);
        //     // std::cout<<"indexROPSlave: "<<indexROPSlave<<", ID: "<< iopCamera[indexSensor]<<std::endl; 

        //     // for book keeping
        //     sensorReferenceID[n] = iopCamera[indexSensor];

        //     //  std::cout<<"EOP: "<< EOP[indexPose][3] <<", " << EOP[indexPose][4] <<", " << EOP[indexPose][5]  <<std::endl;
        //     //  std::cout<<"XYZ: "<< XYZ[indexPoint][0] <<", " << XYZ[indexPoint][1] <<", " << XYZ[indexPoint][2]  <<std::endl;

        //     // ceres::CostFunction* cost_function =
        //     //     new ceres::AutoDiffCostFunction<collinearityMachineLearned, 2, 6, 3, 3, 7, 2>(
        //     //         new collinearityMachineLearned(imageX[n],imageY[n],imageXStdDev[n], imageYStdDev[n],iopXp[indexSensor],iopYp[indexSensor]));
        //     //tproblem.AddResidualBlock(cost_function, NULL, &EOP[indexPose][0], &XYZ[indexPoint][0], &IOP[indexSensor][0], &AP[indexSensor][0], &MLP[n][0]);  

        //     // problem.SetParameterBlockConstant(&IOP[indexSensor][0]);
        //     // problem.SetParameterBlockConstant(&AP[indexSensor][0]);
        //     // problem.SetParameterBlockConstant(&MLP[n][0]);

        //     if (ROPMODE && it!=ropSlave.end() && iopCamera[indexSensor] == *it) // is a slave in ROP constraint
        //     {
        //         it = std::find(eopStation.begin(), eopStation.end(), imageStation[n] - ropID[indexROPSlave][2]);
        //         int indexPoseMaster = std::distance(eopStation.begin(),it);
        //         // std::cout<<"indexPoseMaster: "<<indexPoseMaster<<", ID: "<< imageStation[n] - ropID[indexROPSlave][2]<<std::endl;
        //         // std::cout<<"indexROP: "<< indexROPSlave<<std::endl;        

        //         // Absolute equality constraint version of ROP
        //         ceres::CostFunction* cost_function =
        //             new ceres::AutoDiffCostFunction<collinearityMachineLearnedROP, 2, 6, 6, 3, 3, 7>(
        //                 new collinearityMachineLearnedROP(imageX[n],imageY[n],imageXStdDev[n], imageYStdDev[n],iopXp[indexSensor],iopYp[indexSensor], imageXCorr[n], imageYCorr[n]));
        //         problem.AddResidualBlock(cost_function, loss, &EOP[indexPoseMaster][0], &ROP[indexROPSlave][0], &XYZ[indexPoint][0], &IOP[indexSensor][0], &AP[indexSensor][0]);  
            
        //         problem.SetParameterBlockConstant(&ROP[indexROPSlave][0]);
        //     }
        //     // else if(eopCamera[indexPose] != ropSlave[indexROPSlave]) // not a slave in ROP constraint
        //     else
        //     {
        //         //std::cout<<imageX[n]- IOP[indexSensor][0]<<", "<<imageY[n]- IOP[indexSensor][1]<<", "<<sqrt( std::pow(imageX[n]-IOP[indexSensor][0],2) + std::pow(imageY[n]-IOP[indexSensor][1],2) )<<std::endl;
        //         ceres::CostFunction* cost_function =
        //             new ceres::AutoDiffCostFunction<collinearityMachineLearnedSimple, 2, 6, 3, 3, 7>(
        //                 new collinearityMachineLearnedSimple(imageX[n],imageY[n],imageXStdDev[n], imageYStdDev[n],iopXp[indexSensor],iopYp[indexSensor], imageXCorr[n], imageYCorr[n]));
        //         problem.AddResidualBlock(cost_function, loss, &EOP[indexPose][0], &XYZ[indexPoint][0], &IOP[indexSensor][0], &AP[indexSensor][0]);  

        //         // ceres::CostFunction* cost_function =
        //         //     new ceres::AutoDiffCostFunction<omniCollinearityMachineLearnedSimple, 2, 6, 3, 3, 7>(
        //         //         new omniCollinearityMachineLearnedSimple(imageX[n],imageY[n],imageXStdDev[n], imageYStdDev[n],iopXp[indexSensor],iopYp[indexSensor], imageXCorr[n], imageYCorr[n]));
        //         // problem.AddResidualBlock(cost_function, loss, &EOP[indexPose][0], &XYZ[indexPoint][0], &IOP[indexSensor][0], &AP[indexSensor][0]); 

        //         // problem.SetParameterBlockConstant(&EOP[indexSensor][0]); 
        //         // problem.SetParameterBlockConstant(&XYZ[indexPoint][0]); 
        //     }
        //     problem.SetParameterBlockConstant(&IOP[indexSensor][0]);
        //     problem.SetParameterBlockConstant(&AP[indexSensor][0]);

        //     variances.push_back(imageXStdDev[n]*imageXStdDev[n]);
        //     variances.push_back(imageYStdDev[n]*imageYStdDev[n]);
        // }

        // Collinearity condition with machine learned parameters
        if (false)
        {
            std::cout<<"   Running collinearity equations with machine learning calibration parameters"<<std::endl;

            for(int n = 0; n < imageX.size(); n++) // loop through all observations
            {
                std::vector<int>::iterator it;
                it = std::find(xyzTarget.begin(), xyzTarget.end(), imageTarget[n]);
                int indexPoint = std::distance(xyzTarget.begin(),it);
                // std::cout<<"indexPoint: "<<indexPoint<<", ID: "<< imageTarget[n]<<std::endl;

                it = std::find(eopStation.begin(), eopStation.end(), imageStation[n]);
                int indexPose = std::distance(eopStation.begin(),it);
                // std::cout<<"indexPose: "<<indexPose<<", ID: "<< imageStation[n]<<std::endl;

                it = std::find(iopCamera.begin(), iopCamera.end(), eopCamera[indexPose]);
                int indexSensor = std::distance(iopCamera.begin(),it);
                // std::cout<<"indexSensor: "<<indexSensor<<", ID: "<< eopCamera[indexPose]<<std::endl; 

                // for book keeping
                sensorReferenceID[n] = iopCamera[indexSensor];

                //  std::cout<<"EOP: "<< EOP[indexPose][3] <<", " << EOP[indexPose][4] <<", " << EOP[indexPose][5]  <<std::endl;
                //  std::cout<<"XYZ: "<< XYZ[indexPoint][0] <<", " << XYZ[indexPoint][1] <<", " << XYZ[indexPoint][2]  <<std::endl;

                // ceres::CostFunction* cost_function =
                //     new ceres::AutoDiffCostFunction<collinearityMachineLearned, 2, 6, 3, 3, 7, 2>(
                //         new collinearityMachineLearned(imageX[n],imageY[n],imageXStdDev[n], imageYStdDev[n],iopXp[indexSensor],iopYp[indexSensor]));
                // problem.AddResidualBlock(cost_function, NULL, &EOP[indexPose][0], &XYZ[indexPoint][0], &IOP[indexSensor][0], &AP[indexSensor][0], &MLP[n][0]);  

                // problem.SetParameterBlockConstant(&IOP[indexSensor][0]);
                // problem.SetParameterBlockConstant(&AP[indexSensor][0]);
                // problem.SetParameterBlockConstant(&MLP[n][0]);

                ceres::CostFunction* cost_function =
                    new ceres::AutoDiffCostFunction<collinearityMachineLearnedSimple, 2, 6, 3, 3, 7>(
                        new collinearityMachineLearnedSimple(imageX[n],imageY[n],imageXStdDev[n], imageYStdDev[n],iopXp[indexSensor],iopYp[indexSensor], imageXCorr[n], imageYCorr[n]));
                problem.AddResidualBlock(cost_function, loss, &EOP[indexPose][0], &XYZ[indexPoint][0], &IOP[indexSensor][0], &AP[indexSensor][0]);  

                // problem.SetParameterBlockConstant(&IOP[indexSensor][0]);
                problem.SetParameterBlockConstant(&AP[indexSensor][0]);

                variances.push_back(imageXStdDev[n]*imageXStdDev[n]);
                variances.push_back(imageYStdDev[n]*imageYStdDev[n]);
            }
            // no machine learning so turn off the Python learning script at the end
            doML = true;
        }

        // Stereographical projection collinearity condition with machine learned parameters
        if (false)
        {
            std::cout<<"   Running stereographic projection collinearity equations with machine learning calibration parameters"<<std::endl;

            for(int n = 0; n < imageX.size(); n++) // loop through all observations
            {
                std::vector<int>::iterator it;
                it = std::find(xyzTarget.begin(), xyzTarget.end(), imageTarget[n]);
                int indexPoint = std::distance(xyzTarget.begin(),it);
                // std::cout<<"indexPoint: "<<indexPoint<<", ID: "<< imageTarget[n]<<std::endl;

                it = std::find(eopStation.begin(), eopStation.end(), imageStation[n]);
                int indexPose = std::distance(eopStation.begin(),it);
                // std::cout<<"indexPose: "<<indexPose<<", ID: "<< imageStation[n]<<std::endl;

                it = std::find(iopCamera.begin(), iopCamera.end(), eopCamera[indexPose]);
                int indexSensor = std::distance(iopCamera.begin(),it);
                // std::cout<<"indexSensor: "<<indexSensor<<", ID: "<< eopCamera[indexPose]<<std::endl; 

                // for book keeping
                sensorReferenceID[n] = iopCamera[indexSensor];

                //  std::cout<<"EOP: "<< EOP[indexPose][3] <<", " << EOP[indexPose][4] <<", " << EOP[indexPose][5]  <<std::endl;
                //  std::cout<<"XYZ: "<< XYZ[indexPoint][0] <<", " << XYZ[indexPoint][1] <<", " << XYZ[indexPoint][2]  <<std::endl;

                ceres::CostFunction* cost_function =
                    new ceres::AutoDiffCostFunction<collinearityStereographicMachineLearnedSimple, 2, 6, 3, 3, 7>(
                        new collinearityStereographicMachineLearnedSimple(imageX[n],imageY[n],imageXStdDev[n], imageYStdDev[n],iopXp[indexSensor],iopYp[indexSensor], imageXCorr[n], imageYCorr[n]));
                problem.AddResidualBlock(cost_function, loss, &EOP[indexPose][0], &XYZ[indexPoint][0], &IOP[indexSensor][0], &AP[indexSensor][0]);  

                problem.SetParameterLowerBound(&IOP[indexSensor][0], 2, 0.0);

                // problem.SetParameterBlockConstant(&IOP[indexSensor][0]);
                problem.SetParameterBlockConstant(&AP[indexSensor][0]);
                // problem.SetParameterBlockConstant(&XYZ[indexPoint][0]);

                variances.push_back(imageXStdDev[n]*imageXStdDev[n]);
                variances.push_back(imageYStdDev[n]*imageYStdDev[n]);

            }
            // no machine learning so turn off the Python learning script at the end
            doML = true;
        }

        if (WEIGHTEDROPMODE)
        {
            std::cout<<"  Doing weighted ROP constraint..."<<std::endl;
            // Weighted ROP
            for(int n = 0; n < eopStation.size(); n++) // loop through all observations
            {
                std::vector<int>::iterator it;
                int indexPose = n;
                // std::cout<<"indexPose: "<<indexPose<<", ID: "<< eopStation[n]<<std::endl;

                it = std::find(iopCamera.begin(), iopCamera.end(), eopCamera[indexPose]);
                int indexSensor = std::distance(iopCamera.begin(),it);
                // std::cout<<"indexSensor: "<<indexSensor<<", ID: "<< eopCamera[indexPose]<<std::endl; 

                it = std::find(ropSlave.begin(), ropSlave.end(), iopCamera[indexSensor]);
                int indexROPSlave = std::distance(ropSlave.begin(),it);
                // std::cout<<"indexROPSlave: "<<indexROPSlave<<", ID: "<< iopCamera[indexSensor]<<std::endl; 

                if (it!=ropSlave.end() && iopCamera[indexSensor] == *it) // is a slave in ROP constraint
                {
                    //std::cout<<"  Doing weighted ROP constraint..."<<std::endl;

                    it = std::find(eopStation.begin(), eopStation.end(), eopStation[n] - ropID[indexROPSlave][2]);
                    int indexPoseMaster = std::distance(eopStation.begin(),it);

                // // print the indices for debugging purposes
                // std::cout<<"indexPose: "<<indexPose<<", ID: "<< eopStation[n]<<std::endl;
                // std::cout<<"indexPoseMaster: "<<indexPoseMaster<<", ID: "<< eopStation[indexPoseMaster]<<std::endl;

                    // double deltaOmegaStdDev = 1000.0 * PI / 180.0; //rad
                    // double deltaPhiStdDev   = 1000.0 * PI / 180.0;
                    // double deltaKappaStdDev = 1000.0 * PI / 180.0;
                    // double deltaXoStdDev    = 1000.0;  // mm
                    // double deltaYoStdDev    = 1000.0;
                    // double deltaZoStdDev    = 1000.0;

                    double deltaOmegaStdDev = 0.05 * PI / 180.0; //rad
                    double deltaPhiStdDev   = 0.05 * PI / 180.0;
                    double deltaKappaStdDev = 0.05 * PI / 180.0;
                    double deltaXoStdDev    = 500.0;  // mm
                    double deltaYoStdDev    = 500.0;
                    double deltaZoStdDev    = 500.0;

                    ceres::CostFunction* cost_function =
                        new ceres::AutoDiffCostFunction<ropConstraint, 6, 6, 6, 6>(
                            new ropConstraint(deltaOmegaStdDev, deltaPhiStdDev, deltaKappaStdDev, deltaXoStdDev, deltaYoStdDev, deltaZoStdDev));
                    problem.AddResidualBlock(cost_function, loss, &EOP[indexPoseMaster][0], &EOP[indexPose][0], &ROP[indexROPSlave][0]);  

                    variances.push_back(deltaOmegaStdDev*deltaOmegaStdDev);
                    variances.push_back(deltaPhiStdDev*deltaPhiStdDev);
                    variances.push_back(deltaKappaStdDev*deltaKappaStdDev);
                    variances.push_back(deltaXoStdDev*deltaXoStdDev);
                    variances.push_back(deltaYoStdDev*deltaYoStdDev);
                    variances.push_back(deltaZoStdDev*deltaZoStdDev);

                    //problem.SetParameterBlockConstant(&ROP[indexROPSlave][0]);


                    // Test the ROP calculation
                    //   // rotation from map to sensor 1
                    //   double a11 = cos(EOP[indexPoseMaster][1]) * cos(EOP[indexPoseMaster][2]);
                    //   double a12 = cos(EOP[indexPoseMaster][0]) * sin(EOP[indexPoseMaster][2]) + sin(EOP[indexPoseMaster][0]) * sin(EOP[indexPoseMaster][1]) * cos(EOP[indexPoseMaster][2]);
                    //   double a13 = sin(EOP[indexPoseMaster][0]) * sin(EOP[indexPoseMaster][2]) - cos(EOP[indexPoseMaster][0]) * sin(EOP[indexPoseMaster][1]) * cos(EOP[indexPoseMaster][2]);

                    //   double a21 = -cos(EOP[indexPoseMaster][1]) * sin(EOP[indexPoseMaster][2]);
                    //   double a22 = cos(EOP[indexPoseMaster][0]) * cos(EOP[indexPoseMaster][2]) - sin(EOP[indexPoseMaster][0]) * sin(EOP[indexPoseMaster][1]) * sin(EOP[indexPoseMaster][2]);
                    //   double a23 = sin(EOP[indexPoseMaster][0]) * cos(EOP[indexPoseMaster][2]) + cos(EOP[indexPoseMaster][0]) * sin(EOP[indexPoseMaster][1]) * sin(EOP[indexPoseMaster][2]);

                    //   double a31 = sin(EOP[indexPoseMaster][1]);
                    //   double a32 = -sin(EOP[indexPoseMaster][0]) * cos(EOP[indexPoseMaster][1]);
                    //   double a33 = cos(EOP[indexPoseMaster][0]) * cos(EOP[indexPoseMaster][1]); 

                    //   // rotation from map to sensor 2
                    //   double b11 = cos(EOP[indexPose][1]) * cos(EOP[indexPose][2]);
                    //   double b12 = cos(EOP[indexPose][0]) * sin(EOP[indexPose][2]) + sin(EOP[indexPose][0]) * sin(EOP[indexPose][1]) * cos(EOP[indexPose][2]);
                    //   double b13 = sin(EOP[indexPose][0]) * sin(EOP[indexPose][2]) - cos(EOP[indexPose][0]) * sin(EOP[indexPose][1]) * cos(EOP[indexPose][2]);

                    //   double b21 = -cos(EOP[indexPose][1]) * sin(EOP[indexPose][2]);
                    //   double b22 = cos(EOP[indexPose][0]) * cos(EOP[indexPose][2]) - sin(EOP[indexPose][0]) * sin(EOP[indexPose][1]) * sin(EOP[indexPose][2]);
                    //   double b23 = sin(EOP[indexPose][0]) * cos(EOP[indexPose][2]) + cos(EOP[indexPose][0]) * sin(EOP[indexPose][1]) * sin(EOP[indexPose][2]);

                    //   double b31 = sin(EOP[indexPose][1]);
                    //   double b32 = -sin(EOP[indexPose][0]) * cos(EOP[indexPose][1]);
                    //   double b33 = cos(EOP[indexPose][0]) * cos(EOP[indexPose][1]); 

                    //   // rotation from sensor 2 to sensor 1
                    //   double r11 = cos(ROP[indexROPSlave][1]) * cos(ROP[indexROPSlave][2]);
                    //   double r21 = cos(ROP[indexROPSlave][0]) * sin(ROP[indexROPSlave][2]) + sin(ROP[indexROPSlave][0]) * sin(ROP[indexROPSlave][1]) * cos(ROP[indexROPSlave][2]);
                    //   double r31 = sin(ROP[indexROPSlave][0]) * sin(ROP[indexROPSlave][2]) - cos(ROP[indexROPSlave][0]) * sin(ROP[indexROPSlave][1]) * cos(ROP[indexROPSlave][2]);

                    //   double r12 = -cos(ROP[indexROPSlave][1]) * sin(ROP[indexROPSlave][2]);
                    //   double r22 = cos(ROP[indexROPSlave][0]) * cos(ROP[indexROPSlave][2]) - sin(ROP[indexROPSlave][0]) * sin(ROP[indexROPSlave][1]) * sin(ROP[indexROPSlave][2]);
                    //   double r32 = sin(ROP[indexROPSlave][0]) * cos(ROP[indexROPSlave][2]) + cos(ROP[indexROPSlave][0]) * sin(ROP[indexROPSlave][1]) * sin(ROP[indexROPSlave][2]);

                    //   double r13 = sin(ROP[indexROPSlave][1]);
                    //   double r23 = -sin(ROP[indexROPSlave][0]) * cos(ROP[indexROPSlave][1]);
                    //   double r33 = cos(ROP[indexROPSlave][0]) * cos(ROP[indexROPSlave][1]); 

                    //   // R_1To2 = R_mTo2 * R_1Tom 
                    //   double m11 = b11 * a11 + b12 * a12 + b13 * a13;
                    //   double m12 = b11 * a21 + b12 * a22 + b13 * a23;
                    //   double m13 = b11 * a31 + b12 * a32 + b13 * a33;

                    //   double m21 = b21 * a11 + b22 * a12 + b23 * a13;
                    //   double m22 = b21 * a21 + b22 * a22 + b23 * a23;
                    //   double m23 = b21 * a31 + b22 * a32 + b23 * a33;

                    //   double m31 = b31 * a11 + b32 * a12 + b33 * a13;
                    //   double m32 = b31 * a21 + b32 * a22 + b33 * a23;
                    //   double m33 = b31 * a31 + b32 * a32 + b33 * a33;

                    //     double Tx = EOP[indexPose][3] - EOP[indexPoseMaster][3];
                    //     double Ty = EOP[indexPose][4] - EOP[indexPoseMaster][4];
                    //     double Tz = EOP[indexPose][5] - EOP[indexPoseMaster][5];

                    //     // I = boresight_2To1 * R_1To2
                    //     double deltaR32 = r31*m12 + r32*m22 + r33*m32;
                    //     double deltaR33 = r31*m13 + r32*m23 + r33*m33;
                    //     double deltaR31 = r31*m11 + r32*m21 + r33*m31;
                    //     double deltaR21 = r21*m11 + r22*m21 + r23*m31;
                    //     double deltaR11 = r11*m11 + r12*m21 + r13*m31;

                    //     // T deltaR22 = r21*m12 + r22*m22 + r23*m32;

                    //     double deltaOmega = atan2(-deltaR32, deltaR33);
                    //     double deltaPhi   = asin(deltaR31);
                    //     double deltaKappa = atan2(-deltaR21, deltaR11);

                    //     // defined in the coordinate frame of sensor 1
                    //     double bx = a11*Tx + a12*Ty + a13*Tz;
                    //     double by = a21*Tx + a22*Ty + a23*Tz;
                    //     double bz = a31*Tx + a32*Ty + a33*Tz;

                    //     std::cout<<"bx: "<<bx<<", by: "<<by<<", bz: "<<bz<<std::endl;
                    //     std::cout<<"Tx: "<<Tx<<", Ty: "<<Ty<<", Tz: "<<Tz<<std::endl;
                    //     std::cout<<"ROP: "<<ROP[indexROPSlave][3]<<", ROP: "<<ROP[indexROPSlave][4]<<", ROP: "<<ROP[indexROPSlave][5]<<std::endl;

                    //   // actual cost function
                    //   std::cout<<"ROP: " <<deltaOmega<<", "<< deltaPhi<<", "<< deltaKappa<<", "<<bx - ROP[indexROPSlave][3]<<", "<<by - ROP[indexROPSlave][4]<<", "<<bz - ROP[indexROPSlave][5]<<std::endl;


                }
            }        
        }



        /////////////////////////////////////////////////////////
        /////////////////////////////////////////////////////////
        /// Absolute constraints
        /////////////////////////////////////////////////////////
        /////////////////////////////////////////////////////////

        // if(true)
        // {
        //     for(int n = 0; n < iopCamera.size(); n++)
        //     {
        //         // Fix part of IOPs instead of all
        //         std::vector<int> fixIOP;
        //         fixIOP.push_back(0); //xp
        //         fixIOP.push_back(1); //yp
        //         fixIOP.push_back(2); //c
        //         ceres::SubsetParameterization* subset_parameterization = new ceres::SubsetParameterization(3, fixIOP);
        //         problem.SetParameterization(&IOP[n][0], subset_parameterization);
        //     }
        // }

        // if(true)
        // {   
        //     // Does not work with Cv estimations. Switch to a strong prior to disable APs if need Cv information
        //     std::cout<<"   Fixing a subset of the AP"<<std::endl;
        //     std::cout<<"      When using this mode cannot esimate Cv, so please disable"<<std::endl;
        //     for(int n = 0; n < iopCamera.size(); n++)
        //     {
        //         // Fix part of APs instead of all
        //         std::vector<int> fixAP;
        //         fixAP.push_back(0); //a1
        //         fixAP.push_back(1); //a2
        //         fixAP.push_back(2); //k1
        //         fixAP.push_back(3); //k2
        //         fixAP.push_back(4); //k3
        //         fixAP.push_back(5); //p1
        //         fixAP.push_back(6); //p2

        //         ceres::SubsetParameterization* subset_parameterization = new ceres::SubsetParameterization(7, fixAP);
        //         problem.SetParameterization(&AP[n][0], subset_parameterization);
        //     }
        // }

        // if (true)
        // {
        //     problem.SetParameterBlockConstant(&XYZ[0][0]);
        //     problem.SetParameterBlockConstant(&XYZ[10][0]);
        // }

        // if(true)
        // {
        //         // Fix part of IOPs instead of all
        //         std::vector<int> fixXYZ;
        //         fixXYZ.push_back(0); 
        //         // fixXYZ.push_back(1); 
        //         // fixXYZ.push_back(2); 

        //         ceres::SubsetParameterization* subset_parameterization = new ceres::SubsetParameterization(3, fixXYZ);
        //         problem.SetParameterization(&XYZ[0][0], subset_parameterization);
        // }
        // if(true)
        // {
        //         // Fix part of IOPs instead of all
        //         std::vector<int> fixXYZ;
        //         fixXYZ.push_back(1); 
        //         // fixXYZ.push_back(2); 
        //         ceres::SubsetParameterization* subset_parameterization = new ceres::SubsetParameterization(3, fixXYZ);
        //         problem.SetParameterization(&XYZ[2][0], subset_parameterization);
        // }
        // if(true)
        // {
        //         // Fix part of IOPs instead of all
        //         std::vector<int> fixXYZ;
        //         fixXYZ.push_back(2); 
        //         ceres::SubsetParameterization* subset_parameterization = new ceres::SubsetParameterization(3, fixXYZ);
        //         problem.SetParameterization(&XYZ[10][0], subset_parameterization);
        // }
        // if(true)
        // {
        //         // Fix part of IOPs instead of all
        //         std::vector<int> fixXYZ;
        //         fixXYZ.push_back(0); 
        //         ceres::SubsetParameterization* subset_parameterization = new ceres::SubsetParameterization(3, fixXYZ);
        //         problem.SetParameterization(&XYZ[12][0], subset_parameterization);
        // }
        // if(true)
        // {
        //         // Fix part of IOPs instead of all
        //         std::vector<int> fixXYZ;
        //         fixXYZ.push_back(1); 
        //         ceres::SubsetParameterization* subset_parameterization = new ceres::SubsetParameterization(3, fixXYZ);
        //         problem.SetParameterization(&XYZ[14][0], subset_parameterization);
        // }
        // if(true)
        // {
        //         // Fix part of IOPs instead of all
        //         std::vector<int> fixXYZ;
        //         fixXYZ.push_back(2); 
        //         ceres::SubsetParameterization* subset_parameterization = new ceres::SubsetParameterization(3, fixXYZ);
        //         problem.SetParameterization(&XYZ[16][0], subset_parameterization);
        // }
        // if(true)
        // {
        //         // Fix part of IOPs instead of all
        //         std::vector<int> fixXYZ;
        //         fixXYZ.push_back(0); 
        //         ceres::SubsetParameterization* subset_parameterization = new ceres::SubsetParameterization(3, fixXYZ);
        //         problem.SetParameterization(&XYZ[8][0], subset_parameterization);
        // }

        /////////////////////////////////////////////////////////
        /////////////////////////////////////////////////////////
        /// weighted constraints
        /////////////////////////////////////////////////////////
        /////////////////////////////////////////////////////////
        // define the datum by pseduo observations of the positions for defining the datum
        if(true)
        {
            for(int n = 0; n < xyzTarget.size(); n++)
            {
                xyzXStdDev[n] /= 100.0; //only used for debugging
                xyzYStdDev[n] /= 100.0;
                xyzZStdDev[n] /= 100.0;

                ceres::CostFunction* cost_function =
                    new ceres::AutoDiffCostFunction<constrainPoint, 3, 3>(
                        new constrainPoint(xyzX[n], xyzY[n], xyzZ[n], xyzXStdDev[n], xyzYStdDev[n], xyzZStdDev[n]));
                problem.AddResidualBlock(cost_function, NULL, &XYZ[n][0]);

                variances.push_back(xyzXStdDev[n]*xyzXStdDev[n]);
                variances.push_back(xyzYStdDev[n]*xyzYStdDev[n]);
                variances.push_back(xyzZStdDev[n]*xyzZStdDev[n]);
            }
        }

        // // prior on the IOP. Useful for X-ray data
        // if (true)
        // {
        //     for(int n = 0; n < iopCamera.size(); n++)
        //     {
        //         // double xpStdDev = 10.0;
        //         // double ypStdDev = 10.0;
        //         // double cStdDev  = 10.0;
        //         double xpStdDev = 1E-6;
        //         double ypStdDev = 1E-6;
        //         double cStdDev  = 1E-6;
        //         ceres::CostFunction* cost_function =
        //             new ceres::AutoDiffCostFunction<constrainPoint, 3, 3>(
        //                 new constrainPoint(iopXp[n], iopYp[n], iopC[n], xpStdDev, ypStdDev, cStdDev));
        //         problem.AddResidualBlock(cost_function, NULL, &IOP[n][0]);

        //         variances.push_back(xpStdDev*xpStdDev);
        //         variances.push_back(ypStdDev*ypStdDev);
        //         variances.push_back(cStdDev*cStdDev);
        //     }
        // }


        // // prior on the AP
        // if (true)
        // {
        //     for(int n = 0; n < iopCamera.size(); n++)
        //     {
        //         double a1StdDev  = 1.0E-6;
        //         double a2StdDev  = 1.0E-6;
        //         double k1StdDev  = 1.0E-6;
        //         double k2StdDev  = 1.0E-6;
        //         double k3StdDev  = 1.0E-6;
        //         double p1StdDev  = 1.0E-6;
        //         double p2StdDev  = 1.0E-6;

        //         ceres::CostFunction* cost_function =
        //             new ceres::AutoDiffCostFunction<constrainAP, 7, 7>(
        //                 new constrainAP(iopA1[n], iopA2[n], iopK1[n], iopK2[n], iopK3[n], iopP1[n], iopP2[n], a1StdDev, a2StdDev, k1StdDev, k2StdDev, k3StdDev, p1StdDev, p2StdDev));
        //         problem.AddResidualBlock(cost_function, NULL, &AP[n][0]);

        //         variances.push_back(a1StdDev*a1StdDev);
        //         variances.push_back(a2StdDev*a2StdDev);
        //         variances.push_back(k1StdDev*k1StdDev);
        //         variances.push_back(k2StdDev*k2StdDev);
        //         variances.push_back(k3StdDev*k3StdDev);
        //         variances.push_back(p1StdDev*p1StdDev);
        //         variances.push_back(p2StdDev*p2StdDev);

        //     }
        // }

        // if(DEBUGMODE)
        // {
        //     for(int n = 0; n < X.size(); n++)
        //     {
        //         ceres::CostFunction* cost_function =
        //             new ceres::AutoDiffCostFunction<constantConstraint, 1, 1>(
        //                 new constantConstraint(0, 1.0/(double(n+1)*pow(10.0,-4.0))));
        //         problem.AddResidualBlock(cost_function, NULL, &X[n]);
        //     }
        // }

        // for(int n = 0; n < X.size(); n++)
        // std::cout<<"before X: "<<X[n]<<std::endl;

        PyRun_SimpleString("print( 'Done building Ceres-Solver cost functions:', round(TIME.process_time()-t0, 3), 's' )");

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; // sparse solver
        // options.linear_solver_type = ceres::DENSE_QR;
        // options.linear_solver_type = ceres::SPARSE_SCHUR;
        // options.linear_solver_type = ceres::CGNR;
        // options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        options.max_lm_diagonal = 1.0E-150; // force it behave like a Gauss-Newton update
        options.min_lm_diagonal = 1.0E-150;
        // options.minimizer_type = ceres::LINE_SEARCH;
        // options.line_search_direction_type = ceres::BFGS;
        // options.trust_region_strategy_type = ceres::DOGLEG;
        // options.max_num_iterations = 1000;
        options.max_num_iterations = 100;
        // options.max_num_iterations = 10;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << "\n";
        std::cout << summary.FullReport() << "\n";
       
        // condition for terminating the global ML least squares bundle adjustment routine
        // if ( leastSquaresCost.size() > 1 && (leastSquaresCost[leastSquaresCost.size()-1]) > (leastSquaresCost[leastSquaresCost.size()-2]) )
        if ( leastSquaresCost.size() > 50 && (summary.final_cost) > (leastSquaresCost[leastSquaresCost.size()-1]) )
        {
            std::cout<<"-------------------------!!!!!!Machine Learning Bundle Adjustment CONVERGED!!!!!!-------------------------"<<std::endl;
            // std::cout<<"LSA Cost Increased: "<<(leastSquaresCost[leastSquaresCost.size()-1])<< " > " << (leastSquaresCost[leastSquaresCost.size()-2]) <<std::endl;
            std::cout<<"  LSA Cost Increased: "<<(summary.final_cost)<< " > " << (leastSquaresCost[leastSquaresCost.size()-1]) <<std::endl;
            break;
        }

        // storing it for comparison in this EM like routine
        leastSquaresCost.push_back(summary.final_cost);
        double aposterioriVarianceImageSpace = 1.0;
        double aposterioriVarianceObjectSpace = 1.0;

        //////////////////////////////////////////
        /// Computer the incidence angle
        //////////////////////////////////////////
                // Stereographic collinearity condition, no machine learning
        if (true)
        {
            std::cout<<"  Computing final incidence angle..."<<std::endl;
            std::cout<<"     Writing refraction vs incidence angles to file..."<<std::endl;
            FILE *fout = fopen("angles.jck", "w");
            for(int n = 0; n < imageX.size(); n++) // loop through all observations
            {
                std::vector<int>::iterator it;
                it = std::find(xyzTarget.begin(), xyzTarget.end(), imageTarget[n]);
                int indexPoint = std::distance(xyzTarget.begin(),it);
                //  std::cout<<"indexPoint: "<<indexPoint<<", ID: "<< imageTarget[n]<<std::endl;

                it = std::find(eopStation.begin(), eopStation.end(), imageStation[n]);
                int indexPose = std::distance(eopStation.begin(),it);
                //  std::cout<<"index: "<<indexPose<<", ID: "<< imageStation[n]<<std::endl;

                it = std::find(iopCamera.begin(), iopCamera.end(), eopCamera[indexPose]);
                int indexSensor = std::distance(iopCamera.begin(),it);
                // std::cout<<"index: "<<indexSensor<<", ID: "<< eopCamera[indexPose]<<std::endl;   

                double alpha = incidenceAngle(&EOP[indexPose][0], &XYZ[indexPoint][0]);
                double beta  = refractionAngle(imageX[n], imageY[n], &IOP[indexSensor][0], &AP[indexSensor][0]);

                fprintf(fout, "%i %i %i %.6lf %.6lf\n", imageTarget[n], imageStation[n], eopCamera[indexPose], beta*180.0/PI, alpha*180.0/PI );
            }
            fclose(fout);
        }

        /////////////
        // Ad-hoc fix
        // When fixing a subset of AP the covariance matrix cannot be calculated, therefore output EOP and IOP without the covariances
        if (true)
        {
            std::cout<<"  Writing EOPs to file..."<<std::endl;
            FILE *fout = fopen("EOP.jck", "w");
            for(int i = 0; i < EOP.size(); ++i)
            {
                fprintf(fout, "%i %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf\n", eopStation[i], EOP[i][0]*180.0/PI, EOP[i][1]*180.0/PI, EOP[i][2]*180.0/PI, EOP[i][3], EOP[i][4], EOP[i][5] );
            }
            fclose(fout);
        }

        if (true)
        {
            std::cout<<"  Writing IOPs to file..."<<std::endl;
            FILE *fout = fopen("iop.jck", "w");
            for(int i = 0; i < IOP.size(); ++i)
            {
                fprintf(fout, "%i %.6lf %.6lf %.6lf\n", iopCamera[i], IOP[i][0], IOP[i][1], IOP[i][2] );
            }
            fclose(fout);
        }

        if (true)
        {
            
            std::cout<<"  Writing IOPs (xp, yp, c) to screen..."<<std::endl;
            std::cout<<"       Sensor " << iopCamera[0]<<": "<< IOP[0][0]<<", "<< IOP[0][1]<<", "<< IOP[0][2]<<std::endl;
        }

        if (true)
        {
            
            std::cout<<"  Writing APs (a1, a2, k1, k2, k3, p1, p2) to screen..."<<std::endl;
            std::cout<<"       Sensor " << iopCamera[0]<<": "<< AP[0][0]<<", "<< AP[0][1]<<", "<< AP[0][2]<<", "<< AP[0][3]<<", "<< AP[0][4]<<", "<< AP[0][5]<<", "<< AP[0][6] <<std::endl;
        }


        if (true)
        {
            std::vector<double> residuals;
            ceres::CRSMatrix jacobian;
            problem.Evaluate(ceres::Problem::EvaluateOptions(), NULL, &residuals, NULL, NULL);

            Eigen::MatrixXd imageResiduals(imageX.size(), 2);
            for (int n = 0; n<imageX.size(); n++)
            {
                imageResiduals(n,0) = residuals[2*n] * imageXStdDev[n];
                imageResiduals(n,1) = residuals[2*n+1] * imageYStdDev[n];
            }
            
                // std::cout<<"  Writing residuals to file..."<<std::endl;
                // FILE *fout = fopen("image.jck", "w");
                // for(int i = 0; i < imageTarget.size(); ++i)
                // {
                //     fprintf(fout, "%i %.6lf %.6lf %.6lf %.6lf\n", sensorReferenceID[i], imageX[i], imageY[i], imageResiduals(i,0), imageResiduals(i,1));
                // }
                // fclose(fout);       

            // assumes the pseudo-observations (normal prior) on the XYZ position is the last cost functions we add
            Eigen::MatrixXd XYZResiduals(XYZ.size(), 3);
            for (int n = 0; n<XYZ.size(); n++)
            {
                int startIndex = residuals.size() - 3 * XYZ.size();
                XYZResiduals(n,0) = residuals[startIndex + 3*n];
                XYZResiduals(n,1) = residuals[startIndex + 3*n+1];
                XYZResiduals(n,2) = residuals[startIndex + 3*n+2];
            }


            if (true) // compute the a posteriori variance factor for scaling the Cx later
            {
                std::cout<<"  Estimating A Posterior Variance Factor..."<<std::endl;
                // Eigen::SparseMatrix<double> P;
                // P.resize(variances.size(), variances.size());
                // std::vector<Eigen::Triplet<double> > tripletP(variances.size());
                // int indexTripletP= 0;
                // for (int i = 0; i < variances.size(); i++)
                // {
                //     tripletP[indexTripletP] = Eigen::Triplet<double>(i, i, 1.0 / variances[i]);
                //     indexTripletP++;
                // }
                // P.setFromTriplets(tripletP.begin(), tripletP.end());

                // Eigen::VectorXd v = Eigen::VectorXd::Map(&residuals[0],residuals.size());
                // //std::cout<<"size: "<<v.size()<<std::endl;
                // Eigen::MatrixXd vTPv = v.transpose() * v;
                // //std::cout<<"size: "<<aposteriorVariance.size()<<std::endl;

                // std::cout<<"     vTPv: "<<vTPv(0,0)<<std::endl;
                // std::cout<<"     sqrt(vTPv): "<<sqrt(vTPv(0,0))<<std::endl;

                // approximate the redundancy as 2*numImagePts - 6*EOP -3*XYZ - 7DatumPoints, this ignores #AP, IOP and pseudo obs
                // double redundancy = 2*imageX.size() - 6*imageFrameID.size() - 3*imageTargetID.size() - 7;
                // std::cout<<"     Estimated Redundancy: "<<redundancy<<std::endl;
                std::cout<<"     CERES GLOBAL A Posteriori Variance (using the robust weights): "<<std::endl;
                double redundancy =  summary.num_residuals_reduced - summary.num_parameters_reduced;
                std::cout<<"        Ceres Error: "<<2*summary.final_cost<<std::endl;
                std::cout<<"        Ceres Redundancy: "<<redundancy<<std::endl;
                double aposterioriVariance = 2*summary.final_cost / redundancy;
                double aposterioriStdDev = sqrt(aposterioriVariance);
                std::cout<<"        Ceres A Posteriori Variance: "<<aposterioriVariance<<std::endl;
                std::cout<<"        Ceres A Posteriori Std Dev: "<<aposterioriStdDev<<std::endl;

                std::cout<<"        AIC: "<<calculateAIC(summary.num_residuals_reduced, 2*summary.final_cost, summary.num_parameters_reduced)<<std::endl;
                std::cout<<"        BIC: "<<calculateBIC(summary.num_residuals_reduced, 2*summary.final_cost, summary.num_parameters_reduced)<<std::endl;

                Eigen::VectorXd v = Eigen::VectorXd::Map(&residuals[0],residuals.size());
                // std::cout<<"size: "<<v.size()<<std::endl;
                // std::cout<<"image size: "<<imageX.size()<<std::endl;

                Eigen::MatrixXd vTPv = v.transpose() * v;
                //std::cout<<"size: "<<aposteriorVariance.size()<<std::endl;
                std::cout<<"     MANUAL GLOBAL A Posteriori Variance (assumes normal weights): "<<std::endl;
                std::cout<<"        vTPv: "<<vTPv(0,0)<<std::endl;
                std::cout<<"        Approximate dof (2*img - 6*EOP - 3*XYZ + 7datum): "<< 2*imageX.size() - 6*imageFrameID.size() - 3*imageTargetID.size() + 7 << std::endl;
                std::cout<<"        vTPv/dof: "<<vTPv(0,0)/(2*imageX.size() - 6*imageFrameID.size() - 3*imageTargetID.size() + 7)<<std::endl;
                std::cout<<"        sqrt(vTPv/dof): "<<sqrt(vTPv(0,0)/(2*imageX.size() - 6*imageFrameID.size() - 3*imageTargetID.size() + 7))<<std::endl;
                std::cout<<"        vTPv/ceresRedundancy: "<<vTPv(0,0)/redundancy<<std::endl;
                std::cout<<"        sqrt(vTPv/ceresRedundancy): "<<sqrt(vTPv(0,0)/redundancy)<<std::endl;

                std::cout<<"     Image A Posteriori Variance: "<<std::endl;
                vTPv = v.topRows(2*imageX.size()).transpose() * v.topRows(2*imageX.size());
                std::cout<<"        vTPv: "<<vTPv(0,0)<<std::endl;
                std::cout<<"        Approx dof: "<<(2*imageX.size() - 6*imageFrameID.size() - 3*imageTargetID.size() + 7)<<std::endl;
                std::cout<<"        Approx a posteriori variance image (used for scaling Cx = apostVar*Qx): "<<vTPv(0,0)/(2*imageX.size() - 6*imageFrameID.size() - 3*imageTargetID.size() + 7)<<std::endl;
                std::cout<<"        Approx a posteriori std dev image: "<<sqrt(vTPv(0,0)/(2*imageX.size() - 6*imageFrameID.size() - 3*imageTargetID.size() + 7))<<std::endl;

                aposterioriVarianceImageSpace = vTPv(0,0)/(2*imageX.size() - 6*imageFrameID.size() - 3*imageTargetID.size());

                std::cout<<"     XYZ A Posteriori Variance: "<<std::endl;
                vTPv = v.bottomRows(3*XYZ.size()).transpose() * v.bottomRows(3*XYZ.size());
                std::cout<<"        vTPv: "<<vTPv(0,0)<<std::endl;
                std::cout<<"        Approx dof: "<<(3*XYZ.size())<<std::endl;
                std::cout<<"        Approx a posteriori variance XYZ (used for scaling Cx = apostVar*Qx): "<<vTPv(0,0)/(3*XYZ.size())<<std::endl;
                std::cout<<"        Approx a posteriori std dev XYZ: "<<sqrt(vTPv(0,0)/(3*XYZ.size()))<<std::endl;

                
                aposterioriVarianceObjectSpace = vTPv(0,0)/(3*XYZ.size());


                // The above calculation does not work, leave it as 1.0
                aposterioriVarianceImageSpace = 1.0;
                aposterioriVarianceObjectSpace = 1.0;
            }
        }























































        
        //////////////////////////////////////////////////
        /// Start doing covariance matrix calculation
        //////////////////////////////////////////////////

        Eigen::MatrixXd xyzVariance(XYZ.size(),3);
        Eigen::MatrixXd eopVariance(EOP.size(),6);
        Eigen::MatrixXd iopVariance(IOP.size(),3);
        Eigen::MatrixXd apVariance(AP.size(),7);
        Eigen::MatrixXd mlpVariance(MLP.size(),2);
        Eigen::MatrixXd ropVariance(ROP.size(),6);

        xyzVariance.setConstant(1E6);
        eopVariance.setConstant(1E6);
        iopVariance.setConstant(1E6);
        apVariance.setConstant(1E6);
        mlpVariance.setConstant(1E6);
        ropVariance.setConstant(1E6);

        Eigen::MatrixXd Cx(summary.num_parameters,summary.num_parameters);
        Cx.setZero();

        if (COMPUTECX)
        {
            PyRun_SimpleString("t0 = TIME.process_time()");     
            PyRun_SimpleString("print( 'Start computing cofactor matrix' )");  
            ceres::Covariance::Options covarianceOptions;
            covarianceOptions.apply_loss_function = true;
            // covarianceOptions.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
            covarianceOptions.algorithm_type = ceres::DENSE_SVD;
            covarianceOptions.null_space_rank = -1;
            ceres::Covariance covariance(covarianceOptions);

            std::vector<std::pair<const double*, const double*> > covariance_blocks;

            // Estimating the main and most useful variances; only the main block diagonal
            if (false)
            {
                for(int i = 0; i < EOP.size(); i++)
                    covariance_blocks.push_back(std::make_pair(&EOP[i][0], &EOP[i][0])); // do 6x6 block diagonal of the XYZ object space target points

                for(int i = 0; i < XYZ.size(); i++)
                    covariance_blocks.push_back(std::make_pair(&XYZ[i][0], &XYZ[i][0])); // do 3x3 block diagonal of the XYZ object space target points

                for(int i = 0; i < IOP.size(); i++)
                    covariance_blocks.push_back(std::make_pair(&IOP[i][0], &IOP[i][0])); // do 3x3 block diagonal of the XYZ object space target points

                for(int i = 0; i < AP.size(); i++)
                    covariance_blocks.push_back(std::make_pair(&AP[i][0], &AP[i][0])); // do 7x7 block diagonal of the XYZ object space target points

                // in the simple mode, MLP is just a constant not a parameter
                // for(int i = 0; i < MLP.size(); i++)
                //     covariance_blocks.push_back(std::make_pair(&MLP[i][0], &MLP[i][0])); 

                if (ROPMODE || WEIGHTEDROPMODE)
                {
                    for(int i = 0; i < ROP.size(); i++)
                        covariance_blocks.push_back(std::make_pair(&ROP[i][0], &ROP[i][0])); // do 7x7 block diagonal of the XYZ object space target points
                }
            }

            // Estimate variances and covariances within the SAME group
            if (true)
            {

                //       EOP XYZ IOP  AP  MLP
                // EOP    *    
                // XYZ        *   
                // IOP            *    
                // AP                 *    
                // MLP                     *

                for(int i = 0; i < EOP.size(); i++)
                    for(int j = i; j < EOP.size(); j++)
                        covariance_blocks.push_back(std::make_pair(&EOP[i][0], &EOP[j][0]));

                for(int i = 0; i < XYZ.size(); i++)
                    for(int j = i; j < XYZ.size(); j++)
                        covariance_blocks.push_back(std::make_pair(&XYZ[i][0], &XYZ[j][0]));

                for(int i = 0; i < IOP.size(); i++)
                    for(int j = i; j < IOP.size(); j++)
                        covariance_blocks.push_back(std::make_pair(&IOP[i][0], &IOP[j][0]));

                for(int i = 0; i < AP.size(); i++)
                    for(int j = i; j < AP.size(); j++)
                        covariance_blocks.push_back(std::make_pair(&AP[i][0], &AP[j][0]));

                // for(int i = 0; i < MLP.size(); i++)
                //     for(int j = i; j < MLP.size(); j++)
                //         covariance_blocks.push_back(std::make_pair(&MLP[i][0], &MLP[j][0]));

                if (ROPMODE || WEIGHTEDROPMODE)
                {
                    for(int i = 0; i < ROP.size(); i++)
                        for(int j = i; j < ROP.size(); j++)
                            covariance_blocks.push_back(std::make_pair(&ROP[i][0], &ROP[j][0]));
                }
            }

            // Additional covariances between DIFFERENT groups
            if (true)
            {
                //       EOP XYZ IOP  AP  MLP
                // EOP        *   *   *   *
                // XYZ            *   *   *
                // IOP                *   *
                // AP                     *
                // MLP

                // covariances
                for(int i = 0; i < EOP.size(); i++)
                {
                    for(int j = 0; j < XYZ.size(); j++)
                        covariance_blocks.push_back(std::make_pair(&EOP[i][0], &XYZ[j][0]));

                    for(int j = 0; j < IOP.size(); j++)
                        covariance_blocks.push_back(std::make_pair(&EOP[i][0], &IOP[j][0]));

                    for(int j = 0; j < AP.size(); j++)
                        covariance_blocks.push_back(std::make_pair(&EOP[i][0], &AP[j][0]));

                    // for(int j = 0; j < MLP.size(); j++)
                    //     covariance_blocks.push_back(std::make_pair(&EOP[i][0], &MLP[j][0]));

                    if(ROPMODE || WEIGHTEDROPMODE)
                    {
                        for(int j = 0; j < ROP.size(); j++)
                            covariance_blocks.push_back(std::make_pair(&EOP[i][0], &ROP[j][0]));
                    }
                }

                for(int i = 0; i < XYZ.size(); i++)
                {
                    for(int j = 0; j < IOP.size(); j++)
                        covariance_blocks.push_back(std::make_pair(&XYZ[i][0], &IOP[j][0]));

                    for(int j = 0; j < AP.size(); j++)
                        covariance_blocks.push_back(std::make_pair(&XYZ[i][0], &AP[j][0]));

                    // for(int j = 0; j < MLP.size(); j++)
                    //     covariance_blocks.push_back(std::make_pair(&XYZ[i][0], &MLP[j][0]));

                    if(ROPMODE || WEIGHTEDROPMODE)
                    {
                        for(int j = 0; j < ROP.size(); j++)
                            covariance_blocks.push_back(std::make_pair(&XYZ[i][0], &ROP[j][0]));
                    }
                }

                for(int i = 0; i < IOP.size(); i++)
                {
                    for(int j = 0; j < AP.size(); j++)
                        covariance_blocks.push_back(std::make_pair(&IOP[i][0], &AP[j][0]));

                    // for(int j = 0; j < MLP.size(); j++)
                    //     covariance_blocks.push_back(std::make_pair(&IOP[i][0], &MLP[j][0]));
                    if(ROPMODE || WEIGHTEDROPMODE)
                    {
                        for(int j = 0; j < ROP.size(); j++)
                            covariance_blocks.push_back(std::make_pair(&IOP[i][0], &ROP[j][0]));
                    }
                }

                for(int i = 0; i < AP.size(); i++)
                {
                    // for(int j = 0; j < MLP.size(); j++)
                    //     covariance_blocks.push_back(std::make_pair(&AP[i][0], &MLP[j][0]));

                    if(ROPMODE || WEIGHTEDROPMODE)
                    {
                        for(int j = 0; j < ROP.size(); j++)
                            covariance_blocks.push_back(std::make_pair(&AP[i][0], &ROP[j][0]));
                    }
                }
            }

            CHECK(covariance.Compute(covariance_blocks, &problem));

            std::cout<<"   Done Computing Cofactor Matrix Block"<<std::endl;
            // //double covariance_X[X.size() * X.size()];
            // Eigen::MatrixXd covariance_X(XYZ.size(), 3);
            // // covariance_X.resize(X.size() * X.size());
            // //covariance.GetCovarianceBlock(&X[0], &X[0], covariance_X.data());

            // for(int i = 0; i < XYZ.size(); i++)
            //     for(int j = 0; j < XYZ.size(); j++)
            //     {
            //         double temp[9];
            //         //std::cout<<"std: "<<sqrt(temp[0])<<std::endl;
            //         covariance_X(i,0) = temp[0];
            //     }

            // Eigen::MatrixXd xyzVariance(XYZ.size(),3);
            for(int i = 0; i < XYZ.size(); i++)
            {
                // double covariance_xx[3 * 3];
                // covariance.GetCovarianceBlockInTangentSpace(&XYZ[i][0], &XYZ[i][0], covariance_xx);
                // std::cout<<sqrt(covariance_xx[0])<<", "<<sqrt(covariance_xx[4])<<", "<<sqrt(covariance_xx[8])<<std::endl;

                Eigen::MatrixXd covariance_X(3, 3);
                covariance.GetCovarianceBlock(&XYZ[i][0], &XYZ[i][0], covariance_X.data());
                Eigen::VectorXd variance_X(3);
                variance_X = covariance_X.diagonal();
                xyzVariance(i,0) = variance_X(0);
                xyzVariance(i,1) = variance_X(1);
                xyzVariance(i,2) = variance_X(2);

                // std::cout<<covariance_X<<std::endl;
                // sleep(100000);

                xyzVariance(i,0) *= aposterioriVarianceObjectSpace;
                xyzVariance(i,1) *= aposterioriVarianceObjectSpace;
                xyzVariance(i,2) *= aposterioriVarianceObjectSpace;
            //  std::cout << "Variance: " << covariance_X.diagonal() << std::endl; 
            //  std::cout<<"covariance matrix: "<<std::endl;
            //  std::cout<<covariance_X<<std::endl;

                // // store the full variance-covariance matrix
                // for (int n = 0; n < covariance_X.rows(); n++)
                //     for (int m = 0; m < covariance_X.cols(); m++)
                //         Cx(i*3+n + 6*EOP.size(),i*3+m + 6*EOP.size()) = covariance_X(n,m);
            }

            // Eigen::MatrixXd eopVariance(EOP.size(),6);
            for(int i = 0; i < EOP.size(); i++)
            {
                Eigen::MatrixXd covariance_X(6, 6);
                covariance.GetCovarianceBlock(&EOP[i][0], &EOP[i][0], covariance_X.data());
                Eigen::VectorXd variance_X(6);
                variance_X = covariance_X.diagonal();
                eopVariance(i,0) = variance_X(0);
                eopVariance(i,1) = variance_X(1);
                eopVariance(i,2) = variance_X(2);
                eopVariance(i,3) = variance_X(3);
                eopVariance(i,4) = variance_X(4);
                eopVariance(i,5) = variance_X(5);

                eopVariance(i,0) *= aposterioriVarianceObjectSpace;
                eopVariance(i,1) *= aposterioriVarianceObjectSpace;
                eopVariance(i,2) *= aposterioriVarianceObjectSpace;
                eopVariance(i,3) *= aposterioriVarianceObjectSpace;
                eopVariance(i,4) *= aposterioriVarianceObjectSpace;
                eopVariance(i,5) *= aposterioriVarianceObjectSpace;
            }

            // Eigen::MatrixXd iopVariance(IOP.size(),3);
            for(int i = 0; i < IOP.size(); i++)
            {
                Eigen::MatrixXd covariance_X(3, 3);
                covariance.GetCovarianceBlock(&IOP[i][0], &IOP[i][0], covariance_X.data());
                Eigen::VectorXd variance_X(3);
                variance_X = covariance_X.diagonal();
                iopVariance(i,0) = variance_X(0);
                iopVariance(i,1) = variance_X(1);
                iopVariance(i,2) = variance_X(2);


                iopVariance(i,0) *= aposterioriVarianceImageSpace;
                iopVariance(i,1) *= aposterioriVarianceImageSpace;
                iopVariance(i,2) *= aposterioriVarianceImageSpace;
                // // store the full variance-covariance matrix
                // for (int n = 0; n < covariance_X.rows(); n++)
                //     for (int m = 0; m < covariance_X.cols(); m++)
                //         Cx(i*3+n + 6*EOP.size()+3*XYZ.size(),i*3+m + 6*EOP.size()+3*XYZ.size()) = covariance_X(n,m);
            }

            // Eigen::MatrixXd apVariance(AP.size(),7);
            for(int i = 0; i < AP.size(); i++)
            {
                Eigen::MatrixXd covariance_X(7, 7);
                covariance.GetCovarianceBlock(&AP[i][0], &AP[i][0], covariance_X.data());
                Eigen::VectorXd variance_X(7);
                variance_X = covariance_X.diagonal();
                apVariance(i,0) = variance_X(0);
                apVariance(i,1) = variance_X(1);
                apVariance(i,2) = variance_X(2);
                apVariance(i,3) = variance_X(3);
                apVariance(i,4) = variance_X(4);
                apVariance(i,5) = variance_X(5);
                apVariance(i,6) = variance_X(6);

                apVariance(i,0) *= aposterioriVarianceImageSpace;
                apVariance(i,1) *= aposterioriVarianceImageSpace;
                apVariance(i,2) *= aposterioriVarianceImageSpace;
                apVariance(i,3) *= aposterioriVarianceImageSpace;
                apVariance(i,4) *= aposterioriVarianceImageSpace;
                apVariance(i,5) *= aposterioriVarianceImageSpace;
                apVariance(i,6) *= aposterioriVarianceImageSpace;

                std::cout<<"AP Std Dev: "<<sqrt(apVariance(i,0))<<", "<<sqrt(apVariance(i,1))<<", "<<sqrt(apVariance(i,2))<<", "<<sqrt(apVariance(i,3))<<", "<<sqrt(apVariance(i,4))<<", "<<sqrt(apVariance(i,5))<<", "<<sqrt(apVariance(i,6))<<std::endl;
                // // store the full variance-covariance matrix
                // for (int n = 0; n < covariance_X.rows(); n++)
                //     for (int m = 0; m < covariance_X.cols(); m++)
                //         Cx(i*7+n + 6*EOP.size()+3*XYZ.size()+3*IOP.size(),i*7+m + 6*EOP.size()+3*XYZ.size()+3*IOP.size()) = covariance_X(n,m);
            }

            // Eigen::MatrixXd mlpVariance(MLP.size(),2);
            for(int i = 0; i < MLP.size(); i++)
            {
                // Eigen::MatrixXd covariance_X(2, 2);
                // covariance.GetCovarianceBlock(&MLP[i][0], &MLP[i][0], covariance_X.data());
                // Eigen::VectorXd variance_X(2);
                // variance_X = covariance_X.diagonal();
                // mlpVariance(i,0) = variance_X(0);
                // mlpVariance(i,1) = variance_X(1);

                // // store the full variance-covariance matrix
                // for (int n = 0; n < covariance_X.rows(); n++)
                //     for (int m = 0; m < covariance_X.cols(); m++)
                //         Cx(i*2+n + 6*EOP.size()+3*XYZ.size()+3*IOP.size()+7*AP.size(),i*2+m + 6*EOP.size()+3*XYZ.size()+3*IOP.size()+7*AP.size()) = covariance_X(n,m);
            }

            // Eigen::MatrixXd ropVariance(ROP.size(),6);
            if(ROPMODE || WEIGHTEDROPMODE)
            {
                for(int i = 0; i < ROP.size(); i++)
                {
                    Eigen::MatrixXd covariance_X(6, 6);
                    covariance.GetCovarianceBlock(&ROP[i][0], &ROP[i][0], covariance_X.data());
                    Eigen::VectorXd variance_X(6);
                    variance_X = covariance_X.diagonal();
                    ropVariance(i,0) = variance_X(0);
                    ropVariance(i,1) = variance_X(1);
                    ropVariance(i,2) = variance_X(2);
                    ropVariance(i,3) = variance_X(3);
                    ropVariance(i,4) = variance_X(4);
                    ropVariance(i,5) = variance_X(5);

                    ropVariance(i,0) *= aposterioriVarianceObjectSpace;
                    ropVariance(i,1) *= aposterioriVarianceObjectSpace;
                    ropVariance(i,2) *= aposterioriVarianceObjectSpace;
                    ropVariance(i,3) *= aposterioriVarianceObjectSpace;
                    ropVariance(i,4) *= aposterioriVarianceObjectSpace;
                    ropVariance(i,5) *= aposterioriVarianceObjectSpace;
                }
            }

            //       EOP XYZ IOP  AP  MLP  ROP
            // EOP    x   x   x    x   x    x
            // XYZ        x   x    x   x    x
            // IOP            x    x   x    x
            // AP                  x   x    x      
            // MLP                     x    x
            // ROP                          x
            // Eigen::MatrixXd Cx(summary.num_parameters,summary.num_parameters);
            // Cx.setZero();

            if (true)
            {
                // Get the full variance-covariance matrix Cx
                for(int i = 0; i < EOP.size(); i++)
                {
                    for(int j = i; j < EOP.size(); j++)
                    {
                        Eigen::MatrixXd covariance_X(6, 6);
                        covariance.GetCovarianceBlock(&EOP[i][0], &EOP[j][0], covariance_X.transpose().data());

                        // // store the full variance-covariance matrix
                        // for (int n = 0; n < covariance_X.rows(); n++)
                        //     for (int m = 0; m < covariance_X.cols(); m++)
                        //         Cx(i*6+n,i*6+m) = covariance_X(n,m);

                        Cx.block<6,6>(i*6,j*6) = covariance_X.transpose();
                    }

                    for(int j = 0; j < XYZ.size(); j++)
                    {
                        Eigen::MatrixXd covariance_X(3, 6); // note the size is opposite
                        covariance.GetCovarianceBlock(&EOP[i][0], &XYZ[j][0], covariance_X.data()); // what we get is the lower triangle matrix

                        Cx.block<6,3>(i*6,j*3 + 6*EOP.size()) = covariance_X.transpose(); // what we store is the upper triangule matrix
                    }

                    for(int j = 0; j < IOP.size(); j++)
                    {
                        Eigen::MatrixXd covariance_X(3, 6);
                        covariance.GetCovarianceBlock(&EOP[i][0], &IOP[j][0], covariance_X.data());

                        Cx.block<6,3>(i*6,j*3 + 6*EOP.size()+3*XYZ.size()) = covariance_X.transpose();
                    }

                    for(int j = 0; j < AP.size(); j++)
                    {
                        Eigen::MatrixXd covariance_X(7, 6);
                        covariance.GetCovarianceBlock(&EOP[i][0], &AP[j][0], covariance_X.data());

                        Cx.block<6,7>(i*6,j*7 + 6*EOP.size()+3*XYZ.size()+3*IOP.size()) = covariance_X.transpose();
                    }

                    // for(int j = 0; j < MLP.size(); j++)
                    // {
                    //     Eigen::MatrixXd covariance_X(2, 6);
                    //     covariance.GetCovarianceBlock(&EOP[i][0], &MLP[j][0], covariance_X.data());

                    //     Cx.block<6,2>(i*6,j*2 + 6*EOP.size()+3*XYZ.size()+3*IOP.size()+7*AP.size()) = covariance_X.transpose();
                    // }

                    if(ROPMODE || WEIGHTEDROPMODE)
                    {
                        for(int j = 0; j < ROP.size(); j++)
                        {
                            Eigen::MatrixXd covariance_X(6, 6);
                            covariance.GetCovarianceBlock(&EOP[i][0], &ROP[j][0], covariance_X.data());

                            Cx.block<6,6>(i*6,j*6 + 6*EOP.size()+3*XYZ.size()+3*IOP.size()+7*AP.size()) = covariance_X.transpose();
                        }
                            
                    }
                }

                for(int i = 0; i < XYZ.size(); i++)
                {
                    for(int j = i; j < XYZ.size(); j++)
                    {
                        Eigen::MatrixXd covariance_X(3, 3);
                        covariance.GetCovarianceBlock(&XYZ[i][0], &XYZ[j][0], covariance_X.data());

                        Cx.block<3,3>(i*3 + 6*EOP.size(),j*3 + 6*EOP.size()) = covariance_X.transpose();
                    }

                    for(int j = 0; j < IOP.size(); j++)
                    {
                        Eigen::MatrixXd covariance_X(3, 3);
                        covariance.GetCovarianceBlock(&XYZ[i][0], &IOP[j][0], covariance_X.data());

                        Cx.block<3,3>(i*3 + 6*EOP.size(),j*3 + 6*EOP.size()+3*XYZ.size()) = covariance_X.transpose();
                    }

                    for(int j = 0; j < AP.size(); j++)
                    {
                        Eigen::MatrixXd covariance_X(7, 3);
                        covariance.GetCovarianceBlock(&XYZ[i][0], &AP[j][0], covariance_X.data());

                        Cx.block<3,7>(i*3 + 6*EOP.size(),j*7 + 6*EOP.size()+3*XYZ.size()+3*IOP.size()) = covariance_X.transpose();
                    }

                    // for(int j = 0; j < MLP.size(); j++)
                    // {
                    //     Eigen::MatrixXd covariance_X(2, 3);
                    //     covariance.GetCovarianceBlock(&XYZ[i][0], &MLP[j][0], covariance_X.data());

                    //     Cx.block<3,2>(i*3 + 6*EOP.size(),j*2 + 6*EOP.size()+3*XYZ.size()+3*IOP.size()+7*AP.size()) = covariance_X.transpose();
                    // }

                    if(ROPMODE || WEIGHTEDROPMODE)
                    {
                        for(int j = 0; j < ROP.size(); j++)
                        {
                            Eigen::MatrixXd covariance_X(6, 3);
                            covariance.GetCovarianceBlock(&XYZ[i][0], &ROP[j][0], covariance_X.data());

                            Cx.block<3,6>(i*3 + 6*EOP.size(),j*6 + 6*EOP.size()+3*XYZ.size()+3*IOP.size()+7*AP.size()) = covariance_X.transpose();
                        }    
                    }
                }

                for(int i = 0; i < IOP.size(); i++)
                {
                    for(int j = i; j < IOP.size(); j++)
                    {
                        Eigen::MatrixXd covariance_X(3, 3);
                        covariance.GetCovarianceBlock(&IOP[i][0], &IOP[j][0], covariance_X.data());

                        Cx.block<3,3>(i*3 + 6*EOP.size()+3*XYZ.size(),j*3 + 6*EOP.size()+3*XYZ.size()) = covariance_X.transpose();
                    }

                    for(int j = 0; j < AP.size(); j++)
                    {
                        Eigen::MatrixXd covariance_X(7, 3); 
                        covariance.GetCovarianceBlock(&IOP[i][0], &AP[j][0], covariance_X.data());

                        Cx.block<3,7>(i*3 + 6*EOP.size()+3*XYZ.size(),j*7 + 6*EOP.size()+3*XYZ.size()+3*IOP.size()) = covariance_X.transpose();
                    }

                    // for(int j = 0; j < MLP.size(); j++)
                    // {
                    //     Eigen::MatrixXd covariance_X(2, 3);
                    //     covariance.GetCovarianceBlock(&IOP[i][0], &MLP[j][0], covariance_X.data());

                    //     Cx.block<3,2>(i*3 + 6*EOP.size()+3*XYZ.size(),j*2 + 6*EOP.size()+3*XYZ.size()+3*IOP.size()+7*AP.size()) = covariance_X.transpose();
                    // }

                    if(ROPMODE || WEIGHTEDROPMODE)
                    {
                        for(int j = 0; j < ROP.size(); j++)
                        {
                            Eigen::MatrixXd covariance_X(6, 3);
                            covariance.GetCovarianceBlock(&IOP[i][0], &ROP[j][0], covariance_X.data());

                            Cx.block<3,6>(i*3 + 6*EOP.size()+3*XYZ.size(),j*6 + 6*EOP.size()+3*XYZ.size()+3*IOP.size()+7*AP.size()) = covariance_X.transpose();
                        }
                            
                    }
                }

                for(int i = 0; i < AP.size(); i++)
                {
                    for(int j = i; j < AP.size(); j++)
                    {
                        Eigen::MatrixXd covariance_X(7, 7);
                        covariance.GetCovarianceBlock(&AP[i][0], &AP[j][0], covariance_X.data());

                        Cx.block<7,7>(i*7 + 6*EOP.size()+3*XYZ.size()+3*IOP.size(),j*7 + 6*EOP.size()+3*XYZ.size()+3*IOP.size()) = covariance_X.transpose();
                    }

                    // for(int j = 0; j < MLP.size(); j++)
                    // {
                    //     Eigen::MatrixXd covariance_X(2, 7);
                    //     covariance.GetCovarianceBlock(&AP[i][0], &MLP[j][0], covariance_X.data());

                    //     Cx.block<7,2>(i*7 + 6*EOP.size()+3*XYZ.size()+3*IOP.size(),j*2 + 6*EOP.size()+3*XYZ.size()+3*IOP.size()+7*AP.size()) = covariance_X.transpose();
                    // }

                    if(ROPMODE || WEIGHTEDROPMODE)
                    {
                        for(int j = 0; j < ROP.size(); j++)
                        {
                            Eigen::MatrixXd covariance_X(6, 7);
                            covariance.GetCovarianceBlock(&AP[i][0], &ROP[j][0], covariance_X.data());

                            Cx.block<7,6>(i*7 + 6*EOP.size()+3*XYZ.size()+3*IOP.size(),j*6 + 6*EOP.size()+3*XYZ.size()+3*IOP.size()+7*AP.size()) = covariance_X.transpose();
                        }
                            
                    }
                }

                for(int i = 0; i < MLP.size(); i++)
                {
                    // for(int j = 0; j < MLP.size(); j++)
                    // {
                    //     Eigen::MatrixXd covariance_X(2, 2);
                    //     covariance.GetCovarianceBlock(&MLP[i][0], &MLP[j][0], covariance_X.data());

                    //     Cx.block<2,2>(i*2 + 6*EOP.size()+3*XYZ.size()+3*IOP.size()+7*AP.size(),j*2 + 6*EOP.size()+3*XYZ.size()+3*IOP.size()+7*AP.size()) = covariance_X.transpose();
                    // }
                }

                if(ROPMODE || WEIGHTEDROPMODE)
                {
                    for(int i = 0; i < ROP.size(); i++)
                    {
                        for(int j = 0; j < ROP.size(); j++)
                        {
                            Eigen::MatrixXd covariance_X(6, 6);
                            covariance.GetCovarianceBlock(&ROP[i][0], &ROP[j][0], covariance_X.data());

                            Cx.block<6,6>(i*6 + 6*EOP.size()+3*XYZ.size()+3*IOP.size()+7*AP.size(),j*6 + 6*EOP.size()+3*XYZ.size()+3*IOP.size()+7*AP.size()) = covariance_X.transpose();
                        }
                    }
                }
            }


            if (DEBUGMODE)
            {
                std::cout<<"  Writing Cx_before to file..."<<std::endl;
                FILE *fout = fopen("Cx_before.jck", "w");
                for(int i = 0; i < Cx.rows(); ++i)
                {
                    for(int j = 0; j < Cx.cols(); ++j)
                    {
                        fprintf(fout, "%.6lf \t ", Cx(i,j));
                    }
                    fprintf(fout, "\n");
                }
                fclose(fout);
            }

            // copy it to make a symmetrical matrix
            //Cx.triangularView<Eigen::Lower>() = Cx.transpose();

            if (DEBUGMODE)
            {
                std::cout<<"  Writing Cx to file..."<<std::endl;
                FILE *fout = fopen("Cx.jck", "w");
                for(int i = 0; i < Cx.rows(); ++i)
                {
                    for(int j = 0; j < Cx.cols(); ++j)
                    {
                        fprintf(fout, "%.6lf \t ", Cx(i,j));
                    }
                    fprintf(fout, "\n");
                }
                fclose(fout);
            }

            PyRun_SimpleString("print( 'Done computing covariance matrix:', round(TIME.process_time()-t0, 3), 's' )");
        
        }
        else // if we are not computing the Cx matrix
        {
            std::cout<<"NOT computing Cx covariance matrix"<<std::endl;
        }

        // Output results to screen
        if(DEBUGMODE)
        {
            std::cout<<"Before Adjustment XYZ"<<std::endl;
            std::cout<<"  TargetID: X, Y, Z"<<std::endl;
            for (int n = 0; n < xyzTarget.size(); n++)
            {
                std::cout<<"  "<<xyzTarget[n]<<": "<<xyzX[n]<<", "<<xyzY[n]<<", "<<xyzZ[n]<<std::endl;
            }
            std::cout<<"After Adjustment XYZ"<<std::endl;
            std::cout<<"  TargetID: X, Y, Z, XStdDev, YStdDev, ZStdDev"<<std::endl;
            for (int n = 0; n < XYZ.size(); n++)
            {
                std::cout<<"  "<<xyzTarget[n]<<": "<<XYZ[n][0]<<", "<<XYZ[n][1]<<", "<<XYZ[n][2]<<", "<<sqrt(xyzVariance(n,0))<<", "<<sqrt(xyzVariance(n,1))<<", "<<sqrt(xyzVariance(n,2))<<std::endl;
            }
            std::cout<<"Before Adjustment EOP"<<std::endl;
            std::cout<<"  StationID: roll, pitch, yaw[deg], Xo, Yo, Zo"<<std::endl;
            for (int n = 0; n < eopStation.size(); n++)
            {
                std::cout<<eopStation[n]<<": "<<eopOmega[n]*180.0/PI<<", "<<eopPhi[n]*180.0/PI<<", "<<eopKappa[n]*180.0/PI<<", "<<eopXo[n]<<", "<<eopYo[n]<<", "<<eopZo[n]<<std::endl;
            }
            std::cout<<"After Adjustment EOP"<<std::endl;
            std::cout<<"  StationID: roll, pitch, yaw[deg], Xo, Yo, Zo, rollStdDev, pitchStdDev, yawStdDev[deg], XoStdDev, YoStdDev, ZoStdDev"<<std::endl;
            for (int n = 0; n < EOP.size(); n++)
            {
                std::cout<<eopStation[n]<<": "<<EOP[n][0]*180.0/PI<<", "<<EOP[n][1]*180.0/PI<<", "<<EOP[n][2]*180.0/PI<<", "<<EOP[n][3]<<", "<<EOP[n][4]<<", "<<EOP[n][5]<<", "<<sqrt(eopVariance(n,0))*180.0/PI<<", "<<sqrt(eopVariance(n,1))*180.0/PI<<", "<<sqrt(eopVariance(n,2))*180.0/PI<<", "<<sqrt(eopVariance(n,3))<<", "<<sqrt(eopVariance(n,4))<<", "<<sqrt(eopVariance(n,5))<<std::endl;
            }
            std::cout<<"Before Adjustment IOP"<<std::endl;
            std::cout<<"  CameraID: xp, yp, c"<<std::endl;
            for (int n = 0; n < iopCamera.size(); n++)
            {
                std::cout<<"  "<<iopCamera[n]<<": "<<iopXp[n]<<", "<<iopYp[n]<<", "<<iopC[n]<<std::endl;
            }
            std::cout<<"After Adjustment IOP"<<std::endl;
            std::cout<<"  CameraID: xp, yp, c, xpStdDev, ypStdDev, cStdDev"<<std::endl;
            for (int n = 0; n < iopCamera.size(); n++)
            {
                std::cout<<"  "<<iopCamera[n]<<": "<<IOP[n][0]<<", "<<IOP[n][1]<<", "<<IOP[n][2]<<", "<<sqrt(iopVariance(n,0))<<", "<<sqrt(iopVariance(n,1))<<", "<<sqrt(iopVariance(n,2))<<std::endl;
            }
            std::cout<<"Before Adjustment AP"<<std::endl;
            std::cout<<"  CameraID: a1, a2, k1, k2, k3, p1, p2"<<std::endl;
            for (int n = 0; n < iopCamera.size(); n++)
            {
                std::cout<<"  "<<iopCamera[n]<<": "<<iopA1[n]<<", "<<iopA2[n]<<", "<<iopK1[n]<<", "<<iopK2[n]<<", "<<iopK3[n]<<", "<<iopP1[n]<<", "<<iopP2[n]<<std::endl;
            }
            std::cout<<"After Adjustment AP"<<std::endl;
            std::cout<<"  CameraID: a1, a2, k1, k2, k3, p1, p2, a1StdDev, a2StdDev, k1StdDev, k2StdDev, k3StdDev, p1StdDev, p2StdDev"<<std::endl;
            for (int n = 0; n < iopCamera.size(); n++)
            {
                std::cout<<"  "<<iopCamera[n]<<": "<<AP[n][0]<<", "<<AP[n][1]<<", "<<AP[n][2]<<AP[n][3]<<", "<<AP[n][4]<<", "<<AP[n][5]<<", "<<AP[n][6]<<", "<< sqrt(apVariance(n,0))<<", "<< sqrt(apVariance(n,1))<<", "<< sqrt(apVariance(n,2))<<", "<< sqrt(apVariance(n,3))<<", "<< sqrt(apVariance(n,4))<<", "<< sqrt(apVariance(n,5))<<", "<< sqrt(apVariance(n,6))<<std::endl;
            }
        }

        double cost = 0.0;
        std::vector<double> residuals;
        ceres::CRSMatrix jacobian;
        Eigen::VectorXd redundancyNumber(variances.size());
        // Eigen::MatrixXd Cv; //covariance of the residuals
        Eigen::VectorXd CvDiag;

        // ceres::Problem::EvaluateOptions evaluateOptions;
        // // std::cout<<"Number of param blocks: "<<problem.NumParameterBlocks()<<std::endl;
        // // std::vector<double*> parameterBlocks;
        // // problem.GetParameterBlocks(&parameterBlocks);
        // // evaluateOptions.parameter_blocks = parameterBlocks;
        // evaluateOptions.apply_loss_function = true;
        // problem.Evaluate(evaluateOptions, &cost, &residuals, NULL, &jacobian);

        problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residuals, NULL, &jacobian);

        // problem2.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residuals, NULL, &jacobian);


        // set some default values if we are not actually calculating Cv
        redundancyNumber.setConstant(1E6);
        CvDiag.resize(jacobian.num_rows);
        // CvDiag.setConstant(1E6);
        CvDiag.setConstant(imageXStdDev[0]*imageYStdDev[0]);

        if(COMPUTECV)
        {
            PyRun_SimpleString("t0 = TIME.process_time()");        
            PyRun_SimpleString("print( 'Start computing covariance matrix of the residuals' )");  

            if (true)
            {
                std::cout<<"  Compute jacobian matrix..."<<std::endl;
                // int rowsA []   = { 0,      2,          5,     7};
                // int colsA []   = { 1,  3,  1,  2,  3,  0,  1};
                // double valuesA [] = {10.0,  4.0,  2.0, -3.0,  2.0,  1.0,  2.0};

                // std::vector<int> rows (rowsA, rowsA + sizeof(rowsA) / sizeof(int) );
                // std::vector<int> cols (colsA, colsA + sizeof(colsA) / sizeof(int) );
                // std::vector<double> values (valuesA, valuesA + sizeof(valuesA) / sizeof(double) );

                // Eigen::MatrixXd A(3,4);
                // A.setZero();
                // for (int i = 0; i < 3; i++)
                // {
                //     for (int j = rows[i]; j < rows[i+1]; j++)
                //     {
                //     A(i,cols[j]) = values[j]; 
                //     }
                // }
                // std::cout<<"A: "<<std::endl;
                // std::cout<<A<<std::endl;

                std::cout<<"    A matrix rows: "<<jacobian.num_rows<<std::endl;
                std::cout<<"    A matrix cols: "<<jacobian.num_cols<<std::endl;
                std::cout<<"    Cl parameters: "<<variances.size()<<std::endl;

                // Eigen::MatrixXd ADense(jacobian.num_rows,jacobian.num_cols);
                // //Eigen::MatrixXd J(jacobian.num_rows,jacobian.num_cols);
                // ADense.setZero();
                // for (int i = 0; i < jacobian.num_rows; i++)
                // {
                //     double weight = sqrt(variances[i]);
                //     for (int j = jacobian.rows[i]; j < jacobian.rows[i+1]; j++)
                //     {
                //         ADense(i,jacobian.cols[j]) = jacobian.values[j];                    
                //         ADense(i,jacobian.cols[j]) *= weight; // undo the weighting during cost functions
                //         //J(i,jacobian.cols[j]) = jacobian.values[j]; 

                //     }
                // }

                Eigen::SparseMatrix<double> A;
                A.resize(jacobian.num_rows,jacobian.num_cols);
                std::vector< Eigen::Triplet<double> > tripletA(jacobian.values.size());
                int indexTripletA = 0;
                for (int i = 0; i < jacobian.num_rows; i++)
                {
                    double weight = sqrt(variances[i]);
                    for (int j = jacobian.rows[i]; j < jacobian.rows[i+1]; j++)
                    {
                        tripletA[indexTripletA] = Eigen::Triplet<double>(i,jacobian.cols[j], jacobian.values[j]*weight);
                        indexTripletA++;
                    }
                }            
                A.setFromTriplets(tripletA.begin(), tripletA.end());

                std::cout<<"  Done computing jacobian matrix"<<std::endl;            

                // if(DEBUGMODE)
                // {
                //     std::cout<<"    Writing A to file..."<<std::endl;
                //     FILE *fout = fopen("A.jck", "w");
                //     for(int i = 0; i < A.rows(); ++i)
                //     {
                //         for(int j = 0; j < A.cols(); ++j)
                //         {
                //             fprintf(fout, "%.6lf \t ", A(i,j));
                //         }
                //         fprintf(fout, "\n");
                //     }
                //     fclose(fout);
                // }

                // Eigen::Map<Eigen::VectorXd> temp(variances.data(), variances.size());
                // Eigen::MatrixXd Cl = temp.asDiagonal();

                Eigen::SparseMatrix<double> Cl;
                Cl.resize(variances.size(), variances.size());
                std::vector< Eigen::Triplet<double> > tripletCl(variances.size());
                int indexTripletCl = 0;
                for (int i = 0; i < variances.size(); i++)
                {
                    tripletCl[indexTripletCl] = Eigen::Triplet<double>(i,i,variances[i]);
                    indexTripletCl++;                
                }            
                Cl.setFromTriplets(tripletCl.begin(), tripletCl.end());

                // Eigen::MatrixXd A2 = A.block<198,96>(0,0);
                // Eigen::MatrixXd Cx2 = (A2.transpose() * Cl.inverse() * A2).inverse();


                // if(DEBUGMODE)
                // {
                //     std::cout<<"    Writing Cx2 to file..."<<std::endl;
                //     FILE *fout = fopen("Cx2.jck", "w");
                //     for(int i = 0; i < Cx2.rows(); ++i)
                //     {
                //         for(int j = 0; j < Cx2.cols(); ++j)
                //         {
                //             fprintf(fout, "%.6lf \t ", Cx2(i,j));
                //         }
                //         fprintf(fout, "\n");
                //     }
                //     fclose(fout);
                // }
                // computing the covariance matrix of the adjusted observations
                std::cout<<"  Start computing Cv..."<<std::endl;
                std::cout<<"    Cx dimensions: "<<Cx.rows()<<" by "<<Cx.cols()<<std::endl;
                PyRun_SimpleString("t0 = TIME.process_time()");        
                Eigen::SparseMatrix<double> CxSparse = Cx.sparseView();
                PyRun_SimpleString("print( '    Converting matrices:', round(TIME.process_time()-t0, 3), 's' )");      
                PyRun_SimpleString("t0 = TIME.process_time()");  

                // Eigen::SparseMatrix<double> Cl_hat = A * (CxSparse.selfadjointView<Eigen::Upper>() * A.transpose());

                // // Eigen::SparseMatrix<double> Cl_hat = A * CxSparse * A.transpose();
                Eigen::SparseMatrix<double> CxAT = (CxSparse.selfadjointView<Eigen::Upper>() * A.transpose());
                PyRun_SimpleString("print( '    Multiplying first matrices Cx*AT:', round(TIME.process_time()-t0, 3), 's' )");
                PyRun_SimpleString("t0 = TIME.process_time()");

                for (int i = 0; i < variances.size(); i++)
                {
                    Eigen::SparseMatrix<double> temp = (A.row(i) * CxAT.col(i));
                    // std::cout<<"temp: "<<temp.rows()<<", "<<temp.cols()<<" = "<<temp.coeff(0,0)<<std::endl;
                    CvDiag(i) = variances[i] - temp.coeff(0,0);
                }

                // Eigen::SparseMatrix<double> Cl_hat;
                // Cl_hat.resize(variances.size(), variances.size());
                // std::vector< Eigen::Triplet<double> > tripletCl_hat(variances.size());
                // int indexTripletCl_hat = 0;
                // for (int i = 0; i < variances.size(); i++)
                // {
                //     Eigen::SparseMatix<double> temp = (A.row(i) * CxAT.col(i));
                //     tripletCl_hat[indexTripletCl_hat] = Eigen::Triplet<double>(i,i, temp.coeff(0, 0));
                //     indexTripletCl_hat++;                
                // }            
                // Cl_hat.setFromTriplets(tripletCl_hat.begin(), tripletCl_hat.end());

                //Eigen::SparseMatrix<double> D = ttt.sparseView();
                // Eigen::SparseMatrix<double> Cl_hat = A * Cx.selfadjointView<Eigen::Upper>() * A.transpose();
                PyRun_SimpleString("print( '    Multiplying matrices A*Cx:', round(TIME.process_time()-t0, 3), 's' )");

                // Eigen::MatrixXd Cl_hat = A * Cx * A.transpose();
                // Eigen::SparseMatrix<double> Cl_hat = A * Cx * A.transpose();

                //PyRun_SimpleString("t0 = TIME.process_time()");        
                // Cv.noalias() = Eigen::MatrixXd(Cl) - Eigen::MatrixXd(Cl_hat);
                // Cv.noalias() = Eigen::MatrixXd(Cl - Cl_hat);
                //PyRun_SimpleString("print '    Subtracting matrices:', round(TIME.process_time()-t0, 3), 's' ");

                // Cv = Cl - Cl_hat;
                std::cout<<"  Done computing Cv = Cl(assumed not robust version) - Cl_hat"<<std::endl;

                std::cout<<"    Ceres Redundancy: "<<summary.num_residuals_reduced - summary.num_parameters_reduced<<std::endl;

                // compute the redundancy numbers
                double sumRedundancyNumber = 0.0;
                for (int i = 0; i < variances.size(); i++) // includes the variance for defining the datum
                {
                    // redundancyNumber(i) = Cv(i,i) / Cl(i,i);
                    // redundancyNumber(i) = Cv(i,i) / variances[i];
                    redundancyNumber(i) = CvDiag(i) / variances[i];
                    sumRedundancyNumber += redundancyNumber(i);
                }
                leastSquaresRedundancy.push_back(sumRedundancyNumber);
                std::cout<<"    Sum of ALL redundancy numbers: "<<sumRedundancyNumber<<std::endl;

                double vTPvImage = 0.0;
                double dofImage = 0.0;
                // std::cout<<"Variance size: "<<variances.size()<<std::endl;
                // std::cout<<"imageX size: "<<imageX.size()<<std::endl;
                // std::cout<<"residuals size: "<<residuals.size()<<std::endl;
                // std::cout<<"redundancyNumber size: "<<redundancyNumber.size()<<std::endl;
                for (int i = 0; i < imageX.size(); i++)
                {
                    vTPvImage += pow(residuals[2*i], 2.0);
                    vTPvImage += pow(residuals[2*i+1], 2.0);
                    dofImage += redundancyNumber(2*i);
                    dofImage += redundancyNumber(2*i+1);
                }
                std::cout<<"    Sum of image redundancy numbers: "<<dofImage<<std::endl;
                std::cout<<"       Image Only vTPv: "<<vTPvImage<<std::endl;
                std::cout<<"       A posteriori of image for rescaling (aposteriori/apriori): "<<vTPvImage / dofImage <<std::endl;
                std::cout<<"       A posteriori stdDev of image for rescaling (aposteriori/apriori): "<<sqrt(vTPvImage / dofImage) <<std::endl;

                double vTPvXYZ = 0.0;
                double dofXYZ = 0.0;
                for (int i = 0; i < XYZ.size(); i++)
                {
                    int startIndex = residuals.size() - 3*XYZ.size();
                    vTPvXYZ += pow(residuals[startIndex+3*i], 2.0);
                    vTPvXYZ += pow(residuals[startIndex+3*i+1], 2.0);
                    vTPvXYZ += pow(residuals[startIndex+3*i+2], 2.0);
                    dofXYZ += redundancyNumber(startIndex+3*i);
                    dofXYZ += redundancyNumber(startIndex+3*i+1);
                    dofXYZ += redundancyNumber(startIndex+3*i+2);
                }
                std::cout<<"    Sum of XYZ redundancy numbers: "<<dofXYZ<<std::endl;
                std::cout<<"       XYZ Only vTPv: "<<vTPvXYZ<<std::endl;
                std::cout<<"       A posteriori of XYZ for rescaling (aposteriori/apriori): "<<vTPvXYZ / dofXYZ <<std::endl;
                std::cout<<"       A posteriori stdDev of XYZ for rescaling (aposteriori/apriori): "<<sqrt(vTPvXYZ / dofXYZ) <<std::endl;

                // if(DEBUGMODE)
                // {
                //     std::cout<<"    Writing Cl to file..."<<std::endl;
                //     FILE *fout = fopen("Cl.jck", "w");
                //     for(int i = 0; i < Cl.rows(); ++i)
                //     {
                //         for(int j = 0; j < Cl.cols(); ++j)
                //         {
                //             fprintf(fout, "%.6lf \t ", Cl(i,j));
                //         }
                //         fprintf(fout, "\n");
                //     }
                //     fclose(fout);
                // }

                // if(DEBUGMODE)
                // {
                //     std::cout<<"    Writing Cl_hat to file..."<<std::endl;
                //     FILE *fout = fopen("Cl_hat.jck", "w");
                //     for(int i = 0; i < Cl_hat.rows(); ++i)
                //     {
                //         for(int j = 0; j < Cl_hat.cols(); ++j)
                //         {
                //             fprintf(fout, "%.6lf \t ", Cl_hat(i,j));
                //         }
                //         fprintf(fout, "\n");
                //     }
                //     fclose(fout);
                // }

                // if(DEBUGMODE)
                // {
                //     std::cout<<"    Writing Cv to file..."<<std::endl;
                //     FILE *fout = fopen("Cv.jck", "w");
                //     for(int i = 0; i < Cv.rows(); ++i)
                //     {
                //         for(int j = 0; j < Cv.cols(); ++j)
                //         {
                //             fprintf(fout, "%.6lf \t ", Cv(i,j));
                //         }
                //         fprintf(fout, "\n");
                //     }
                //     fclose(fout);
                // }
            }

            PyRun_SimpleString("print( 'Done computing covariance matrix of the residuals:', round(TIME.process_time()-t0, 3), 's' )");
        }
        else // if not computing Cv
        {
            std::cout<<"NOT computing Cv covariance matrix of the residuals"<<std::endl;
            // leastSquaresRedundancy.push_back(0.0);
            leastSquaresRedundancy.push_back(summary.num_residuals_reduced - summary.num_parameters_reduced);
        }

        Eigen::MatrixXd imageResiduals(imageX.size(), 2);
        Eigen::MatrixXd imageResidualsStdDev(imageX.size(), 2);
        Eigen::MatrixXd imageRedundancy(imageX.size(), 2);
        Eigen::MatrixXd reprojectionErrors(1, 3);
        reprojectionErrors.setZero();

        // std::cout<<"size: "<<imageX.size()<<std::endl;
        // std::cout<<"size: "<<CvDiag.size()<<std::endl;
        // std::cout<<"size: "<<residuals.size()<<std::endl;
        // std::cout<<"size: "<<redundancyNumber.size()<<std::endl;
        for (int n = 0; n<imageX.size(); n++)
        {
            // std::cout<<residuals[2*n]<<", "<< residuals[2*n+1]<<std::endl;
            imageResiduals(n,0) = residuals[2*n] * imageXStdDev[n]; 
            imageResiduals(n,1) = residuals[2*n+1] * imageYStdDev[n];

            imageRedundancy(n,0) = redundancyNumber(n*2);
            imageRedundancy(n,1) = redundancyNumber(n*2+1);

            // imageResidualsStdDev(n,0) = sqrt(Cv(n*2,n*2));
            // imageResidualsStdDev(n,1) = sqrt(Cv(n*2+1,n*2+1));

            imageResidualsStdDev(n,0) = sqrt(CvDiag(n*2));
            imageResidualsStdDev(n,1) = sqrt(CvDiag(n*2+1));

            // compute the reprojection error as the RMSE of v_x and v_y
            reprojectionErrors(0,0) +=  imageResiduals(n,0) * imageResiduals(n,0);
            reprojectionErrors(0,1) +=  imageResiduals(n,1) * imageResiduals(n,1);      
            reprojectionErrors(0,2) +=  imageResiduals(n,0) * imageResiduals(n,0) + imageResiduals(n,1) * imageResiduals(n,1);

        }
        // std::cout<<"Done size: "<<imageX.size()<<std::endl;
        if(DEBUGMODE)
        {
            std::cout<<"Residuals:"<<std::endl;
            std::cout<<imageResiduals<<std::endl;
        }
        std::cout<<"  Reprojection errors (RMSE in x, y, and total): " << sqrt(reprojectionErrors(0,0) / imageX.size()) << ", " << sqrt(reprojectionErrors(0,1) / imageX.size()) << " --> " << sqrt(reprojectionErrors(0,2) / (2*imageX.size())) << std::endl;

        // Output results to file
        PyRun_SimpleString("t0 = TIME.process_time()");        
        PyRun_SimpleString("print( 'Start outputting bundle adjustment results to file' )");     
        //Output results back to Python for plotting
        if (true)
        {
            std::cout<<"  Writing residuals to file..."<<std::endl;
            FILE *fout = fopen("image.jck", "w");
            for(int i = 0; i < imageTarget.size(); ++i)
            {
                fprintf(fout, "%i %i %i %.6lf %.6lf %.6lf %.6lf %.2lf %.2lf %.6lf %.6lf\n", pointReferenceID[i], frameReferenceID[i], sensorReferenceID[i], imageX[i], imageY[i], imageResiduals(i,0), imageResiduals(i,1), imageRedundancy(i,0), imageRedundancy(i,1), imageResidualsStdDev(i,0), imageResidualsStdDev(i,1));
            }
            fclose(fout);
        }
        
        // if (true)
        // {
        //     // // convert residuals to PCL point cloud format
        //     // std::cout<<"  Downsampling residuals..."<<std::endl;
        //     // pcl::PointCloud<pcl::PointXYZI>::Ptr downsampledCloud (new pcl::PointCloud<pcl::PointXYZI>);
        //     // pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
        //     // cloud->width = imageTarget.size();
        //     // cloud->height = 1;
        //     // cloud->is_dense = false;
        //     // cloud->points.resize(cloud->width * cloud->height);
        //     // std::cout<<"imageTarget size: "<<imageTarget.size()<<std::endl;
        //     // std::cout<<"cloud size: "<<cloud->size()<<std::endl;
        //     // for (int i = 0; i < cloud->size(); i++)
        //     // {
        //     //     cloud->points.at(i).x = imageX[i];
        //     //     cloud->points.at(i).y = imageY[i];
        //     //     cloud->points.at(i).z = imageResiduals(i,0);
        //     //     cloud->points.at(i).intensity = imageResiduals(i,1);
        //     // }

        //     // // voxel downsampling
        //     // std::cout<<"start voxelgrid..."<<std::endl;
        //     // pcl::VoxelGrid<pcl::PointXYZ> vg;
        //     // //vg.setDownsampleAllData(true);
        //     // vg.setLeafSize(1.0, 1.0, 1.0);
        //     // vg.setInputCloud(cloud);
        //     // vg.filter(*downsampledCloud);
        //     // std::cout<<"done voxelgrid..."<<std::endl;
        //     // std::cout<<"  Downsampled residuals from "<<imageTarget.size()<<" to "<<downsampledCloud->size()<<std::endl;

        //     // // convert residuals to PCL point cloud format
        //     // std::cout<<"  Downsampling residuals..."<<std::endl;
        //     // pcl::PointCloud<pcl::PointXYZI> downsampledCloud;
        //     // pcl::PointCloud<pcl::PointXYZI> cloud;
        //     // cloud.width = imageTarget.size();
        //     // cloud.height = 1;
        //     // cloud.is_dense = false;
        //     // cloud.points.resize(cloud.width * cloud.height);
        //     // std::cout<<"imageTarget size: "<<imageTarget.size()<<std::endl;
        //     // std::cout<<"cloud size: "<<cloud.size()<<std::endl;
        //     // for (int i = 0; i < cloud.size(); i++)
        //     // {
        //     //     cloud.points.at(i).x = imageX[i];
        //     //     cloud.points.at(i).y = imageY[i];
        //     //     cloud.points.at(i).z = imageResiduals(i,0);
        //     //     cloud.points.at(i).intensity = imageResiduals(i,1);
        //     // }

        //     // // voxel downsampling
        //     // std::cout<<"start voxelgrid..."<<std::endl;
        //     // pcl::VoxelGrid<pcl::PointXYZI> downsample;
        //     // //downsample.setDownsampleAllData(true);
        //     // downsample.setLeafSize(1.0, 1.0, 100.0);
        //     // downsample.setInputCloud(cloud);
        //     // downsample.filter(downsampledCloud);
        //     // std::cout<<"done voxelgrid..."<<std::endl;
        //     // std::cout<<"  Downsampled residuals from "<<imageTarget.size()<<" to "<<downsampledCloud.size()<<std::endl;

        //     // std::cout<<"  Writing downsampled residuals to file..."<<std::endl;
        //     // FILE *fout = fopen("imageDownsampled.jck", "w");
        //     // for(int i = 0; i < downsampledCloud->size(); ++i)
        //     // {
        //     //     fprintf(fout, "%.6lf %.6lf %.6lf %.6lf\n", downsampledCloud->points.at(i).x, downsampledCloud->points.at(i).y, downsampledCloud->points.at(i).z, downsampledCloud->points.at(i).intensity);
        //     // }
        //     // fclose(fout);            
        // }

        if (true)
        {
            std::cout<<"  Writing targets to file..."<<std::endl;
            FILE *fout = fopen("XYZ.jck", "w");
            for(int i = 0; i < xyzTarget.size(); ++i)
            {
                fprintf(fout, "%i %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf\n", xyzTarget[i], XYZ[i][0], XYZ[i][1], XYZ[i][2], sqrt(xyzVariance(i,0)), sqrt(xyzVariance(i,1)), sqrt(xyzVariance(i,2)));
            }
            fclose(fout);
        }

        if (true)
        {
            std::cout<<"  Writing EOPs to file..."<<std::endl;
            FILE *fout = fopen("EOP.jck", "w");
            for(int i = 0; i < EOP.size(); ++i)
            {
                fprintf(fout, "%i %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf\n", eopStation[i], EOP[i][0]*180.0/PI, EOP[i][1]*180.0/PI, EOP[i][2]*180.0/PI, EOP[i][3], EOP[i][4], EOP[i][5], sqrt(eopVariance(i,0))*180.0/PI, sqrt(eopVariance(i,1))*180.0/PI, sqrt(eopVariance(i,2))*180.0/PI, sqrt(eopVariance(i,3)), sqrt(eopVariance(i,4)), sqrt(eopVariance(i,5)) );
            }
            fclose(fout);
        }

        if (true)
        {
            std::cout<<"  Writing IOPs to file..."<<std::endl;
            FILE *fout = fopen("iop.jck", "w");
            for(int i = 0; i < IOP.size(); ++i)
            {
                fprintf(fout, "%i %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf\n", iopCamera[i], IOP[i][0], IOP[i][1], IOP[i][2], sqrt(iopVariance(i,0)), sqrt(iopVariance(i,1)), sqrt(iopVariance(i,2)) );
            }
            fclose(fout);
        }

        if (ROPMODE || WEIGHTEDROPMODE)
        {
            std::cout<<"  Writing ROPs to file..."<<std::endl;
            FILE *fout = fopen("ROP.jck", "w");
            for(int i = 0; i < ROP.size(); ++i)
            {
                fprintf(fout, "%i --> %i: %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf\n", ropID[i][0], ropID[i][1], ROP[i][0]*180.0/PI, ROP[i][1]*180.0/PI, ROP[i][2]*180.0/PI, ROP[i][3], ROP[i][4], ROP[i][5], sqrt(ropVariance(i,0))*180.0/PI, sqrt(ropVariance(i,1))*180.0/PI, sqrt(ropVariance(i,2))*180.0/PI, sqrt(ropVariance(i,3)), sqrt(ropVariance(i,4)), sqrt(ropVariance(i,5)) );
                std::cout<<"    " << ropID[i][0]<<" --> "<< ropID[i][1]<<": "<<ROP[i][0]*180.0/PI<<", "<< ROP[i][1]*180.0/PI<<", "<< ROP[i][2]*180.0/PI <<", "<< ROP[i][3]<<", "<< ROP[i][4]<<", "<< ROP[i][5]<<". Distance: "<< sqrt(ROP[i][3]*ROP[i][3] + ROP[i][4]*ROP[i][4] + ROP[i][5]*ROP[i][5]) <<std::endl;
            }
            fclose(fout);
        }

        PyRun_SimpleString("print( 'Done outputting bundle adjustment results to file:', round(TIME.process_time()-t0, 3), 's' )");
        
        
















































        /////// OLD OBSELETE
        // // Do quality control
        // if(true)
        // {
        //     PyRun_SimpleString("t0 = TIME.process_time()");        
        //     PyRun_SimpleString("print( 'Start computing the object space reconstruction error relative to the ground truth (QC): Assumes everything is in the same order' )");  
        //     std::cout<<"  Ground truth filename: "<<INPUTXYZTRUTHFILENAME<<std::endl;

        //     inp.open(INPUTXYZTRUTHFILENAME);
        //     std::vector<std::vector<double> >XYZTruth;
        //     while (true) 
        //     {
        //         int c0;
        //         double c1, c2, c3, c4, c5, c6; 
        //         inp >> c0 >> c1 >> c2 >> c3 >> c4 >> c5 >> c6;

        //         std::vector<double>temp;
        //         temp.resize(3);
        //         temp[0] = c1;
        //         temp[1] = c2;
        //         temp[2] = c3;
        //         XYZTruth.push_back(temp);

        //         if( inp.eof() ) 
        //             break;
        //     }

        //     XYZTruth.pop_back();
        //     inp.close();

        //     std::cout << "  Number of XYZ Ground Truth Read: "<< XYZTruth.size() << std::endl;
        //     std::cout << "  Number of XYZ estimated        : "<< XYZ.size() << std::endl;

        //     double RMSE_X = 0.0;
        //     double RMSE_Y = 0.0;
        //     double RMSE_Z = 0.0;
        //     for (int i = 0; i < XYZ.size(); i++)
        //     {
        //         RMSE_X += pow(XYZ[i][0] - XYZTruth[i][0],2.0);
        //         RMSE_Y += pow(XYZ[i][1] - XYZTruth[i][1],2.0);
        //         RMSE_Z += pow(XYZ[i][2] - XYZTruth[i][2],2.0);
        //     }
        //     RMSE_X /= XYZTruth.size();
        //     RMSE_Y /= XYZTruth.size();
        //     RMSE_Z /= XYZTruth.size();

        //     RMSE_X = sqrt(RMSE_X);
        //     RMSE_Y = sqrt(RMSE_Y);
        //     RMSE_Z = sqrt(RMSE_Z);
            
        //     std::cout<<"    RMSE X, Y, Z, Total: "<<RMSE_X<<", "<<RMSE_Y<<", "<<RMSE_Z<<". "<<sqrt((RMSE_X*RMSE_X+RMSE_Y*RMSE_Y+RMSE_Z*RMSE_Z)/3.0)<<std::endl;
        //     PyRun_SimpleString("print( 'Done QC:', round(TIME.process_time()-t0, 3), 's' )");
        // }   

                
        // Do quality control
        if(true)
        {
            PyRun_SimpleString("t0 = TIME.process_time()");        
            PyRun_SimpleString("print( 'Start computing the object space reconstruction error relative to the ground truth (QC): DOES NOT assume everything is in the same order' )");  
            std::cout<<"  Ground truth filename: "<<INPUTXYZTRUTHFILENAME<<std::endl;

            inp.open(INPUTXYZTRUTHFILENAME);
            std::vector<double> XYZTruthID;
            std::vector<std::vector<double> >XYZTruth;
            while (true) 
            {
                int c0;
                double c1, c2, c3, c4, c5, c6; 
                inp >> c0 >> c1 >> c2 >> c3 >> c4 >> c5 >> c6;

                XYZTruthID.push_back(c0);

                std::vector<double>temp;
                temp.resize(3);
                temp[0] = c1;
                temp[1] = c2;
                temp[2] = c3;
                XYZTruth.push_back(temp);

                if( inp.eof() ) 
                    break;
            }

            XYZTruthID.pop_back();
            XYZTruth.pop_back();
            inp.close();

            std::cout << "  Number of XYZ Ground Truth Read: "<< XYZTruth.size() << std::endl;
            std::cout << "  Number of XYZ estimated        : "<< XYZ.size() << std::endl;

            double RMSE_X = 0.0;
            double RMSE_Y = 0.0;
            double RMSE_Z = 0.0;
            for (int i = 0; i < XYZTruthID.size(); i++)
            {
                for (int j = 0; j < xyzTarget.size(); j++)
                {
                    if (xyzTarget[j] == XYZTruthID[i])
                    {
                        RMSE_X += pow(XYZ[j][0] - XYZTruth[i][0],2.0);
                        RMSE_Y += pow(XYZ[j][1] - XYZTruth[i][1],2.0);
                        RMSE_Z += pow(XYZ[j][2] - XYZTruth[i][2],2.0);
                        break;
                    }
                }
            }
            RMSE_X /= XYZTruth.size();
            RMSE_Y /= XYZTruth.size();
            RMSE_Z /= XYZTruth.size();

            RMSE_X = sqrt(RMSE_X);
            RMSE_Y = sqrt(RMSE_Y);
            RMSE_Z = sqrt(RMSE_Z);
            
            std::cout<<"    RMSE X, Y, Z, Total: "<<RMSE_X<<", "<<RMSE_Y<<", "<<RMSE_Z<<" --> "<<sqrt((RMSE_X*RMSE_X+RMSE_Y*RMSE_Y+RMSE_Z*RMSE_Z)/3.0)<<std::endl;
            PyRun_SimpleString("print( 'Done QC:', round(TIME.process_time()-t0, 3), 's' )");
        }   

        // for(int i = 0; i < leastSquaresCost.size(); ++i)
        //     {
        //     std::cout<<2.0*leastSquaresCost[i]<<std::endl;
        //     }


        if (doML) // if not running a conventional bundle adjustment
        {
            PyRun_SimpleString("t0 = TIME.process_time()");        
            PyRun_SimpleString("print( 'Start doing machine learning in Python' )");    

            //system("python ~/BundleAdjustment/python/gaussianProcess.py");
            system("python ~/BundleAdjustment/python/nearestNeighbour.py");

            PyRun_SimpleString("print( 'Done doing machine learning regression:', round(TIME.process_time()-t0, 3), 's' )");

            // read in the machine learned cost
            inp.open("/home/jckchow/BundleAdjustment/build/kNNCost.jck");
            std::vector<double> MLCost;
            std::vector<double> MLRedundancy;
            while (true) 
            {
                double c1, c2;
                inp >> c1 >> c2;

                MLCost.push_back(c1);
                MLRedundancy.push_back(c2);

                if( inp.eof() )
                    break;
            }
            
            MLCost.pop_back();
            MLRedundancy.pop_back();

            inp.close();

            machineLearnedCost.push_back(MLCost[0]);
            machineLearnedRedundancy.push_back(MLRedundancy[0]);
        }
    }
 
    if (doML)
    {
        std::cout<<"  Writing least squares costs to file..."<<std::endl;
        FILE *fout = fopen("costs.jck", "w");
        for(int i = 0; i < leastSquaresCost.size(); ++i)
        {
            fprintf(fout, "%.6lf %.6lf %.6lf %.6lf\n", 2.0*leastSquaresCost[i], leastSquaresRedundancy[i], machineLearnedCost[i], machineLearnedRedundancy[i] );
        }
        fclose(fout);
    }


    if (PLOTRESULTS)
    {
        PyRun_SimpleString("t0 = TIME.process_time()");        
        PyRun_SimpleString("print( 'Start plotting results in Python' )");    
        system("python ~/BundleAdjustment/python/plotBundleAdjustment.py");
        PyRun_SimpleString("print( 'Done plotting results:', round(TIME.process_time()-t0, 3), 's' )");

    }

    PyRun_SimpleString("print( '----------------------------Program Successful ^-^, Total Run Time:', round(TIME.process_time()-totalTime, 3), 's', '----------------------------', )");
    return 0;
}