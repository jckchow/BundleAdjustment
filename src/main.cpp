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
//#include <pcl/point_types.h>
//#include <pcl/filters/voxel_grid.h>

// Define constants
#define PI 3.141592653589793238462643383279502884197169399
#define NUMITERATION 1
#define DEBUGMODE 0
#define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/Data/Dcs28mm.pho"
#define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/Data/Dcs28mmTemp.pho" 
#define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/Data/Dcs28mm.iop"
#define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/Data/Dcs28mm.eop"
#define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/Data/Dcs28mm.xyz"

// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1Training.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/xrayData1/xray1TrainingTemp.pho" 
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1Training.eop"
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.xyz"

// #define INPUTIMAGEFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1Testing.pho"
// #define INPUTIMAGEFILENAMETEMP "/home/jckchow/BundleAdjustment/xrayData1/xray1TestingTemp.pho" 
// #define INPUTIOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1.iop"
// #define INPUTEOPFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1Testing.eop"
// #define INPUTXYZFILENAME "/home/jckchow/BundleAdjustment/xrayData1/xray1Truth.xyz"

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
  T x_bar = T(x_) - T(xp_);
  T y_bar = T(y_) - T(yp_);
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
  T x_bar = T(x_) - T(xp_);
  T y_bar = T(y_) - T(yp_);
  T r = sqrt(x_bar*x_bar + y_bar*y_bar);

//   T delta_x = x_bar*(AP[2]*pow(r,2.0)+AP[3]*pow(r,4.0)+AP[4]*pow(r,6.0)) + AP[5]*(pow(r,2.0)+T(2.0)*pow(x_bar,2.0))+T(2.0)*AP[6]*x_bar*y_bar + AP[0]*x_bar+AP[1]*y_bar;
//   T delta_y = y_bar*(AP[2]*pow(r,2.0)+AP[3]*pow(r,4.0)+AP[4]*pow(r,6.0)) + AP[6]*(pow(r,2.0)+T(2.0)*pow(y_bar,2.0))+T(2.0)*AP[5]*x_bar*y_bar;

  T delta_x = x_bar*(AP[2]*r*r+AP[3]*r*r*r*r+AP[4]*r*r*r*r*r*r) + AP[5]*(r*r+T(2.0)*x_bar*x_bar)+T(2.0)*AP[6]*x_bar*y_bar + AP[0]*x_bar+AP[1]*y_bar;
  T delta_y = y_bar*(AP[2]*r*r+AP[3]*r*r*r*r+AP[4]*r*r*r*r*r*r) + AP[6]*(r*r+T(2.0)*y_bar*y_bar)+T(2.0)*AP[5]*x_bar*y_bar;


  T x_true = x + IOP[0] + delta_x + MLP[0]; // MLP is the machine learned parameters
  T y_true = y + IOP[1] + delta_y + MLP[1];

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

//   std::cout<<"x, y: "<<x+T(xp_)<<", "<<y+T(yp_)<<std::endl;
//   std::cout<<"x_obs, y_obs: "<<T(x_)<<", "<<T(y_)<<std::endl;


  // camera correction model AP = a1, a2, k1, k2, k3, p1, p2, ...
  T x_bar = T(x_) - T(xp_);
  T y_bar = T(y_) - T(yp_);
  T r = sqrt(x_bar*x_bar + y_bar*y_bar);

//   T delta_x = x_bar*(AP[2]*pow(r,2.0)+AP[3]*pow(r,4.0)+AP[4]*pow(r,6.0)) + AP[5]*(pow(r,2.0)+T(2.0)*pow(x_bar,2.0))+T(2.0)*AP[6]*x_bar*y_bar + AP[0]*x_bar+AP[1]*y_bar;
//   T delta_y = y_bar*(AP[2]*pow(r,2.0)+AP[3]*pow(r,4.0)+AP[4]*pow(r,6.0)) + AP[6]*(pow(r,2.0)+T(2.0)*pow(y_bar,2.0))+T(2.0)*AP[5]*x_bar*y_bar;

  T delta_x = x_bar*(AP[2]*r*r+AP[3]*r*r*r*r+AP[4]*r*r*r*r*r*r) + AP[5]*(r*r+T(2.0)*x_bar*x_bar)+T(2.0)*AP[6]*x_bar*y_bar + AP[0]*x_bar+AP[1]*y_bar;
  T delta_y = y_bar*(AP[2]*r*r+AP[3]*r*r*r*r+AP[4]*r*r*r*r*r*r) + AP[6]*(r*r+T(2.0)*y_bar*y_bar)+T(2.0)*AP[5]*x_bar*y_bar;


  T x_true = x + IOP[0] + delta_x + T(xMLP_); // MLP is the machine learned parameters
  T y_true = y + IOP[1] + delta_y + T(yMLP_);

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

/////////////////////////
/// MAIN CERES-SOLVER ///
/////////////////////////
int main(int argc, char** argv) {
    Py_Initialize();
    google::InitGoogleLogging(argv[0]);

    PyRun_SimpleString("import matplotlib.pyplot as plt");
    PyRun_SimpleString("import numpy as np");
    PyRun_SimpleString("import time as TIME");
    PyRun_SimpleString("totalTime = TIME.clock()");        
     
    std::ifstream inp;      
    std::vector<double> leastSquaresCost;

    //////////////////////////////////////
    /// Read in the data from files
    //////////////////////////////////////
    for (int iterNum = 0; iterNum < NUMITERATION; iterNum++)
    {

        std::cout<<"---------------------------------------- Global Iteration: " << iterNum <<"----------------------------------------"<<std::endl;
        PyRun_SimpleString("t0 = TIME.clock()");        
        PyRun_SimpleString("print 'Start reading data' ");   
        // Reading *.pho file
        PyRun_SimpleString("print '  Start reading image observations' ");  
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
        PyRun_SimpleString("print '  Start reading EOPs' ");          
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

    // Reading *.iop file
        PyRun_SimpleString("print '  Start reading IOPs' ");
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


        // Reading *.xyz file
        PyRun_SimpleString("print '  Start reading XYZ' ");  
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

        PyRun_SimpleString("print 'Done reading data:', round(TIME.clock()-t0, 3), 's' ");

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
        PyRun_SimpleString("t0 = TIME.clock()");        
        PyRun_SimpleString("print 'Start building Ceres-Solver cost functions' ");
    
        std::vector<int> imageReferenceID; // for use when outting the residuals
        imageReferenceID.resize(imageX.size());
        ceres::Problem problem;
        ceres::LossFunction* loss = NULL;
        //loss = new ceres::HuberLoss(1.0);
        // loss = new ceres::CauchyLoss(0.5);

        // // Conventional collinearity condition
        // for(int n = 0; n < imageX.size(); n++) // loop through all observations
        // {
        //      std::vector<int>::iterator it;
        //      it = std::find(xyzTarget.begin(), xyzTarget.end(), imageTarget[n]);
        //      int indexPoint = std::distance(xyzTarget.begin(),it);
        //      // std::cout<<"indexPoint: "<<indexPoint<<", ID: "<< imageTarget[n]<<std::endl;

        //      it = std::find(eopStation.begin(), eopStation.end(), imageStation[n]);
        //      int indexPose = std::distance(eopStation.begin(),it);
        //      // std::cout<<"index: "<<indexPose<<", ID: "<< imageStation[n]<<std::endl;

        //      it = std::find(iopCamera.begin(), iopCamera.end(), eopCamera[indexPose]);
        //      int indexSensor = std::distance(iopCamera.begin(),it);
        //      // std::cout<<"index: "<<indexSensor<<", ID: "<< eopCamera[indexPose]<<std::endl;   

        //     //  std::cout<<"EOP: "<< EOP[indexPose][3] <<", " << EOP[indexPose][4] <<", " << EOP[indexPose][5]  <<std::endl;
        //     //  std::cout<<"XYZ: "<< XYZ[indexPoint][0] <<", " << XYZ[indexPoint][1] <<", " << XYZ[indexPoint][2]  <<std::endl;

        //     ceres::CostFunction* cost_function =
        //         new ceres::AutoDiffCostFunction<collinearity, 2, 6, 3, 3, 7>(
        //             new collinearity(imageX[n],imageY[n],imageXStdDev[n], imageYStdDev[n],iopXp[indexSensor],iopYp[indexSensor]));
        //     problem.AddResidualBlock(cost_function, NULL, &EOP[indexPose][0], &XYZ[indexPoint][0], &IOP[indexSensor][0], &AP[indexSensor][0]);  

        //     problem.SetParameterBlockConstant(&IOP[indexSensor][0]);
        //     problem.SetParameterBlockConstant(&AP[indexSensor][0]);
        // }

        // Collinearity condition with machine learned parameters
        for(int n = 0; n < imageX.size(); n++) // loop through all observations
        {
            std::vector<int>::iterator it;
            it = std::find(xyzTarget.begin(), xyzTarget.end(), imageTarget[n]);
            int indexPoint = std::distance(xyzTarget.begin(),it);
            // std::cout<<"indexPoint: "<<indexPoint<<", ID: "<< imageTarget[n]<<std::endl;

            it = std::find(eopStation.begin(), eopStation.end(), imageStation[n]);
            int indexPose = std::distance(eopStation.begin(),it);
            // std::cout<<"index: "<<indexPose<<", ID: "<< imageStation[n]<<std::endl;

            it = std::find(iopCamera.begin(), iopCamera.end(), eopCamera[indexPose]);
            int indexSensor = std::distance(iopCamera.begin(),it);
            // std::cout<<"index: "<<indexSensor<<", ID: "<< eopCamera[indexPose]<<std::endl;   

            // for book keeping
            imageReferenceID[n] = iopCamera[indexSensor];

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
            problem.AddResidualBlock(cost_function, NULL, &EOP[indexPose][0], &XYZ[indexPoint][0], &IOP[indexSensor][0], &AP[indexSensor][0]);  

            problem.SetParameterBlockConstant(&IOP[indexSensor][0]);
            problem.SetParameterBlockConstant(&AP[indexSensor][0]);

        }
    
        // define the datum by pseduo observations of the positions for defining the datum
        if(true)
        {
            for(int n = 0; n < xyzTarget.size(); n++)
            {
                ceres::CostFunction* cost_function =
                    new ceres::AutoDiffCostFunction<constrainPoint, 3, 3>(
                        new constrainPoint(xyzX[n], xyzY[n], xyzZ[n], xyzXStdDev[n], xyzYStdDev[n], xyzZStdDev[n]));
                problem.AddResidualBlock(cost_function, NULL, &XYZ[n][0]);
            }
        }

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

        PyRun_SimpleString("print 'Done building Ceres-Solver cost functions:', round(TIME.clock()-t0, 3), 's' ");

        ceres::Solver::Options options;
        options.max_num_iterations = 50;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; // sparse solver
        // options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = true;
        options.max_lm_diagonal = 1.0E-150; // force it behave like a Gauss-Newton update
        options.min_lm_diagonal = 1.0E-150;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << "\n";
        std::cout << summary.FullReport() << "\n";

        // storing it for comparison in this EM like routine
        leastSquaresCost.push_back(summary.final_cost);

        PyRun_SimpleString("t0 = TIME.clock()");        
        PyRun_SimpleString("print 'Start computing covariance matrix' ");  
        ceres::Covariance::Options covarianceOoptions;
        ceres::Covariance covariance(covarianceOoptions);

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
                for(int j = 0; j < EOP.size(); j++)
                    covariance_blocks.push_back(std::make_pair(&EOP[i][0], &EOP[j][0]));

            for(int i = 0; i < XYZ.size(); i++)
                for(int j = 0; j < XYZ.size(); j++)
                    covariance_blocks.push_back(std::make_pair(&XYZ[i][0], &XYZ[j][0]));

            for(int i = 0; i < IOP.size(); i++)
                for(int j = 0; j < IOP.size(); j++)
                    covariance_blocks.push_back(std::make_pair(&IOP[i][0], &IOP[j][0]));

            for(int i = 0; i < AP.size(); i++)
                for(int j = 0; j < AP.size(); j++)
                    covariance_blocks.push_back(std::make_pair(&AP[i][0], &AP[j][0]));

            // for(int i = 0; i < MLP.size(); i++)
            //     for(int j = 0; j < MLP.size(); j++)
            //         covariance_blocks.push_back(std::make_pair(&MLP[i][0], &MLP[j][0]));
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
            }

            for(int i = 0; i < XYZ.size(); i++)
            {
                for(int j = 0; j < IOP.size(); j++)
                    covariance_blocks.push_back(std::make_pair(&XYZ[i][0], &IOP[j][0]));

                for(int j = 0; j < AP.size(); j++)
                    covariance_blocks.push_back(std::make_pair(&XYZ[i][0], &AP[j][0]));

                // for(int j = 0; j < MLP.size(); j++)
                //     covariance_blocks.push_back(std::make_pair(&XYZ[i][0], &MLP[j][0]));
            }

            for(int i = 0; i < IOP.size(); i++)
            {
                for(int j = 0; j < AP.size(); j++)
                    covariance_blocks.push_back(std::make_pair(&IOP[i][0], &AP[j][0]));

                // for(int j = 0; j < MLP.size(); j++)
                //     covariance_blocks.push_back(std::make_pair(&IOP[i][0], &MLP[j][0]));
            }

            for(int i = 0; i < AP.size(); i++)
            {
                // for(int j = 0; j < MLP.size(); j++)
                //     covariance_blocks.push_back(std::make_pair(&AP[i][0], &MLP[j][0]));
            }
        }

        covariance.Compute(covariance_blocks, &problem);

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

        Eigen::MatrixXd xyzVariance(XYZ.size(),3);
        for(int i = 0; i < XYZ.size(); i++)
        {
            Eigen::MatrixXd covariance_X(3, 3);
            covariance.GetCovarianceBlock(&XYZ[i][0], &XYZ[i][0], covariance_X.data());
            Eigen::VectorXd variance_X(3);
            variance_X = covariance_X.diagonal();
            xyzVariance(i,0) = variance_X(0);
            xyzVariance(i,1) = variance_X(1);
            xyzVariance(i,2) = variance_X(2);

        //  std::cout << "Variance: " << covariance_X.diagonal() << std::endl; 
        //  std::cout<<"covariance matrix: "<<std::endl;
        //  std::cout<<covariance_X<<std::endl;

            // // store the full variance-covariance matrix
            // for (int n = 0; n < covariance_X.rows(); n++)
            //     for (int m = 0; m < covariance_X.cols(); m++)
            //         Cx(i*3+n + 6*EOP.size(),i*3+m + 6*EOP.size()) = covariance_X(n,m);
        }

        Eigen::MatrixXd eopVariance(EOP.size(),6);
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
        }

        Eigen::MatrixXd iopVariance(IOP.size(),3);
        for(int i = 0; i < IOP.size(); i++)
        {
            Eigen::MatrixXd covariance_X(3, 3);
            covariance.GetCovarianceBlock(&IOP[i][0], &IOP[i][0], covariance_X.data());
            Eigen::VectorXd variance_X(3);
            variance_X = covariance_X.diagonal();
            iopVariance(i,0) = variance_X(0);
            iopVariance(i,1) = variance_X(1);
            iopVariance(i,2) = variance_X(2);

            // // store the full variance-covariance matrix
            // for (int n = 0; n < covariance_X.rows(); n++)
            //     for (int m = 0; m < covariance_X.cols(); m++)
            //         Cx(i*3+n + 6*EOP.size()+3*XYZ.size(),i*3+m + 6*EOP.size()+3*XYZ.size()) = covariance_X(n,m);
        }

        Eigen::MatrixXd apVariance(AP.size(),7);
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

            // // store the full variance-covariance matrix
            // for (int n = 0; n < covariance_X.rows(); n++)
            //     for (int m = 0; m < covariance_X.cols(); m++)
            //         Cx(i*7+n + 6*EOP.size()+3*XYZ.size()+3*IOP.size(),i*7+m + 6*EOP.size()+3*XYZ.size()+3*IOP.size()) = covariance_X(n,m);
        }

        Eigen::MatrixXd mlpVariance(MLP.size(),2);
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

        //       EOP XYZ IOP  AP  MLP
        // EOP    x   x   x    x   x    
        // XYZ        x   x    x   x
        // IOP            x    x   x
        // AP                  x   x        
        // MLP                     x
        Eigen::MatrixXd Cx(summary.num_parameters,summary.num_parameters);

        if (true)
        {
            // Get the full variance-covariance matrix Cx
            for(int i = 0; i < EOP.size(); i++)
            {
                for(int j = 0; j < EOP.size(); j++)
                {
                    Eigen::MatrixXd covariance_X(6, 6);
                    covariance.GetCovarianceBlock(&EOP[i][0], &EOP[j][0], covariance_X.data());

                    // // store the full variance-covariance matrix
                    // for (int n = 0; n < covariance_X.rows(); n++)
                    //     for (int m = 0; m < covariance_X.cols(); m++)
                    //         Cx(i*6+n,i*6+m) = covariance_X(n,m);

                    Cx.block<6,6>(i*6,j*6) = covariance_X;
                }

                for(int j = 0; j < XYZ.size(); j++)
                {
                    Eigen::MatrixXd covariance_X(6, 3);
                    covariance.GetCovarianceBlock(&EOP[i][0], &XYZ[j][0], covariance_X.data());

                    Cx.block<6,3>(i*6,j*3 + 6*EOP.size()) = covariance_X;
                }

                for(int j = 0; j < IOP.size(); j++)
                {
                    Eigen::MatrixXd covariance_X(6, 3);
                    covariance.GetCovarianceBlock(&EOP[i][0], &IOP[j][0], covariance_X.data());

                    Cx.block<6,3>(i*6,j*3 + 6*EOP.size()+3*XYZ.size()) = covariance_X;
                }

                for(int j = 0; j < AP.size(); j++)
                {
                    Eigen::MatrixXd covariance_X(6, 7);
                    covariance.GetCovarianceBlock(&EOP[i][0], &AP[j][0], covariance_X.data());

                    Cx.block<6,7>(i*6,j*7 + 6*EOP.size()+3*XYZ.size()+3*IOP.size()) = covariance_X;
                }

                // for(int j = 0; j < MLP.size(); j++)
                // {
                //     Eigen::MatrixXd covariance_X(6, 2);
                //     covariance.GetCovarianceBlock(&EOP[i][0], &MLP[j][0], covariance_X.data());

                //     Cx.block<6,2>(i*6,j*2 + 6*EOP.size()+3*XYZ.size()+3*IOP.size()+7*AP.size()) = covariance_X;
                // }
            }

            for(int i = 0; i < XYZ.size(); i++)
            {
                for(int j = 0; j < XYZ.size(); j++)
                {
                    Eigen::MatrixXd covariance_X(3, 3);
                    covariance.GetCovarianceBlock(&XYZ[i][0], &XYZ[j][0], covariance_X.data());

                    Cx.block<3,3>(i*3 + 6*EOP.size(),j*3 + 6*EOP.size()) = covariance_X;
                }

                for(int j = 0; j < IOP.size(); j++)
                {
                    Eigen::MatrixXd covariance_X(3, 3);
                    covariance.GetCovarianceBlock(&XYZ[i][0], &IOP[j][0], covariance_X.data());

                    Cx.block<3,3>(i*3 + 6*EOP.size(),j*3 + 6*EOP.size()+3*XYZ.size()) = covariance_X;
                }

                for(int j = 0; j < AP.size(); j++)
                {
                    Eigen::MatrixXd covariance_X(3, 7);
                    covariance.GetCovarianceBlock(&XYZ[i][0], &AP[j][0], covariance_X.data());

                    Cx.block<3,7>(i*3 + 6*EOP.size(),j*7 + 6*EOP.size()+3*XYZ.size()+3*IOP.size()) = covariance_X;
                }

                // for(int j = 0; j < MLP.size(); j++)
                // {
                //     Eigen::MatrixXd covariance_X(3, 2);
                //     covariance.GetCovarianceBlock(&XYZ[i][0], &MLP[j][0], covariance_X.data());

                //     Cx.block<3,2>(i*3 + 6*EOP.size(),j*2 + 6*EOP.size()+3*XYZ.size()+3*IOP.size()+7*AP.size()) = covariance_X;
                // }
            }

            for(int i = 0; i < IOP.size(); i++)
            {
                for(int j = 0; j < IOP.size(); j++)
                {
                    Eigen::MatrixXd covariance_X(3, 3);
                    covariance.GetCovarianceBlock(&IOP[i][0], &IOP[j][0], covariance_X.data());

                    Cx.block<3,3>(i*3 + 6*EOP.size()+3*XYZ.size(),j*3 + 6*EOP.size()+3*XYZ.size()) = covariance_X;
                }

                for(int j = 0; j < AP.size(); j++)
                {
                    Eigen::MatrixXd covariance_X(3, 7);
                    covariance.GetCovarianceBlock(&IOP[i][0], &AP[j][0], covariance_X.data());

                    Cx.block<3,7>(i*3 + 6*EOP.size()+3*XYZ.size(),j*7 + 6*EOP.size()+3*XYZ.size()+3*IOP.size()) = covariance_X;
                }

                // for(int j = 0; j < MLP.size(); j++)
                // {
                //     Eigen::MatrixXd covariance_X(3, 2);
                //     covariance.GetCovarianceBlock(&IOP[i][0], &MLP[j][0], covariance_X.data());

                //     Cx.block<3,2>(i*3 + 6*EOP.size()+3*XYZ.size(),j*2 + 6*EOP.size()+3*XYZ.size()+3*IOP.size()+7*AP.size()) = covariance_X;
                // }
            }

            for(int i = 0; i < AP.size(); i++)
            {
                for(int j = 0; j < AP.size(); j++)
                {
                    Eigen::MatrixXd covariance_X(7, 7);
                    covariance.GetCovarianceBlock(&AP[i][0], &AP[j][0], covariance_X.data());

                    Cx.block<7,7>(i*3 + 6*EOP.size()+3*XYZ.size()+3*IOP.size(),j*7 + 6*EOP.size()+3*XYZ.size()+3*IOP.size()) = covariance_X;
                }

                // for(int j = 0; j < MLP.size(); j++)
                // {
                //     Eigen::MatrixXd covariance_X(7, 2);
                //     covariance.GetCovarianceBlock(&AP[i][0], &MLP[j][0], covariance_X.data());

                //     Cx.block<7,2>(i*3 + 6*EOP.size()+3*XYZ.size()+3*IOP.size(),j*2 + 6*EOP.size()+3*XYZ.size()+3*IOP.size()+7*AP.size()) = covariance_X;
                // }
            }

            for(int i = 0; i < MLP.size(); i++)
            {
                // for(int j = 0; j < MLP.size(); j++)
                // {
                //     Eigen::MatrixXd covariance_X(2, 2);
                //     covariance.GetCovarianceBlock(&MLP[i][0], &MLP[j][0], covariance_X.data());

                //     Cx.block<2,2>(i*2 + 6*EOP.size()+3*XYZ.size()+3*IOP.size()+7*AP.size(),j*2 + 6*EOP.size()+3*XYZ.size()+3*IOP.size()+7*AP.size()) = covariance_X;
                // }
            }
        }

        // copy it to make a symmetrical matrix
        Cx.triangularView<Eigen::Lower>() = Cx.transpose();

        if (true)
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

        PyRun_SimpleString("print 'Done computing covariance matrix:', round(TIME.clock()-t0, 3), 's' ");

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
        problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residuals, NULL, &jacobian);
        Eigen::MatrixXd imageResiduals(imageX.size(), 2);
        for (int n = 0; n<imageX.size(); n++)
        {
            imageResiduals(n,0) = residuals[2*n] * imageXStdDev[n];
            imageResiduals(n,1) = residuals[2*n+1] * imageYStdDev[n];
        }
        if(DEBUGMODE)
        {
            std::cout<<"Residuals:"<<std::endl;
            std::cout<<imageResiduals<<std::endl;
        }


        PyRun_SimpleString("t0 = TIME.clock()");        
        PyRun_SimpleString("print 'Start computing covariance matrix of the residuals' ");  

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

            Eigen::MatrixXd A(jacobian.num_rows,jacobian.num_cols);
            A.setZero();
            for (int i = 0; i < jacobian.num_rows; i++)
            {
                for (int j = jacobian.rows[i]; j < jacobian.rows[i+1]; j++)
                {
                A(i,jacobian.cols[j]) = jacobian.values[j]; 
                }
            }

            std::cout<<"    Writing A to file..."<<std::endl;
            FILE *fout = fopen("A.jck", "w");
            for(int i = 0; i < A.rows(); ++i)
            {
                for(int j = 0; j < A.cols(); ++j)
                {
                    fprintf(fout, "%.6lf \t ", A(i,j));
                }
                fprintf(fout, "\n");
            }
            fclose(fout);
        }



        PyRun_SimpleString("print 'Done computing covariance matrix of the residuals:', round(TIME.clock()-t0, 3), 's' ");


        // Output results to file
        PyRun_SimpleString("t0 = TIME.clock()");        
        PyRun_SimpleString("print 'Start outputting bundle adjustment results to file' ");     
        //Output results back to Python for plotting
        if (true)
        {
            std::cout<<"  Writing residuals to file..."<<std::endl;
            FILE *fout = fopen("image.jck", "w");
            for(int i = 0; i < imageTarget.size(); ++i)
            {
                fprintf(fout, "%i %.6lf %.6lf %.6lf %.6lf\n", imageReferenceID[i], imageX[i], imageY[i], imageResiduals(i,0), imageResiduals(i,1));
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

        PyRun_SimpleString("print 'Done outputting bundle adjustment results to file:', round(TIME.clock()-t0, 3), 's' ");


        PyRun_SimpleString("t0 = TIME.clock()");        
        PyRun_SimpleString("print 'Start doing machine learning in Python' ");    

        system("python ~/BundleAdjustment/python/gaussianProcess.py");

        PyRun_SimpleString("print 'Done doing machine learning regression:', round(TIME.clock()-t0, 3), 's' ");

    }

    if (true)
        {
            std::cout<<"  Writing least squares costs to file..."<<std::endl;
            FILE *fout = fopen("costs.jck", "w");
            for(int i = 0; i < leastSquaresCost.size(); ++i)
            {
                fprintf(fout, "%i %.6lf\n", i, leastSquaresCost[i]);
            }
            fclose(fout);
        }   

    PyRun_SimpleString("print '----------------------------Program Successful ^-^, Total Run Time:', round(TIME.clock()-totalTime, 3), 's', '----------------------------', ");
    return 0;
}
