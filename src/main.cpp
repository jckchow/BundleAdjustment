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

// Define constants
#define PI 3.141592653589793238462643383279502884197169399


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

  // collinearity condition
  T x = -IOP[2] * XTemp / ZTemp;
  T y = -IOP[2] * YTemp / ZTemp;

  // camera correction model AP = a1, a2, k1, k2, k3, p1, p2, ...
  T x_bar = T(x_) - T(xp_);
  T y_bar = T(y_) - T(yp_);
  T r = sqrt(x_bar*x_bar + y_bar*y_bar);

  T delta_x = x_bar*(AP[2]*pow(r,2.0)+AP[3]*pow(r,4.0)+AP[4]*pow(r,6.0)) + AP[5]*(pow(r,2.0)+2*pow(x_bar,2.0))+2.0*AP[6]*x_bar*y_bar + AP[0]*x_bar+AP[1]*y_bar;
  T delta_y = y_bar*(AP[2]*pow(r,2.0)+AP[3]*pow(r,4.0)+AP[4]*pow(r,6.0)) + AP[6]*(pow(r,2.0)+2*pow(y_bar,2.0))+2.0*AP[5]*x_bar*y_bar;

  T x_true = x + IOP[0] + delta_x;
  T y_true = y + IOP[1] + delta_y;

  // actual cost function
  residual[0] = x_true - T(x_); // x-residual
  residual[1] = y_true - T(y_); // y-residual    

  residual[0] *= T(xStdDev_);
  residual[1] *= T(yStdDev_);

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
struct lidarResidual3DCartesianRelative {
  lidarResidual3DCartesianRelative(double x, double y, double z, double weight)
      : x_(x), y_(y), z_(z), weight_(weight) {}

  template <typename T>
  // unknown parameters followed by the output residual
  bool operator()(const T* const deltaPose, const T* const roll, const T* const pitch, const T* const yaw, const T* const Tx, const T* const Ty, const T* const Tz, const T* const XYZ, T* residual) const {

//   std::cout<<"deltaPose: "<<deltaPose[0]<<", "<<deltaPose[1]<<", "<<deltaPose[2]<<", "<<deltaPose[3]<<", "<<deltaPose[4]<<", "<<deltaPose[5]<<std::endl;

  // rotation from map to sensor
  T r11 = cos(pitch[0]) * cos(yaw[0]);
  T r12 = cos(roll[0]) * sin(yaw[0]) + sin(roll[0]) * sin(pitch[0]) * cos(yaw[0]);
  T r13 = sin(roll[0]) * sin(yaw[0]) - cos(roll[0]) * sin(pitch[0]) * cos(yaw[0]);

  T r21 = -cos(pitch[0]) * sin(yaw[0]);
  T r22 = cos(roll[0]) * cos(yaw[0]) - sin(roll[0]) * sin(pitch[0]) * sin(yaw[0]);
  T r23 = sin(roll[0]) * cos(yaw[0]) + cos(roll[0]) * sin(pitch[0]) * sin(yaw[0]);

  T r31 = sin(pitch[0]);
  T r32 = -sin(roll[0]) * cos(pitch[0]);
  T r33 = cos(roll[0]) * cos(pitch[0]);

  // EOP
  T XTemp = r11 * ( XYZ[0] - Tx[0] ) + r12 * ( XYZ[1] - Ty[0] ) + r13 * ( XYZ[2] - Tz[0] );
  T YTemp = r21 * ( XYZ[0] - Tx[0] ) + r22 * ( XYZ[1] - Ty[0] ) + r23 * ( XYZ[2] - Tz[0] );
  T ZTemp = r31 * ( XYZ[0] - Tx[0] ) + r32 * ( XYZ[1] - Ty[0] ) + r33 * ( XYZ[2] - Tz[0] );

  // boresight rotation matrix
  T m11 = cos(deltaPose[1]) * cos(deltaPose[2]);
  T m12 = cos(deltaPose[0]) * sin(deltaPose[2]) + sin(deltaPose[0]) * sin(deltaPose[1]) * cos(deltaPose[2]);
  T m13 = sin(deltaPose[0]) * sin(deltaPose[2]) - cos(deltaPose[0]) * sin(deltaPose[1]) * cos(deltaPose[2]);

  T m21 = -cos(deltaPose[1]) * sin(deltaPose[2]);
  T m22 = cos(deltaPose[0]) * cos(deltaPose[2]) - sin(deltaPose[0]) * sin(deltaPose[1]) * sin(deltaPose[2]);
  T m23 = sin(deltaPose[0]) * cos(deltaPose[2]) + cos(deltaPose[0]) * sin(deltaPose[1]) * sin(deltaPose[2]);

  T m31 = sin(deltaPose[1]);
  T m32 = -sin(deltaPose[0]) * cos(deltaPose[1]);
  T m33 = cos(deltaPose[0]) * cos(deltaPose[1]);

  // boresight and leverarm
  T x_true =  m11 * ( XTemp - deltaPose[3] ) + m12 * ( YTemp - deltaPose[4] ) + m13 * ( ZTemp - deltaPose[5] );
  T y_true =  m21 * ( XTemp - deltaPose[3] ) + m22 * ( YTemp - deltaPose[4] ) + m23 * ( ZTemp - deltaPose[5] );
  T z_true =  m31 * ( XTemp - deltaPose[3] ) + m32 * ( YTemp - deltaPose[4] ) + m33 * ( ZTemp - deltaPose[5] );

  residual[0] = x_true - T(x_); // x-residual
  residual[1] = y_true - T(y_); // y-residual        
  residual[2] = z_true - T(z_); // z-residual, should be zero if we have a 2D LiDAR

//   std::cout<<"residual: "<<residual[0]<<", "<<residual[1]<<", "<<residual[2]<<std::endl;
//   sleep(1000000);

  residual[0] *= T(weight_);
  residual[1] *= T(weight_);
  residual[2] *= T(1/1.0E-6);

  return true;
  }

 private:
  // Observations for a sample.
  const double x_;
  const double y_;
  const double z_;
  const double weight_;
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
    PyRun_SimpleString("t0 = TIME.clock()");        
     
    PyRun_SimpleString("print 'Start reading data' ");  
    std::ifstream inp;      

    // Reading *.inp file
    PyRun_SimpleString("print '  Start reading image observations' ");  
    inp.open("/home/jckchow/BundleAdjustment/Data/Dcs28mm.pho");
    std::vector<int> imageTarget, imageStation;
    std::vector<double> imageX, imageY, imageXStdDev, imageYStdDev;
    while (true) 
    {
        int c1, c2;
        double c3, c4, c5, c6;
        inp  >> c1 >> c2 >> c3 >> c4 >> c5 >> c6;

        imageTarget.push_back(c1);
        imageStation.push_back(c2);
        imageX.push_back(c3);
        imageY.push_back(c4);
        imageXStdDev.push_back(c5);
        imageYStdDev.push_back(c6);
        if( inp.eof() ) 
            break;
    }
    
    imageTarget.pop_back();
    imageStation.pop_back();
    imageX.pop_back();
    imageY.pop_back();
    imageXStdDev.pop_back();
    imageYStdDev.pop_back();
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



    // Reading *.eop file
    PyRun_SimpleString("print '  Start reading EOPs' ");  
    inp.open("/home/jckchow/BundleAdjustment/Data/Dcs28mm.eop");
    std::vector<int> eopStation, eopCamera;
    std::vector<double> eopXo, eopYo, eopZo, eopOmega, eopPhi, eopKappa;
    while (true) 
    {
        int c1, c2;
        double c3, c4, c5, c6, c7, c8;
        inp  >> c1 >> c2 >> c3 >> c4 >> c5 >> c6 >> c7 >> c8;

        eopStation.push_back(c1);
        eopCamera.push_back(c2);
        eopXo.push_back(c3);
        eopYo.push_back(c4);
        eopZo.push_back(c5);
        eopOmega.push_back(c6);
        eopPhi.push_back(c7);
        eopKappa.push_back(c8);
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

    inp.close();
    std::cout << "    Number of EOPs read: "<< eopStation.size() << std::endl;
    std::vector<int> eopCameraID;
    eopCameraID = eopCamera;
    std::sort (eopCameraID.begin(), eopCameraID.end()); // must sort before the following unique function works
    eopCameraID.erase(std::unique(eopCameraID.begin(), eopCameraID.end()), eopCameraID.end());
    std::cout << "    Number of cameras read: "<< eopCameraID.size() << std::endl;



   // Reading *.iop file
    PyRun_SimpleString("print '  Start reading IOPs' ");  
    inp.open("/home/jckchow/BundleAdjustment/Data/Dcs28mm.iop");
    std::vector<int> iopCamera, iopAxis;
    std::vector<double> iopXMin, iopYMin, iopXMax, iopYMax, iopXp, iopYp, iopC, iopA1, iopA2, iopK1, iopK2, iopK3, iopP1, iopP2;
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

    inp.close();
    std::cout << "    Number of IOPs read: "<< iopCamera.size() << std::endl;

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
    // PyRun_SimpleString("t0 = TIME.clock()");        
    // PyRun_SimpleString("print 'Start building Ceres-Solver cost functions' ");     
   
    // ceres::Problem problem;
    // ceres::LossFunction* loss = NULL;
    // // loss = new ceres::HuberLoss(1.0);
    // // loss = new ceres::CauchyLoss(0.5);

    // if(true)
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

    // ceres::Solver::Options options;
    // options.max_num_iterations = 50;
    // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; // sparse solver
    // options.minimizer_progress_to_stdout = true;
	// options.max_lm_diagonal = 1.0E-150; // force it behave like a Gauss-Newton update
    // options.min_lm_diagonal = 1.0E-150;
    // ceres::Solver::Summary summary;
    // ceres::Solve(options, &problem, &summary);
    // std::cout << summary.BriefReport() << "\n";
    // std::cout << summary.FullReport() << "\n";

    // for(int n = 0; n < X.size(); n++)
    // std::cout<<"final X: "<<X[n]<<std::endl;

    // ceres::Covariance::Options covarianceOoptions;
    // ceres::Covariance covariance(covarianceOoptions);

    // std::vector<std::pair<const double*, const double*> > covariance_blocks;
    // for(int i = 0; i < X.size(); i++)
    //     for(int j = 0; j < X.size(); j++)
    //         covariance_blocks.push_back(std::make_pair(&X[i], &X[j]));

    // covariance.Compute(covariance_blocks, &problem);

    // //double covariance_X[X.size() * X.size()];
    // Eigen::Matrix<double,3,3> covariance_X;
    // // covariance_X.resize(X.size() * X.size());
    // //covariance.GetCovarianceBlock(&X[0], &X[0], covariance_X.data());

    // for(int i = 0; i < X.size(); i++)
    //     for(int j = 0; j < X.size(); j++)
    //     {
    //         double temp[1];
    //         temp[0] = 0.0;
    //         covariance.GetCovarianceBlock(&X[i], &X[j], temp);
    //         std::cout<<"std: "<<sqrt(temp[0])<<std::endl;
    //         covariance_X(i,j) = temp[0];
    //     }

    // // std::cout<<"std: "<<sqrt(covariance_X[0])<<", size: "<<covariance_X.size()<<std::endl;
    // // std::cout<<"std: "<<sqrt(covariance_X[4])<<", size: "<<covariance_X.size()<<std::endl;
    //  std::cout << "Variance: " << covariance_X.diagonal() << std::endl; 

    //  std::cout<<"covariance matrix: "<<std::endl;
    //  std::cout<<covariance_X<<std::endl;

    PyRun_SimpleString("print 'building Ceres-Solver cost functions:', round(TIME.clock()-t0, 3), 's' ");

    return 0;
}
