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
    inp.open("/home/jckchow/BundleAdjustment/Data/Dcs28mm.iop");
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
    inp.open("/home/jckchow/BundleAdjustment/Data/Dcs28mm.xyz");
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
    std::cout << "      Data: " <<std::endl;
    for (int i = 0; i < xyzTarget.size(); i++)
    {
        std::cout<<xyzTarget[i]<<" \t "<<xyzX[i]<<" \t "<<xyzY[i]<<" \t "<<xyzZ[i]<<" \t "<<xyzXStdDev[i]<<" \t "<<xyzYStdDev[i]<<" \t "<<xyzZStdDev[i]<<std::endl;
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
   
    ceres::Problem problem;
    ceres::LossFunction* loss = NULL;
    // loss = new ceres::HuberLoss(1.0);
    // loss = new ceres::CauchyLoss(0.5);

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

        //  std::cout<<"EOP: "<< EOP[indexPose][3] <<", " << EOP[indexPose][4] <<", " << EOP[indexPose][5]  <<std::endl;
        //  std::cout<<"XYZ: "<< XYZ[indexPoint][0] <<", " << XYZ[indexPoint][1] <<", " << XYZ[indexPoint][2]  <<std::endl;

        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<collinearity, 2, 6, 3, 3, 7>(
                new collinearity(imageX[n],imageY[n],imageXStdDev[n], imageYStdDev[n],iopXp[indexSensor],iopYp[indexSensor]));
        problem.AddResidualBlock(cost_function, NULL, &EOP[indexPose][0], &XYZ[indexPoint][0], &IOP[indexSensor][0], &AP[indexSensor][0]);  

        problem.SetParameterBlockConstant(&IOP[indexSensor][0]);
        problem.SetParameterBlockConstant(&AP[indexSensor][0]);
 
    }
 
    // define the datum by pseduo observations of the positions
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

    ceres::Solver::Options options;
    options.max_num_iterations = 50;
    // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; // sparse solver
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
	options.max_lm_diagonal = 1.0E-150; // force it behave like a Gauss-Newton update
    options.min_lm_diagonal = 1.0E-150;
    // options.function_tolerance = 1.0E-20;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
    std::cout << summary.FullReport() << "\n";

    std::cout<<"After XYZ"<<std::endl;
    for (int n = 0; n < XYZ.size(); n++)
    {
        std::cout<<xyzTarget[n]<<": "<<XYZ[n][0]<<", "<<XYZ[n][1]<<", "<<XYZ[n][2]<<std::endl;
    }
    std::cout<<"After EOP"<<std::endl;
    for (int n = 0; n < EOP.size(); n++)
    {
        std::cout<<eopStation[n]<<": "<<EOP[n][0]*180.0/PI<<", "<<EOP[n][1]*180.0/PI<<", "<<EOP[n][2]*180.0/PI<<", "<<EOP[n][3]<<", "<<EOP[n][4]<<", "<<EOP[n][5]<<std::endl;
    }
    std::cout<<"After IOP"<<std::endl;
    for (int n = 0; n < iopCamera.size(); n++)
    {
        std::cout<<iopCamera[n]<<": "<<IOP[n][0]<<", "<<IOP[n][1]<<", "<<IOP[n][2]<<std::endl;
    }
    std::cout<<"After AP"<<std::endl;
    for (int n = 0; n < iopCamera.size(); n++)
    {
        std::cout<<iopCamera[n]<<": "<<AP[n][0]<<", "<<AP[n][1]<<", "<<AP[n][2]<<AP[n][3]<<", "<<AP[n][4]<<", "<<AP[n][5]<<", "<<AP[n][6]<<std::endl;
    }

    ceres::Covariance::Options covarianceOoptions;
    ceres::Covariance covariance(covarianceOoptions);

    std::vector<std::pair<const double*, const double*> > covariance_blocks;
    // for(int i = 0; i < XYZ.size(); i++)
    //     for(int j = 0; j < XYZ.size(); j++)
            // covariance_blocks.push_back(std::make_pair(&XYZ[i][0], &XYZ[j][0]));
    covariance_blocks.push_back(std::make_pair(&XYZ[0][0], &XYZ[0][0]));
    covariance_blocks.push_back(std::make_pair(&XYZ[1][0], &XYZ[1][0]));
    covariance_blocks.push_back(std::make_pair(&XYZ[2][0], &XYZ[2][0]));

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


    Eigen::MatrixXd covariance_X(3, 3);
    covariance.GetCovarianceBlock(&XYZ[0][0], &XYZ[0][0], covariance_X.data());
     std::cout << "Variance: " << covariance_X.diagonal() << std::endl; 
     std::cout<<"covariance matrix: "<<std::endl;
     std::cout<<covariance_X<<std::endl;

    covariance.GetCovarianceBlock(&XYZ[1][0], &XYZ[1][0], covariance_X.data());
     std::cout << "Variance: " << covariance_X.diagonal() << std::endl; 
     std::cout<<"covariance matrix: "<<std::endl;
     std::cout<<covariance_X<<std::endl;

    covariance.GetCovarianceBlock(&XYZ[2][0], &XYZ[2][0], covariance_X.data());
     std::cout << "Variance: " << covariance_X.diagonal() << std::endl; 
     std::cout<<"covariance matrix: "<<std::endl;
     std::cout<<covariance_X<<std::endl;

    std::vector<double> residuals;
    double cost = 0.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residuals, NULL, NULL);

    Eigen::MatrixXd imageResiduals(imageX.size(), 2);
    std::cout<<"Residuals:"<<std::endl;
    for (int n = 0; n<imageX.size(); n++)
    {
        imageResiduals(n,0) = residuals[2*n] * imageXStdDev[n];
        imageResiduals(n,1) = residuals[2*n+1] * imageYStdDev[n];
    }
    std::cout<<imageResiduals<<std::endl;

    PyRun_SimpleString("print 'building Ceres-Solver cost functions:', round(TIME.clock()-t0, 3), 's' ");

    return 0;
}
