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
    imageFrameID.erase(std::unique(imageFrameID.begin(), imageFrameID.end()), imageFrameID.end());
    std::cout << "    Number of frames read: "<< imageFrameID.size() << std::endl;


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
