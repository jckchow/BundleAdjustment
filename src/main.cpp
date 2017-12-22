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

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Initialize the unknowns
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////   
    std::vector<double> X;
    X.push_back(1.0);
    X.push_back(10.0);
    X.push_back(100.0);


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Set up cost functions
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    PyRun_SimpleString("t0 = TIME.clock()");        
    PyRun_SimpleString("print 'Start building Ceres-Solver cost functions' ");     
   
    ceres::Problem problem;
    ceres::LossFunction* loss = NULL;
    // loss = new ceres::HuberLoss(1.0);
    // loss = new ceres::CauchyLoss(0.5);

    if(true)
    {
        for(int n = 0; n < X.size(); n++)
        {
            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<constantConstraint, 1, 1>(
                    new constantConstraint(0, 1.0/(double(n+1)*pow(10.0,-4.0))));
            problem.AddResidualBlock(cost_function, NULL, &X[n]);
        }
    }

    for(int n = 0; n < X.size(); n++)
    std::cout<<"before X: "<<X[n]<<std::endl;

    ceres::Solver::Options options;
    options.max_num_iterations = 50;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; // sparse solver
    options.minimizer_progress_to_stdout = true;
	options.max_lm_diagonal = 1.0E-150; // force it behave like a Gauss-Newton update
    options.min_lm_diagonal = 1.0E-150;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
    std::cout << summary.FullReport() << "\n";

    for(int n = 0; n < X.size(); n++)
    std::cout<<"final X: "<<X[n]<<std::endl;

    ceres::Covariance::Options covarianceOoptions;
    ceres::Covariance covariance(covarianceOoptions);

    std::vector<std::pair<const double*, const double*> > covariance_blocks;
    for(int i = 0; i < X.size(); i++)
        for(int j = 0; j < X.size(); j++)
            covariance_blocks.push_back(std::make_pair(&X[i], &X[j]));

    covariance.Compute(covariance_blocks, &problem);

    //double covariance_X[X.size() * X.size()];
    Eigen::Matrix<double,3,3> covariance_X;
    // covariance_X.resize(X.size() * X.size());
    //covariance.GetCovarianceBlock(&X[0], &X[0], covariance_X.data());

    for(int i = 0; i < X.size(); i++)
        for(int j = 0; j < X.size(); j++)
        {
            double temp[1];
            temp[0] = 0.0;
            covariance.GetCovarianceBlock(&X[i], &X[j], temp);
            std::cout<<"std: "<<sqrt(temp[0])<<std::endl;
            covariance_X(i,j) = temp[0];
        }

    // std::cout<<"std: "<<sqrt(covariance_X[0])<<", size: "<<covariance_X.size()<<std::endl;
    // std::cout<<"std: "<<sqrt(covariance_X[4])<<", size: "<<covariance_X.size()<<std::endl;
     std::cout << "Variance: " << covariance_X.diagonal() << std::endl; 

     std::cout<<"covariance matrix: "<<std::endl;
     std::cout<<covariance_X<<std::endl;

    PyRun_SimpleString("print 'building Ceres-Solver cost functions:', round(TIME.clock()-t0, 3), 's' ");

    return 0;
}
