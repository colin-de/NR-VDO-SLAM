/**
BSD 3-Clause License

Copyright (c) 2022, Weihang Li.
All rights reserved.

Redistribution and use in source and binary forms, with or
without modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
                                               SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef DY_NR_SLAM_LOSS_H
#define DY_NR_SLAM_LOSS_H

#pragma once

#include <memory>

#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <utility>
#include <common_types.h>
#include <ceres/ceres.h>
#include <ceres/loss_function.h>
#include <thread>
#include "camera_models.h"
#include "local_parameterization_se3.hpp"

struct BundleAdjustmentOptions {
  /// 0: silent, 1: ceres brief report (one line), 2: ceres full report
  int verbosity_level = 1;

  /// update intrinsics or keep fixed
  bool optimize_intrinsics = false;

  /// use huber robust norm or squared norm
  bool use_huber = true;

  /// parameter for huber loss (in pixel)
  double huber_parameter = 1.0;

  /// maximum number of solver iterations
  int max_num_iterations = 5000;
};

struct BundleAdjustmentCostFunctorNoDef {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BundleAdjustmentCostFunctorNoDef(Eigen::Vector2d pn1, Eigen::Vector3d P_o,
                                   Eigen::Vector2d opt_f)
      : pn1(std::move(pn1)), P_o(std::move(P_o)), opt_f(std::move(opt_f)) {}

  template <class T>
  bool operator()(T const* const sCam_0, T const* const sT_o,
                  T const* const sT_c, T* sResiduals) const {
    // map inputs
    Eigen::Map<Sophus::SE3<T> const> const Cam_0(sCam_0);
    Eigen::Map<Sophus::SE3<T> const> const T_o(sT_o);
    Eigen::Map<Sophus::SE3<T> const> const T_c(sT_c);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(sResiduals);
    Eigen::Vector2d identity(1.0, 1.0);
    double lambda(50.0);
    UNUSED(lambda);

    residuals = (project(T_c * (Cam_0.inverse() * T_o * P_o)) - (pn1 + opt_f));

    //    std::cout << "residuals: " << residuals << " finished" << std::endl;

    return true;
  }

  Eigen::Vector2d pn1;
  Eigen::Vector3d P_o;
  Eigen::Vector2d opt_f;
};

struct BundleAdjustmentCostFunctorStatic {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BundleAdjustmentCostFunctorStatic(Eigen::Vector2d pn1, Eigen::Vector3d P,
                                    Eigen::Vector2d opt_f)
      : pn1(std::move(pn1)), P(std::move(P)), opt_f(std::move(opt_f)) {}

  template <class T>
  bool operator()(T const* const sCam_0, T const* const sT_c,
                  T* sResiduals) const {
    // map inputs
    Eigen::Map<Sophus::SE3<T> const> const Cam_0(sCam_0);
    Eigen::Map<Sophus::SE3<T> const> const T_c(sT_c);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(sResiduals);
    double lambda(1.0);
    UNUSED(lambda);

    residuals = lambda * (project(T_c * (Cam_0.inverse() * P)) - (pn1 + opt_f));

    return true;
  }

  Eigen::Vector2d pn1;
  Eigen::Vector3d P;  // 3d points in world frame
  Eigen::Vector2d opt_f;
};

// This is cost functor for synthetic dataset
struct BundleAdjustmentCostFunctorSyn {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BundleAdjustmentCostFunctorSyn(Eigen::Vector2d pn1, Eigen::Vector3d P_o,
                                 Eigen::Vector2d opt_f)
      : pn1(std::move(pn1)), P_o(std::move(P_o)), opt_f(std::move(opt_f)) {}

  template <class T>
  bool operator()(T const* const sCam_0, T* sResiduals) const {
    // map inputs
    Eigen::Map<Sophus::SE3<T> const> const Cam_1(sCam_0);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(sResiduals);

    residuals = (project2(Cam_1 * P_o) - (pn1 + opt_f));

    //    std::cout << "residuals: " << residuals << " finished" << std::endl;

    return true;
  }

  Eigen::Vector2d pn1;
  Eigen::Vector3d P_o;
  Eigen::Vector2d opt_f;
};

struct BundleAdjustmentCostFunctorReg {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BundleAdjustmentCostFunctorReg(Eigen::Vector2d pn1, Eigen::Vector3d P_o,
                                 Eigen::Vector2d opt_f)
      : pn1(std::move(pn1)), P_o(std::move(P_o)), opt_f(std::move(opt_f)) {}

  template <class T>
  bool operator()(T const* const sCam_0, T const* const sT_o,
                  T const* const sT_c, T const* const sV, T* sResiduals) const {
    // map inputs
    Eigen::Map<Sophus::SE3<T> const> const Cam_0(sCam_0);
    Eigen::Map<Sophus::SE3<T> const> const T_o(sT_o);
    Eigen::Map<Sophus::SE3<T> const> const T_c(sT_c);
    Eigen::Map<Eigen::Matrix<T, 3, 1> const> const V(sV);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(sResiduals);
    Eigen::Vector2d identity(1.0, 1.0);
    double lambda(1.0);

    residuals = lambda * (project(T_c * (Cam_0.inverse() * T_o * P_o + V)) -
                          (pn1 + opt_f));

    //    std::cout << "residuals: " << residuals << " finished" << std::endl;

    return true;
  }

  Eigen::Vector2d pn1;
  Eigen::Vector3d P_o;
  Eigen::Vector2d opt_f;
};

struct VectorFieldRegCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  VectorFieldRegCostFunctor(double K, double inv_P, double num_pts)
      : K(K), inv_P(inv_P), num_pts(num_pts) {}

  template <class T>
  bool operator()(T const* const sV, T* sResiduals) const {
    // map inputs
    Eigen::Map<Eigen::Matrix<T, 3, 1> const> const V(sV);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(sResiduals);
    Eigen::Vector2d identity(1.0, 1.0);
    double lambda_vf(1.0);

    residuals = lambda_vf * identity * (K * pow((inv_P + K), 2.0)) * V.norm();

    return true;
  }

  double K;
  double inv_P;
  double num_pts;
};

struct VectorFieldFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  VectorFieldFunctor(Eigen::MatrixXd& K, Eigen::MatrixXd& inv_P)
      : K(K), inv_P(inv_P) {}

  template <class T>
  bool operator()(T const* const sV, T* sResiduals) const {
    // map inputs
    Eigen::Map<Eigen::Matrix<T, 50, 3> const> const V(sV);
    Eigen::Map<Eigen::Matrix<T, 1, 1>> residuals(sResiduals);
    double lambda_vf(5.0);
    UNUSED(lambda_vf);

    residuals = (V.col(0).transpose() * (inv_P + K).inverse() * K *
                     (inv_P + K).inverse() * V.col(0) +
                 V.col(1).transpose() * (inv_P + K).inverse() * K *
                     (inv_P + K).inverse() * V.col(1) +
                 V.col(2).transpose() * (inv_P + K).inverse() * K *
                     (inv_P + K).inverse() * V.col(2)) *
                lambda_vf;
    // firstly ignore the division of num_pts for debugging
    //    std::cout << "reseiduals: " << residuals << std::endl;

    return true;
  }

  Eigen::MatrixXd K;
  Eigen::MatrixXd inv_P;
};

struct VectorNormRegCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  VectorNormRegCostFunctor(Eigen::Vector2d pn1, Eigen::Vector3d P_o,
                           Eigen::Vector2d opt_f, double K, double inv_P,
                           double num_pts)
      : pn1(std::move(pn1)),
        P_o(std::move(P_o)),
        opt_f(std::move(opt_f)),
        K(K),
        inv_P(inv_P),
        num_pts(num_pts) {}

  template <class T>
  bool operator()(T const* const sCam_0, T const* const sT_o,
                  T const* const sT_c, T const* const sV, T* sResiduals) const {
    // map inputs
    Eigen::Map<Sophus::SE3<T> const> const Cam_0(sCam_0);
    Eigen::Map<Sophus::SE3<T> const> const T_o(sT_o);
    Eigen::Map<Sophus::SE3<T> const> const T_c(sT_c);
    Eigen::Map<Eigen::Matrix<T, 3, 1> const> const V(sV);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(sResiduals);
    Eigen::Vector2d identity(1.0, 1.0);
    double lambda_vn(3.0);

    residuals = lambda_vn * identity * V.norm();

    return true;
  }

  Eigen::Vector2d pn1;
  Eigen::Vector3d P_o;
  Eigen::Vector2d opt_f;
  double K;
  double inv_P;
  double num_pts;
};

void bundle_adjustment(const BundleAdjustmentOptions& options,
                       Sophus::SE3d& Cam0, Sophus::SE3d& T_obj,
                       Sophus::SE3d& T_cam, ObjectPoints& Obj_ps,
                       ObjectPoints& Control_ps, StaticPoints& Static_ps,
                       Eigen::MatrixXd& K, Eigen::MatrixXd& inv_P) {
  ceres::Problem problem;

  ceres::HuberLoss* loss_function = nullptr;
  if (options.use_huber) {
    loss_function = new ceres::HuberLoss(options.huber_parameter);
  }

  int num_pts = int(Obj_ps.size());
  UNUSED(num_pts);

  int control_pts = int(Control_ps.size());
  int deform_mat_param = 3 * control_pts;
  UNUSED(deform_mat_param);

  problem.AddParameterBlock(Cam0.data(), Sophus::SE3d::num_parameters,
                            new Sophus::test::LocalParameterizationSE3);
  problem.SetParameterBlockConstant(Cam0.data());

  problem.AddParameterBlock(T_obj.data(), Sophus::SE3d::num_parameters,
                            new Sophus::test::LocalParameterizationSE3);

  problem.AddParameterBlock(T_cam.data(), Sophus::SE3d::num_parameters,
                            new Sophus::test::LocalParameterizationSE3);

  /// FIX T_cam and T_obj
//    problem.SetParameterBlockConstant(T_obj.data());
  //  problem.SetParameterBlockConstant(T_cam.data());

  // Prepare for vector field regularization
  Eigen::MatrixXd V_mat;
  V_mat.resize(int(Control_ps.size()), 3);
  for (size_t i = 0; i < Control_ps.size(); i++) {
    V_mat.row(i) = Control_ps[i].V.transpose();
  }
//  std::cout << "V_mat: \n" << V_mat << "\n";

  for (size_t i = 0; i < Obj_ps.size(); i++) {
    auto& P_o = Obj_ps.at(i).Pos;
    auto& pn1 = Obj_ps.at(i).pn1;
    auto& pn2 = Obj_ps.at(i).pn2;
    auto& depth = Obj_ps.at(i).depth;
    auto& opt_f = Obj_ps.at(i).f;
    auto& id = Obj_ps.at(i).id;
    Obj_ps.at(i).V = Control_ps.at(id - 1).V;

    UNUSED(P_o);
    UNUSED(pn1);
    UNUSED(pn2);
    UNUSED(depth);

    auto Cam_0 = Cam0.data();
    auto T_o = T_obj.data();  // Object Pose w.r.t. world
    auto T_c = T_cam.data();  // Camera Transformation between frames
    auto V = Control_ps.at(id - 1).V.data();  // Object non-rigid Deformation
    problem.AddParameterBlock(V, 3);
    //  problem.SetParameterBlockConstant(V);

    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<
            BundleAdjustmentCostFunctorReg, 2, Sophus::SE3d::num_parameters,
            Sophus::SE3d::num_parameters, Sophus::SE3d::num_parameters, 3>(
            new BundleAdjustmentCostFunctorReg(pn1, P_o, opt_f)),
        loss_function, Cam_0, T_o, T_c, V);

    // Add Vector Field to the problem
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<VectorFieldFunctor, 1, 150>(
            new VectorFieldFunctor(K, inv_P)),
        loss_function, V_mat.data());

    // Add Vector Norm to the problem
    double K_value = 1.0;
    double inv_P_value = 0.03;
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<
            VectorNormRegCostFunctor, 2, Sophus::SE3d::num_parameters,
            Sophus::SE3d::num_parameters, Sophus::SE3d::num_parameters, 3>(
            new VectorNormRegCostFunctor(pn1, P_o, opt_f, K_value, inv_P_value,
                                         num_pts)),
        loss_function, Cam_0, T_o, T_c, V);
  }

  for (size_t i = 0; i < Static_ps.size(); i++) {
    auto& P = Static_ps.at(i).Pos;
    auto& pn1 = Static_ps.at(i).pn1;
    auto& pn2 = Static_ps.at(i).pn2;
    auto& opt_f = Static_ps.at(i).f;

    UNUSED(pn2);

    auto Cam_0 = Cam0.data();
    auto T_c = T_cam.data();  // Camera Transformation between frames

    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<BundleAdjustmentCostFunctorStatic, 2,
                                        Sophus::SE3d::num_parameters,
                                        Sophus::SE3d::num_parameters>(
            new BundleAdjustmentCostFunctorStatic(pn1, P, opt_f)),
        loss_function, Cam_0, T_c);
  }

  // Solve
  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = options.max_num_iterations;
  ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
  //    ceres_options.linear_solver_type = ceres::DENSE_QR;
  ceres_options.num_threads = std::thread::hardware_concurrency();
  ceres::Solver::Summary summary;
  Solve(ceres_options, &problem, &summary);
  switch (options.verbosity_level) {
    // 0: silent
    case 1:
      std::cout << summary.BriefReport() << std::endl;
      break;
    case 2:
      std::cout << summary.FullReport() << std::endl;
      break;
  }
}

[[maybe_unused]] void bundle_adjustment_old(
    const BundleAdjustmentOptions& options, Sophus::SE3d& Cam0,
    Sophus::SE3d& T_obj, Sophus::SE3d& T_cam, ObjectPoints& Obj_ps,
    StaticPoints& Static_ps, Eigen::VectorXd& K_value, double& inv_P) {
  ceres::Problem problem;

  ceres::HuberLoss* loss_function = nullptr;
  if (options.use_huber) {
    loss_function = new ceres::HuberLoss(options.huber_parameter);
  }

  int num_pts = int(Obj_ps.size());
  UNUSED(num_pts);

  problem.AddParameterBlock(Cam0.data(), Sophus::SE3d::num_parameters,
                            new Sophus::test::LocalParameterizationSE3);
  problem.SetParameterBlockConstant(Cam0.data());

  problem.AddParameterBlock(T_obj.data(), Sophus::SE3d::num_parameters,
                            new Sophus::test::LocalParameterizationSE3);

  problem.AddParameterBlock(T_cam.data(), Sophus::SE3d::num_parameters,
                            new Sophus::test::LocalParameterizationSE3);

  /// FIX T_cam and T_obj
  //  problem.SetParameterBlockConstant(T_obj.data());
  //  problem.SetParameterBlockConstant(T_cam.data());

  for (size_t i = 0; i < Obj_ps.size(); i++) {
    auto& P_o = Obj_ps.at(i).Pos;
    auto& pn1 = Obj_ps.at(i).pn1;
    auto& pn2 = Obj_ps.at(i).pn2;
    auto& depth = Obj_ps.at(i).depth;
    auto& opt_f = Obj_ps.at(i).f;
    auto& K = K_value(i);

    UNUSED(P_o);
    UNUSED(pn1);
    UNUSED(pn2);
    UNUSED(depth);

    auto Cam_0 = Cam0.data();
    auto T_o = T_obj.data();         // Object Pose w.r.t. world
    auto T_c = T_cam.data();         // Camera Transformation between frames
    auto V = Obj_ps.at(i).V.data();  // Object non-rigid Deformation

    problem.AddParameterBlock(V, 3);
    //  problem.SetParameterBlockConstant(V);

    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<
            BundleAdjustmentCostFunctorReg, 2, Sophus::SE3d::num_parameters,
            Sophus::SE3d::num_parameters, Sophus::SE3d::num_parameters, 3>(
            new BundleAdjustmentCostFunctorReg(pn1, P_o, opt_f)),
        loss_function, Cam_0, T_o, T_c, V);

    // Add Vector Field to the problem
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<VectorFieldRegCostFunctor, 2, 3>(
            new VectorFieldRegCostFunctor(K, inv_P, num_pts)),
        loss_function, V);

    // Add Vector Norm to the problem
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<
            VectorNormRegCostFunctor, 2, Sophus::SE3d::num_parameters,
            Sophus::SE3d::num_parameters, Sophus::SE3d::num_parameters, 3>(
            new VectorNormRegCostFunctor(pn1, P_o, opt_f, K, inv_P, num_pts)),
        loss_function, Cam_0, T_o, T_c, V);
  }

  for (size_t i = 0; i < Static_ps.size(); i++) {
    auto& P = Static_ps.at(i).Pos;
    auto& pn1 = Static_ps.at(i).pn1;
    auto& pn2 = Static_ps.at(i).pn2;
    auto& opt_f = Static_ps.at(i).f;

    UNUSED(pn2);

    auto Cam_0 = Cam0.data();
    auto T_c = T_cam.data();  // Camera Transformation between frames

    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<BundleAdjustmentCostFunctorStatic, 2,
                                        Sophus::SE3d::num_parameters,
                                        Sophus::SE3d::num_parameters>(
            new BundleAdjustmentCostFunctorStatic(pn1, P, opt_f)),
        loss_function, Cam_0, T_c);
  }

  // Solve
  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = options.max_num_iterations;
  ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
  //    ceres_options.linear_solver_type = ceres::DENSE_QR;
  ceres_options.num_threads = std::thread::hardware_concurrency();
  ceres::Solver::Summary summary;
  Solve(ceres_options, &problem, &summary);
  switch (options.verbosity_level) {
    // 0: silent
    case 1:
      std::cout << summary.BriefReport() << std::endl;
      break;
    case 2:
      std::cout << summary.FullReport() << std::endl;
      break;
  }
}

// Bundle Adjustment without Deformation
void bundle_adjustment_no_def(const BundleAdjustmentOptions& options,
                              Sophus::SE3d& Cam0, Sophus::SE3d& T_obj,
                              Sophus::SE3d& T_cam, ObjectPoints& Obj_ps,
                              StaticPoints& Static_ps) {
  ceres::Problem problem;

  ceres::HuberLoss* loss_function = nullptr;
  if (options.use_huber) {
    loss_function = new ceres::HuberLoss(options.huber_parameter);
  }

  int num_pts = int(Obj_ps.size());
  UNUSED(num_pts);

  problem.AddParameterBlock(Cam0.data(), Sophus::SE3d::num_parameters,
                            new Sophus::test::LocalParameterizationSE3);
  problem.SetParameterBlockConstant(Cam0.data());

  problem.AddParameterBlock(T_obj.data(), Sophus::SE3d::num_parameters,
                            new Sophus::test::LocalParameterizationSE3);

  problem.AddParameterBlock(T_cam.data(), Sophus::SE3d::num_parameters,
                            new Sophus::test::LocalParameterizationSE3);

  /// FIX T_cam and T_obj
  //  problem.SetParameterBlockConstant(T_obj.data());
  //  problem.SetParameterBlockConstant(T_cam.data());

  for (size_t i = 0; i < Obj_ps.size(); i++) {
    auto& P_o = Obj_ps.at(i).Pos;
    auto& pn1 = Obj_ps.at(i).pn1;
    auto& pn2 = Obj_ps.at(i).pn2;
    auto& depth = Obj_ps.at(i).depth;
    auto& opt_f = Obj_ps.at(i).f;

    UNUSED(P_o);
    UNUSED(pn1);
    UNUSED(pn2);
    UNUSED(depth);

    auto Cam_0 = Cam0.data();
    auto T_o = T_obj.data();  // Object Pose w.r.t. world
    auto T_c = T_cam.data();  // Camera Transformation between frames

    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<
            BundleAdjustmentCostFunctorNoDef, 2, Sophus::SE3d::num_parameters,
            Sophus::SE3d::num_parameters, Sophus::SE3d::num_parameters>(
            new BundleAdjustmentCostFunctorNoDef(pn1, P_o, opt_f)),
        loss_function, Cam_0, T_o, T_c);
  }

  for (size_t i = 0; i < Static_ps.size(); i++) {
    auto& P = Static_ps.at(i).Pos;
    auto& pn1 = Static_ps.at(i).pn1;
    auto& pn2 = Static_ps.at(i).pn2;
    auto& opt_f = Static_ps.at(i).f;

    UNUSED(pn2);

    auto Cam_0 = Cam0.data();
    auto T_c = T_cam.data();  // Camera Transformation between frames

    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<BundleAdjustmentCostFunctorStatic, 2,
                                        Sophus::SE3d::num_parameters,
                                        Sophus::SE3d::num_parameters>(
            new BundleAdjustmentCostFunctorStatic(pn1, P, opt_f)),
        loss_function, Cam_0, T_c);
  }
  // Solve
  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = options.max_num_iterations;
  ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
  //    ceres_options.linear_solver_type = ceres::DENSE_QR;
  ceres_options.num_threads = std::thread::hardware_concurrency();
  ceres::Solver::Summary summary;
  Solve(ceres_options, &problem, &summary);
  switch (options.verbosity_level) {
    // 0: silent
    case 1:
      std::cout << summary.BriefReport() << std::endl;
      break;
    case 2:
      std::cout << summary.FullReport() << std::endl;
      break;
  }
}

// Bundle Adjustment with Synthetic Dataset (only optimize Cam1)
void bundle_adjustment_syn(const BundleAdjustmentOptions& options,
                           Sophus::SE3d& Cam1, ObjectPoints& Obj_ps) {
  ceres::Problem problem;

  ceres::HuberLoss* loss_function = nullptr;
  loss_function = new ceres::HuberLoss(options.huber_parameter);

  int num_pts = int(Obj_ps.size());
  UNUSED(num_pts);

  problem.AddParameterBlock(Cam1.data(), Sophus::SE3d::num_parameters,
                            new Sophus::test::LocalParameterizationSE3);
  //  problem.SetParameterBlockConstant(Cam1.data());

  for (size_t i = 0; i < Obj_ps.size(); i++) {
    auto& P_o = Obj_ps.at(i).Pos;
    auto& pn1 = Obj_ps.at(i).pn1;
    auto& pn2 = Obj_ps.at(i).pn2;
    auto& depth = Obj_ps.at(i).depth;
    auto& opt_f = Obj_ps.at(i).f;

    UNUSED(P_o);
    UNUSED(pn1);
    UNUSED(pn2);
    UNUSED(depth);

    auto Cam_1 = Cam1.data();

    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<BundleAdjustmentCostFunctorSyn, 2,
                                        Sophus::SE3d::num_parameters>(
            new BundleAdjustmentCostFunctorSyn(pn1, P_o, opt_f)),
        loss_function, Cam_1);
  }

  // Solve
  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = options.max_num_iterations;
  ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
  //    ceres_options.linear_solver_type = ceres::DENSE_QR;
  ceres_options.num_threads = std::thread::hardware_concurrency();
  ceres::Solver::Summary summary;
  Solve(ceres_options, &problem, &summary);
  switch (options.verbosity_level) {
    // 0: silent
    case 1:
      std::cout << summary.BriefReport() << std::endl;
      break;
    case 2:
      std::cout << summary.FullReport() << std::endl;
      break;
  }
}

struct BundleAdjustmentCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BundleAdjustmentCostFunctor(Eigen::Vector2d pn1, Eigen::Vector3d P_o,
                              Eigen::Vector2d opt_f)
      : pn1(std::move(pn1)), P_o(std::move(P_o)), opt_f(std::move(opt_f)) {}

  template <class T>
  bool operator()(T const* const sCam_0, T const* const sT_o,
                  T const* const sT_c, T const* const sV, T* sResiduals) const {
    // map inputs
    Eigen::Map<Sophus::SE3<T> const> const Cam_0(sCam_0);
    Eigen::Map<Sophus::SE3<T> const> const T_o(sT_o);
    Eigen::Map<Sophus::SE3<T> const> const T_c(sT_c);
    Eigen::Map<Eigen::Matrix<T, 3, 1> const> const V(sV);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(sResiduals);

    residuals =
        project(T_c * (Cam_0.inverse() * T_o * P_o + V)) - (pn1 + opt_f);

    return true;
  }

  Eigen::Vector2d pn1;
  Eigen::Vector3d P_o;
  Eigen::Vector2d opt_f;
};

#endif  // DY_NR_SLAM_LOSS_H
