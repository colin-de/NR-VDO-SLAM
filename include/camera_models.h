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
#ifndef DY_NR_SLAM_CAM_MODEL_H
#define DY_NR_SLAM_CAM_MODEL_H

#include <memory>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

template <class T>
class AbstractCamera;

template <typename T>
Eigen::Matrix<T, 2, 1> project(const Eigen::Matrix<T, 3, 1>& p) {
  const double fx = 519.9338989257812500;
  const double fy = 519.9338989257812500;
  const double cx = 300.0;
  const double cy = 250.0;

  const T& x = p[0];
  const T& y = p[1];
  const T& z = p[2];

  Eigen::Matrix<T, 2, 1> res;

  T u, v;
  u = fx * x / z + cx;
  v = fy * y / z + cy;
  res << u, v;
  return res;
}

template <typename T>
Eigen::Matrix<T, 2, 1> project2(const Eigen::Matrix<T, 3, 1>& p) {
  const double fx = 800.0;
  const double fy = 800.0;
  const double cx = 320.0;
  const double cy = 240.0;

  const T& x = p[0];
  const T& y = p[1];
  const T& z = p[2];

  Eigen::Matrix<T, 2, 1> res;

  T u, v;
  u = fx * x / z + cx;
  v = fy * y / z + cy;
  res << u, v;
  return res;
}

template <typename T>
Eigen::Matrix<T, 3, 1> unproject(const Eigen::Matrix<T, 2, 1>& p,
                                 double depth) {
  const double fx = 519.9338989257812500;
  const double fy = 519.9338989257812500;
  const double cx = 300.0;
  const double cy = 250.0;

  Eigen::Matrix<T, 3, 1> res;

  T u, v, mx, my;
  u = p(0);
  v = p(1);
  mx = (u - cx) / fx;
  my = (v - cy) / fy;
  res << depth * mx, depth * my, depth * T(1);
  return res;
}

template <typename Scalar>
class AbstractCamera;

template <typename Scalar = double>
class PinholeCamera : public AbstractCamera<Scalar> {
 public:
  static constexpr size_t N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  PinholeCamera() = default;
  PinholeCamera(const VecN& p) : param(p) {}

  static PinholeCamera<Scalar> getTestProjections() {
    VecN vec1;
    vec1 << 0.5 * 805, 0.5 * 800, 505, 509, 0, 0, 0, 0;
    PinholeCamera<Scalar> res(vec1);

    return res;
  }

  Scalar* data() { return param.data(); }

  const Scalar* data() const { return param.data(); }

  static std::string getName() { return "pinhole"; }
  std::string name() const { return getName(); }

  virtual Vec2 project(const Vec3& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];

    const Scalar& x = p[0];
    const Scalar& y = p[1];
    const Scalar& z = p[2];

    Vec2 res;

    Scalar u, v;
    u = fx * x / z + cx;
    v = fy * y / z + cy;
    res << u, v;
    return res;
  }

  virtual Vec3 unproject(const Vec2& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];

    Vec3 res;

    Scalar u, v, mx, my, num;
    u = p(0);
    v = p(1);
    mx = (u - cx) / fx;
    my = (v - cy) / fy;
    num = Scalar(1) / sqrt(mx * mx + my * my + Scalar(1));
    res << num * mx, num * my, num * Scalar(1);
    return res;
  }

  const VecN& getParam() const { return param; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param = VecN::Zero();
};

template <typename Scalar>
class AbstractCamera {
 public:
  static constexpr size_t N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  virtual ~AbstractCamera() = default;

  virtual Scalar* data() = 0;

  virtual const Scalar* data() const = 0;

  virtual Vec2 project(const Vec3& p) const = 0;

  virtual Vec3 unproject(const Vec2& p) const = 0;

  virtual std::string name() const = 0;

  virtual const VecN& getParam() const = 0;

  inline int width() const { return width_; }
  inline int& width() { return width_; }
  inline int height() const { return height_; }
  inline int& height() { return height_; }

  static std::shared_ptr<AbstractCamera> from_data(const std::string& name,
                                                   const Scalar* sIntr) {
    if (name == PinholeCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);
      return std::shared_ptr<AbstractCamera>(new PinholeCamera<Scalar>(intr));
    } else {
      std::cerr << "Camera model " << name << " is not implemented."
                << std::endl;
      std::abort();
    }
  }

  static std::shared_ptr<AbstractCamera> initialize(const std::string& name,
                                                    const Scalar* sIntr) {
    Eigen::Matrix<Scalar, 8, 1> init_intr;

    if (name == PinholeCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);

      init_intr = intr;
      init_intr.template tail<4>().setZero();

      return std::shared_ptr<AbstractCamera>(
          new PinholeCamera<Scalar>(init_intr));
    } else {
      std::cerr << "Camera model " << name << " is not implemented."
                << std::endl;
      std::abort();
    }
  }

 private:
  // image dimensions
  int width_ = 0;
  int height_ = 0;
};

#endif  // DY_NR_SLAM_CAM_MODEL_H
