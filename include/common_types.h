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
#pragma once

#include <bitset>
#include <cstdint>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <sophus/se3.hpp>

#define UNUSED(x) (void)(x)
//#define EXCLUDE_EXPERIMENTAL

using Vec3 = Eigen::Vector3d;
using Vec2 = Eigen::Vector2d;

/// cameras in the map
struct Camera {
  /// camera pose (transforms from camera to world)
  Sophus::SE3d T_w_c;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// dynamic objects in the map
struct ObjectPoint {
  ObjectPoint()
      : Pos(Vec3::Ones()),
        pn1(Vec2::Ones()),
        pn2(Vec2::Ones()),
        V(Vec3::Ones()),
        f(Vec2::Ones()),
        depth(0.0),
        id(0),
        c_point(false){};
  ObjectPoint(Vec3 Pos, Vec2 _pn1, Vec2 _pn2, Vec3 V, Vec2 _f, double _depth,
              int _id, bool _c_point)
      : Pos(std::move(Pos)),
        pn1(std::move(_pn1)),
        pn2(std::move(_pn2)),
        V(std::move(V)),
        f(std::move(_f)),
        depth(_depth),
        id(_id),
        c_point(_c_point){};

  // 3d position in Object frame P’->O28 / P->O24
  Eigen::Vector3d Pos;

  // 2d position in left image coordinates
  Eigen::Vector2d pn1;

  // 2d position in right image coordinates
  Eigen::Vector2d pn2;

  // 3d displacement w.r.t Ct(Camera at t)
  Eigen::Vector3d V;

  // corresponding 2d optical flow
  Eigen::Vector2d f;

  // depth value of pn1
  double depth;

  // belongs to which superpixel
  int id;

  // whether this is a control point
  bool c_point;

  bool operator==(const ObjectPoint& other) const {
    return (Pos == other.Pos) && (V == other.V) && (f == other.f);
  }

  bool operator!=(const ObjectPoint& other) const {
    return (Pos != other.Pos) || (V != other.V) || (f != other.f);
  }
};

/// static objects in the map
struct StaticPoint {
  StaticPoint()
      : Pos(Vec3::Ones()),
        pn1(Vec2::Ones()),
        pn2(Vec2::Ones()),
        f(Vec2::Ones()){};

  StaticPoint(Vec3 Pos, Vec2 _pn1, Vec2 _pn2, Vec2 _f)
      : Pos(std::move(Pos)),
        pn1(std::move(_pn1)),
        pn2(std::move(_pn2)),
        f(std::move(_f)){};

  // 3d position in World frame P’->W28 / P->W24
  Eigen::Vector3d Pos;

  // 2d position in left image coordinates
  Eigen::Vector2d pn1;

  // 2d position in right image coordinates
  Eigen::Vector2d pn2;

  // corresponding 2d optical flow
  Eigen::Vector2d f;
};

using ObjectPoints = std::vector<ObjectPoint>;
using StaticPoints = std::vector<StaticPoint>;
