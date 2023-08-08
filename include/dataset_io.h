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
#ifndef DY_NR_SLAM_DATASET_IO_H
#define DY_NR_SLAM_DATASET_IO_H

#include <memory>
#include <Eigen/Dense>
#include "common_types.h"

void load_pose(const std::string& path, Eigen::Matrix<double, 4, 4>& Mat) {
  std::vector<double> matrixEntries;
  std::ifstream matrixDataFile(path);
  std::string matrixRowString;
  std::string matrixEntry;
  int matrixRowNumber = 0;

  while (getline(matrixDataFile, matrixRowString)) {
    std::stringstream matrixRowStringStream(matrixRowString);

    while (getline(matrixRowStringStream, matrixEntry, ',')) {
      matrixEntries.push_back(stod(matrixEntry));
    }
    matrixRowNumber++;
  }

  Mat = Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      matrixEntries.data(), matrixRowNumber,
      matrixEntries.size() / matrixRowNumber);
}

// Load all StaticPoints with their Position and Optical Flow
void load_static_points(
    const std::string& path, StaticPoints& static_ps,
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        static_points,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        static_points_2d_0,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        static_points_2d_1,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        static_points_opt_flows) {
//  const std::string static_points_w = path + "static_o1o2_20.csv";
//  const std::string pix_0_2d_path = path + "feature_static_pix_20.csv";
//  const std::string pix_1_2d_path = path + "feature_static_pix_24.csv";
//  const std::string opt_f_path = path + "optflow_o3_20_24_static.csv";
  const std::string static_points_w = path + "feature_world_20.csv";
  const std::string pix_0_2d_path = path + "feature_pix_20.csv";
  const std::string pix_1_2d_path = path + "feature_pix_24.csv";
  const std::string opt_f_path = path + "optflow_static_20_24.csv";

  std::ifstream f_pos(static_points_w);
  std::ifstream f_o1_0_2d(pix_0_2d_path);
  std::ifstream f_o1_1_2d(pix_1_2d_path);
  std::ifstream f_opt(opt_f_path);

  std::string line;

  {
    // Get Static Points in World frame (P in W20)
    while (std::getline(f_pos, line)) {
      std::stringstream ss(line);
      std::string item;

      Eigen::Vector3d Pos;
      std::getline(ss, item, ',');
      Pos[0] = stod(item);

      std::getline(ss, item, ',');
      Pos[1] = stod(item);

      std::getline(ss, item);
      Pos[2] = stod(item);

      static_points.push_back(Pos);
    }
  }

  {
    // Get Static 2D Points left image
    while (std::getline(f_o1_0_2d, line)) {
      std::stringstream ss(line);
      std::string item;

      Eigen::Vector2d pn1;
      std::getline(ss, item, ',');
      pn1[0] = stod(item);

      std::getline(ss, item);
      pn1[1] = stod(item);

      static_points_2d_0.push_back(pn1);
    }
  }

  {
    // Get Static 2D Points right image
    while (std::getline(f_o1_1_2d, line)) {
      std::stringstream ss(line);
      std::string item;

      Eigen::Vector2d pn2;
      std::getline(ss, item, ',');
      pn2[0] = stod(item);

      std::getline(ss, item);
      pn2[1] = stod(item);

      static_points_2d_1.push_back(pn2);
    }
  }

  {
    // Get Static Points Optical Flow
    while (std::getline(f_opt, line)) {
      std::stringstream ss(line);
      std::string item;

      Eigen::Vector2d Opt_f;
      std::getline(ss, item, ',');
      Opt_f[0] = stod(item);

      std::getline(ss, item);
      Opt_f[1] = stod(item);

      static_points_opt_flows.push_back(Opt_f);
    }
  }

  for (size_t i = 0; i < static_points.size(); i++) {
    StaticPoint static_p(static_points.at(i), static_points_2d_0.at(i),
                         static_points_2d_1.at(i),
                         static_points_opt_flows.at(i));
    static_ps.push_back(static_p);
  }

  std::cout << "There are " << static_ps.size() << " static points"
            << std::endl;
}

// Load all ObjectPoints with their Position, Displacement and Optical Flow
void load_object_points(
    const std::string& path, ObjectPoints& obj_ps,
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        obj_points,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        obj_points_2d_0,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        obj_points_2d_1,
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        obj_points_dis,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        obj_points_opt_flows,
    std::vector<double> obj_depths,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        obj_points_id_control) {
  const std::string o1_0_position_path = path + "feature_o3_o3_20.csv";
  const std::string o1_0_2d_path = path + "feature_o3_pix_20.csv";
  const std::string o1_1_2d_path = path + "feature_o3_pix_24.csv";
  const std::string dis_path = path + "deformation_c_20_changed.csv";
  //  const std::string dis_path = path + "deformation_c_20_gt.csv";
  //  const std::string opt_f_path = path + "optflow_o3_20_24_no_def.csv";
  const std::string opt_f_path = path + "optflow_o3_20_24.csv";
  const std::string o1_0_depth = path + "feature_depth_20.csv";  // UNUSED
  const std::string cluster_path = path + "o3_cluster_50.csv";

  std::ifstream f_pos(o1_0_position_path);
  std::ifstream f_o1_0_2d(o1_0_2d_path);
  std::ifstream f_o1_1_2d(o1_1_2d_path);
  std::ifstream f_dis(dis_path);
  std::ifstream f_opt(opt_f_path);
  std::ifstream f_depth(o1_0_depth);
  std::ifstream f_cluster(cluster_path);

  std::string line;

  {
    // Get Object Points in Object frame (P in O24)
    while (std::getline(f_pos, line)) {
      std::stringstream ss(line);
      std::string item;

      Eigen::Vector3d Pos;
      std::getline(ss, item, ',');
      Pos[0] = stod(item);

      std::getline(ss, item, ',');
      Pos[1] = stod(item);

      std::getline(ss, item);
      Pos[2] = stod(item);

      obj_points.push_back(Pos);
    }
  }

  {
    // Get Object 2D Points left image
    while (std::getline(f_o1_0_2d, line)) {
      std::stringstream ss(line);
      std::string item;

      Eigen::Vector2d pn1;
      std::getline(ss, item, ',');
      pn1[0] = stod(item);

      std::getline(ss, item);
      pn1[1] = stod(item);

      obj_points_2d_0.push_back(pn1);
    }
  }

  {
    // Get Object 2D Points right image
    while (std::getline(f_o1_1_2d, line)) {
      std::stringstream ss(line);
      std::string item;

      Eigen::Vector2d pn2;
      std::getline(ss, item, ',');
      pn2[0] = stod(item);

      std::getline(ss, item);
      pn2[1] = stod(item);

      obj_points_2d_1.push_back(pn2);
    }
  }

  {
    bool gaussianDis = false;
    // Get Object Points Displacement w.r.t. Ct (Def in C24)
    while (std::getline(f_dis, line)) {
      // This is used for randomlization
      // Define random generator with Gaussian distribution
      std::random_device device_random_;
      std::default_random_engine generator_(device_random_());
      std::normal_distribution<> distribution_x_(1, 0.3);
      std::normal_distribution<> distribution_y_(1, 0.3);
      std::normal_distribution<> distribution_z_(1, 0.3);

      //      std::cout << distribution_x_(generator_) << std::endl;
      //      std::cout << distribution_y_(generator_) << std::endl;
      //      std::cout << distribution_z_(generator_) << std::endl;

      std::stringstream ss(line);
      std::string item;

      Eigen::Vector3d Dis;
      std::getline(ss, item, ',');
      //      Dis[0] = stod(item);
      Dis[0] = 1e-3;
      std::getline(ss, item, ',');
      //      Dis[1] = stod(item);
      Dis[1] = 1e-3;
      std::getline(ss, item);
      //      Dis[2] = stod(item);
      Dis[2] = 1e-3;
      if (gaussianDis) {
        Dis(0) *= distribution_x_(generator_);
        Dis(1) *= distribution_y_(generator_);
        Dis(2) *= distribution_z_(generator_);
      }

      obj_points_dis.push_back(Dis);
    }

    {
      std::ofstream os("original_deformation.txt");

      for (const auto& Dis : obj_points_dis) {
        os << std::scientific << Dis(0) << "," << Dis(1) << "," << Dis(2) << ","
           << std::endl;
      }

      std::cout << "finish writing original_deformation.txt" << std::endl;
      os.close();
    }
  }

  {
    // Get Object Points Optical Flow
    while (std::getline(f_opt, line)) {
      std::stringstream ss(line);
      std::string item;

      Eigen::Vector2d Opt_f;
      std::getline(ss, item, ',');
      Opt_f[0] = stod(item);

      std::getline(ss, item);
      Opt_f[1] = stod(item);

      obj_points_opt_flows.push_back(Opt_f);
    }
  }

  {
    // Get Object Points Depth for 2D Points on left image
    while (std::getline(f_depth, line)) {
      std::stringstream ss(line);
      std::string item;

      std::getline(ss, item);
      double depth = stod(item);

      obj_depths.push_back(depth);
    }
  }

  {
    // Get Object Points Superpixel Information
    while (std::getline(f_cluster, line)) {
      std::stringstream ss(line);
      std::string item;

      Eigen::Vector2d id_control;
      std::getline(ss, item, ',');
      id_control[0] = stod(item);

      std::getline(ss, item);
      id_control[1] = stod(item);

      obj_points_id_control.push_back(id_control);
    }
  }
  //  {
  //    // Get Object Points Superpixel Information
  //    while (std::getline(f_cluster, line)) {
  //      std::stringstream ss(line);
  //      std::string item;
  //
  //      std::getline(ss, item, ',');
  //      obj_ids.push_back(stoi(item));
  //
  //      std::getline(ss, item);
  //
  //      obj_controls.push_back(stoi(item));
  //    }

  for (size_t i = 0; i < obj_points.size(); i++) {
    ObjectPoint obj_p(
        obj_points.at(i), obj_points_2d_0.at(i), obj_points_2d_1.at(i),
        obj_points_dis.at(i), obj_points_opt_flows.at(i), obj_depths.at(i),
        obj_points_id_control.at(i)[0], obj_points_id_control.at(i)[1]);
    obj_ps.push_back(obj_p);
  }
  //  for (size_t i = 0; i < obj_points.size(); i++) {
  //    ObjectPoint obj_p(obj_points.at(i), obj_points_2d_0.at(i),
  //                      obj_points_2d_1.at(i), obj_points_dis.at(i),
  //                      obj_points_opt_flows.at(i), obj_depths.at(i),
  //                      obj_points_id_control.at(i)[0],
  //                      obj_points_id_control.at(i)[1]);
  //    obj_ps.push_back(obj_p);
  //  }

  std::cout << "There are " << obj_ps.size() << " object points" << std::endl;
}

Eigen::MatrixXd load_csv(const std::string& path) {
  std::ifstream indata;
  indata.open(path);
  std::string line;
  std::vector<double> values;
  uint rows = 0;
  while (std::getline(indata, line)) {
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, ',')) {
      values.push_back(std::stod(cell));
    }
    ++rows;
  }
  return Eigen::Map<Eigen::MatrixXd>(values.data(), rows, values.size() / rows);
}

template <typename M>
M load_csv(const std::string& path) {
  std::ifstream indata;
  indata.open(path);
  std::string line;
  std::vector<double> values;
  uint rows = 0;
  while (std::getline(indata, line)) {
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, ',')) {
      values.push_back(std::stod(cell));
    }
    ++rows;
  }
  return Eigen::Map<
      const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime,
                          M::ColsAtCompileTime, Eigen::RowMajor>>(
      values.data(), rows, values.size() / rows);
}

#endif  // DY_NR_SLAM_DATASET_IO_H
