#ifndef _VECTOR_FIELD_REGULARIZATION_H
#define _VECTOR_FIELD_REGULARIZATION_H

#include <Eigen/Dense>
#include <vector>
#include <iostream>

#define MIN_POINT_NUMBER 5

// using namespace std;
// using namespace Eigen;

// Eigen::MatrixXd V2M(std::vector<Eigen::Vector3d> V);
// std::vector<Eigen::Vector3d> M2V(Eigen::MatrixXd M);
// Eigen::MatrixXd ConstructKernel(Eigen::MatrixXd& pts1, Eigen::MatrixXd& pts2,
//                                double beta);
// double VFNorm(Eigen::MatrixXd& C, Eigen::MatrixXd& pstarts,
//              Eigen::MatrixXd& deforms, double beta, double lambda,
//              double sigma2);
// Eigen::MatrixXd Pred(Eigen::MatrixXd& C, Eigen::MatrixXd& samples,
//                     Eigen::MatrixXd& pstarts, double beta);

Eigen::MatrixXd V2M(std::vector<Eigen::Vector3d> V) {
  Eigen::MatrixXd M(V.size(), 3);
  for (unsigned i = 0; i < V.size(); i++) {
    M(i, 0) = V[i][0];
    M(i, 1) = V[i][1];
    M(i, 2) = V[i][2];
  }
  return M;
}

std::vector<Eigen::Vector3d> M2V(Eigen::MatrixXd M) {
  std::vector<Eigen::Vector3d> V;
  for (unsigned i = 0; i < V.size(); i++) {
    V.emplace_back(M(i, 0), M(i, 1), M(i, 2));
  }
  return V;
}

Eigen::MatrixXd ConstructKernel(Eigen::MatrixXd& pts1, Eigen::MatrixXd& pts2,
                                double beta) {
  unsigned n = pts1.rows();
  unsigned m = pts2.rows();
  Eigen::MatrixXd K(m, n);

  for (unsigned s = 0; s < n; s++) {
    K.col(s) = Eigen::MatrixXd(
        (-beta *
         (pts2.rowwise() - pts1.row(s)).matrix().rowwise().squaredNorm())
            .array()
            .exp());
  }
  return K;
}

double VFNorm(Eigen::MatrixXd& C, Eigen::MatrixXd& pstarts,
              Eigen::MatrixXd& deforms, double beta, double lambda,
              double sigma2) {
  unsigned n = pstarts.rows();
  int dim = 3;
  if (n < MIN_POINT_NUMBER || deforms.rows() != n) return 0;

  //  std::cout << "Step 1: Contruct Kernel.\n";
  Eigen::MatrixXd K = ConstructKernel(pstarts, pstarts, beta);
  //  std::cout << "K: \n" << K << "\n";
  std::cout << "-------------------------------------------\n";
  Eigen::MatrixXd inv_P = Eigen::MatrixXd::Identity(n, n);

  //  std::cout << "Step 2: Solve for C.\n";
  C = (K + lambda * sigma2 * inv_P).colPivHouseholderQr().solve(deforms);
  //  std::cout << "C: \n" << C << "\n";
  std::cout << "-------------------------------------------\n";

  //  std::cout << "Step 3: compute vector field norm.\n";
  double vfnorm = 0;
  for (int i = 0; i < dim; i++) {
    Eigen::MatrixXd col = C.col(i).transpose();
    vfnorm += (col * K * col.transpose())(0, 0);
  }

  //  std::cout << "Vector field norm: " << vfnorm / n << "\n";
  //  std::cout << "-------------------------------------------\n";
  return vfnorm / n;
}

Eigen::MatrixXd Pred(Eigen::MatrixXd& C, Eigen::MatrixXd& samples,
                     Eigen::MatrixXd& pstarts, double beta) {
  unsigned n = pstarts.rows();
  unsigned n_s = samples.rows();
  UNUSED(n);
  UNUSED(n_s);
  Eigen::MatrixXd K = ConstructKernel(samples, pstarts, beta);
  return K * C;
}

#endif
