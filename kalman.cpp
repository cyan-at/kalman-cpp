/**
* Implementation of KalmanFilter class.
*
* @author: Hayk Martirosyan
* @date: 2014.11.15
*/

#include <iostream>
#include <stdexcept>

#include "kalman.hpp"

KalmanFilter::KalmanFilter(
    double dt,
    const Eigen::MatrixXd& A,
    const Eigen::MatrixXd& H,
    const Eigen::MatrixXd& Q,
    const Eigen::MatrixXd& R,
    const Eigen::MatrixXd& P)
  : A(A), H(H), Q(Q), R(R), P0(P),
    m(H.rows()), n(A.rows()), dt(dt), initialized(false),
    I(n, n), x_hat(n), x_hat_new(n) {
  I.setIdentity();
}

KalmanFilter::KalmanFilter() {}

void KalmanFilter::init(double t0, const Eigen::VectorXd& x0) {
  x_hat = x0;
  P = P0;
  this->t0 = t0;
  t = t0;
  initialized = true;
}

void KalmanFilter::init() {
  x_hat.setZero();
  P = P0;
  t0 = 0;
  t = t0;
  initialized = true;
}

void KalmanFilter::update(const Eigen::VectorXd& y) {
  if (!initialized)
    throw std::runtime_error("Filter is not initialized!");

  // Time Update "Predict" /////////////////////////////////////////////
  // a. project the state ahead
  x_hat_new = A * x_hat;  // no B * u_k-1 + w_k-1

  // b. project the error covariance ahead
  // A relates prior state to new state
  // P *here* is a priori _estimate error covariance_
  // Q is measurement noise covariance
  P = A*P*A.transpose() + Q;

  // Measurement Update "Correct" //////////////////////////////////////
  // a. compute Kalman gain
  K = P*H.transpose()*(H*P*H.transpose() + R).inverse();

  // b. update estimate with measurement y and new kalman gain
  // notice that the Predict does not care about the measurement
  // at all, it is only considered here
  x_hat = x_hat_new + K * (y - H*x_hat_new);

  // c. update error covariance
  P = (I - K*H)*P;

  t += dt;
}

void KalmanFilter::update(
  const Eigen::VectorXd& y,
  double dt,
  const Eigen::MatrixXd A) {
  this->A = A;
  this->dt = dt;
  update(y);
}
