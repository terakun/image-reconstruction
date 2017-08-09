#ifndef RECONSTRUCTOR_H
#define RECONSTRUCTOR_H
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>

class ImageReconstructor{
  double beta_,mu_;
  double epsilon_;

  Eigen::MatrixXd w,u;

  bool check_stop_criterion();
  void compute_w();
  void compute_u();


  public:
  ImageReconstructor(){}
  void set_epsilon(double e){ epsilon_ = e; }
  void operator()(const cv::Mat &src_img,cv::Mat &dst_img);
};

#endif
