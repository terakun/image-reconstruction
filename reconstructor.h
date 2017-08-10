#ifndef RECONSTRUCTOR_H
#define RECONSTRUCTOR_H
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>

class ImageReconstructor{
  int img_rows_,img_cols_;
  double beta_,mu_;
  double epsilon_;

  Eigen::MatrixXd w_horizontal_,w_vertical_;
  Eigen::MatrixXd u_;

  double compute_horizontal_diff(const Eigen::MatrixXd &,int r,int c);
  double compute_vertical_diff(const Eigen::MatrixXd &,int r,int c);
  bool check_stop_criterion();
  void compute_w();
  void compute_u();
  
  void fft2dim(Eigen::MatrixXcd &dst_mat,const Eigen::MatrixXcd &src_mat,bool forward=true);

  public:
  ImageReconstructor(){}
  void set_epsilon(double e){ epsilon_ = e; }
  void operator()(const cv::Mat &src_img,cv::Mat &dst_img);
};

#endif
