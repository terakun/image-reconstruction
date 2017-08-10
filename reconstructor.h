#ifndef RECONSTRUCTOR_H
#define RECONSTRUCTOR_H
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>

template<typename First> 
First max(First first){
}

template<typename First,typename... Rest> 
First max(First first,Rest... rest){
  return std::max(first,max(rest...));
}

class ImageReconstructor{
  int img_rows_,img_cols_;
  double beta_,mu_;
  double epsilon_;

  Eigen::MatrixXd w_horizontal_,w_vertical_;
  Eigen::MatrixXd u_;
  Eigen::MatrixXd observed_img_;

  Eigen::MatrixXcd D_horizontal_fft_;  // difference operator
  Eigen::MatrixXcd D_vertical_fft_;    // difference operator
  Eigen::MatrixXcd K_fft_;             // blurring operator
  Eigen::MatrixXcd observed_img_fft_;

  double compute_horizontal_diff(const Eigen::MatrixXd &,int r,int c);
  double compute_vertical_diff(const Eigen::MatrixXd &,int r,int c);
  Eigen::Vector2d compute_grad(const Eigen::MatrixXd &,int r,int c);

  void blur(Eigen::MatrixXd &,const Eigen::MatrixXd &);

  bool check_stop_criterion();

  void compute_w();
  void compute_u();
  
  void fft_2dim(Eigen::MatrixXcd &dst_mat,const Eigen::MatrixXcd &src_mat,bool forward=true);

  public:
  ImageReconstructor(){}
  void set_epsilon(double e){ epsilon_ = e; }
  void operator()(const cv::Mat &src_img,cv::Mat &dst_img);
};

#endif
