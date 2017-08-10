#ifndef RECONSTRUCTOR_H
#define RECONSTRUCTOR_H
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>

namespace imagereconstruction{
  template<typename First> 
  First max(First first){
    return first;
  }

  template<typename First,typename... Rest> 
  First max(First first,Rest... rest){
    return std::max(first,max(rest...));
  }

  class ImageReconstructor{
    static constexpr int gaussian_size = 3;
    static const double gaussian[gaussian_size][gaussian_size];

    static constexpr int diff_size = 3;
    static const double diff_horizontal[diff_size][diff_size];
    static const double diff_vertical[diff_size][diff_size];

    int img_rows_,img_cols_;
    double beta_,mu_;
    double epsilon_;

    Eigen::MatrixXd w_horizontal_,w_vertical_;
    Eigen::MatrixXd u_;
    Eigen::MatrixXd observed_img_;

    Eigen::MatrixXd D_horizontal_,D_vertical_;
    Eigen::MatrixXd K_;

    Eigen::MatrixXcd D_horizontal_fft_;  // difference operator
    Eigen::MatrixXcd D_vertical_fft_;    // difference operator
    Eigen::MatrixXcd K_fft_;             // blurring operator
    Eigen::MatrixXcd observed_img_fft_;

    double compute_horizontal_diff(const Eigen::MatrixXd &,int r,int c);
    double compute_vertical_diff(const Eigen::MatrixXd &,int r,int c);
    Eigen::Vector2d compute_grad(const Eigen::MatrixXd &,int r,int c);

    double blur(const Eigen::MatrixXd &src_mat,int r,int c);
    void blur(Eigen::MatrixXd &dst_mat,const Eigen::MatrixXd &src_mat);

    bool check_stop_criterion();

    void compute_w();
    void compute_u();

    void fft_2dim(Eigen::MatrixXcd &dst_mat,const Eigen::MatrixXcd &src_mat,bool forward=true);

    int get_r(int r,int rows){
      if(r < 0) r += rows;
      return r%rows;
    }

    int get_c(int c,int cols){
      if(c < 0) c += cols;
      return c%cols;
    }

    public:
    ImageReconstructor(){}
    void set_epsilon(double e){ epsilon_ = e; }
    void operator()(const cv::Mat &src_img,cv::Mat &dst_img);
  };
};

#endif
