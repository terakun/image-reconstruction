#include <opencv2/imgproc.hpp>
#include <unsupported/Eigen/FFT>
#include "./reconstructor.h"

void ImageReconstructor::operator()(const cv::Mat &src_img,cv::Mat &dst_img){
  img_rows_ = src_img.rows;
  img_cols_ = src_img.cols;

  while(check_stop_criterion()){
    compute_w();
    compute_u();
  }
}

double ImageReconstructor::compute_horizontal_diff(const Eigen::MatrixXd &mat,int r,int c){
  return mat(r,(c+1)%mat.cols())-mat(r,c);
}

double ImageReconstructor::compute_vertical_diff(const Eigen::MatrixXd &mat,int r,int c){
  return mat((r+1)%mat.rows(),c)-mat(r,c);
}

bool ImageReconstructor::check_stop_criterion(){
  return true;
}

void ImageReconstructor::compute_w(){
  for(int r=0;r<img_rows_;++r){
    for(int c=0;c<img_cols_;++c){
      double Du_x = compute_horizontal_diff(u_,r,c);
      double Du_y = compute_vertical_diff(u_,r,c);

      Eigen::Vector2d Du,w_vec;
      Du << Du_x , Du_y;

      double Du_norm = Du.norm();
      double coef = Du_norm-1.0/beta_;
      if(coef > 0){
        w_horizontal_(r,c) = coef*Du_x/Du_norm;
        w_vertical_(r,c) = coef*Du_y/Du_norm;
      }else{
        w_horizontal_(r,c) = 0;
        w_vertical_(r,c) = 0;
      }
    }
  }

}

void ImageReconstructor::compute_u(){
}

void ImageReconstructor::fft2dim(Eigen::MatrixXcd &dst_mat,const Eigen::MatrixXcd &src_mat,bool forward){
  Eigen::FFT<double> fft;
  int rows = src_mat.rows();
  int cols = src_mat.cols();

  Eigen::MatrixXcd tmp_mat(rows,cols);
  for(int c=0;c<cols;++c){
    Eigen::VectorXcd src_vec = src_mat.col(c);
    Eigen::VectorXcd tmp_vec(cols);
    if(forward){
      fft.fwd(tmp_vec,src_vec);
    }else{
      fft.inv(tmp_vec,src_vec);
    }
    tmp_mat.col(c) = tmp_vec;
  }

  tmp_mat.transposeInPlace();
  for(int c=0;c<cols;++c){
    Eigen::VectorXcd tmp_vec = tmp_mat.col(c);
    Eigen::VectorXcd dst_vec(cols);
    if(forward){
      fft.fwd(dst_vec,tmp_vec);
    }else{
      fft.inv(dst_vec,tmp_vec);
    }
    dst_mat.col(c) = dst_vec;
  }
}


