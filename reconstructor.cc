#include <Eigen/Dense>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <unsupported/Eigen/FFT>
#include <complex>
#include <iostream>
#include "./reconstructor.h"

using namespace imagereconstruction;

const double ImageReconstructor::gaussian[gaussian_size][gaussian_size] = {
  {0.0625,0.125,0.0625},
  {0.125 ,0.25 ,0.125 },
  {0.0625,0.125,0.0625}
};

const double ImageReconstructor::diff_horizontal[diff_size][diff_size] = {
  {0,0,0},
  {0,-1,1},
  {0,0,0}
};

const double ImageReconstructor::diff_vertical[diff_size][diff_size] = {
  {0,1,0},
  {0,-1,0},
  {0,0,0}
};

void ImageReconstructor::operator()(const cv::Mat &src_img,cv::Mat &dst_img){
  img_rows_ = src_img.rows;
  img_cols_ = src_img.cols;
  dst_img = src_img.clone();

  K_ = Eigen::MatrixXd::Zero(img_rows_,img_cols_);
  D_horizontal_ = Eigen::MatrixXd::Zero(img_rows_,img_cols_);
  D_vertical_ = Eigen::MatrixXd::Zero(img_rows_,img_cols_);

  u_ = Eigen::MatrixXd::Zero(img_rows_,img_cols_);

  K_fft_ = Eigen::MatrixXd::Zero(img_rows_,img_cols_);
  D_horizontal_fft_ = Eigen::MatrixXd::Zero(img_rows_,img_cols_);
  D_vertical_fft_ = Eigen::MatrixXd::Zero(img_rows_,img_cols_);

  observed_img_fft_ = Eigen::MatrixXd::Zero(img_rows_,img_cols_);

  for(int i=0;i<gaussian_size;++i){
    for(int j=0;j<gaussian_size;++j){
      int r = get_r(i-gaussian_size/2,img_rows_);
      int c = get_c(j-gaussian_size/2,img_cols_);
      K_(r,c) = gaussian[i][j];
    }
  }

  for(int i=0;i<diff_size;++i){
    for(int j=0;j<diff_size;++j){
      int r = get_r(i-diff_size/2,img_rows_);
      int c = get_c(j-diff_size/2,img_cols_);
      D_horizontal_(r,c) = diff_horizontal[i][j];
      D_vertical_(r,c) = diff_vertical[i][j];
    }
  }
 
  fft_2dim(K_fft_,K_);
  fft_2dim(D_horizontal_fft_,D_horizontal_);
  fft_2dim(D_vertical_fft_,D_vertical_);

  cv::cv2eigen(src_img,observed_img_);
  fft_2dim(observed_img_fft_,observed_img_);

  // while(check_stop_criterion()){
  //   compute_w();
  //   compute_u();
  // }
  
  cv::Mat work_img;
  cv::eigen2cv(u_,work_img);
  work_img.convertTo(dst_img,CV_8UC1);
}

double ImageReconstructor::compute_horizontal_diff(const Eigen::MatrixXd &mat,int r,int c){
  return mat(r,(c+1)%mat.cols())-mat(r,c);
}

double ImageReconstructor::compute_vertical_diff(const Eigen::MatrixXd &mat,int r,int c){
  return mat((r+1)%mat.rows(),c)-mat(r,c);
}

Eigen::Vector2d ImageReconstructor::compute_grad(const Eigen::MatrixXd &mat,int r,int c){
  Eigen::Vector2d grad;
  grad << compute_horizontal_diff(mat,r,c) ,
          compute_vertical_diff(mat,r,c);
  return grad;
}


double ImageReconstructor::blur(const Eigen::MatrixXd &src_mat,int cr,int cc){
  double blurred_val = 0;
  for(int i=0;i<gaussian_size;++i){
    for(int j=0;j<gaussian_size;++j){
      int r = get_r(cr+i-gaussian_size/2,img_rows_);
      int c = get_c(cc+j-gaussian_size/2,img_cols_);

      blurred_val += src_mat(r,c)*gaussian[i][j];
    }
  }
  return blurred_val;
}


void ImageReconstructor::blur(Eigen::MatrixXd &dst_mat,const Eigen::MatrixXd &src_mat){
  int rows = src_mat.rows();
  int cols = src_mat.cols();

  for(int r=0;r<rows;++r){
    for(int c=0;c<cols;++c){
      dst_mat(r,c) = blur(src_mat,r,c);
    }
  }
}

bool ImageReconstructor::check_stop_criterion(){
  double max_r1_norm = -1.0e10 , max_r2 = -1.0e10;

  for(int r=0;r<img_rows_;++r){
    for(int c=0;c<img_cols_;++c){
      Eigen::Vector2d Du = compute_grad(u_,r,c);
      double Du_norm = Du.norm();

      if(w_horizontal_(r,c)==0&&w_vertical_(r,c)==0){
        double r2 = Du_norm - 1.0/beta_;
        max_r2 = std::max(r2,max_r2);
      }else{
        Eigen::Vector2d w_vec;
        w_vec << w_horizontal_(r,c) , w_vertical_(r,c);
        Eigen::Vector2d r1 = w_vec/(w_vec.norm()*beta_) + w_vec - Du;
        max_r1_norm = std::max(r1.norm(),max_r1_norm);
      }
    }
  }

  Eigen::MatrixXd r3(img_rows_,img_cols_);

  double r3_norm = r3.lpNorm<Eigen::Infinity>(); 

  return max(max_r1_norm,max_r2,r3_norm) < epsilon_;
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
  Eigen::MatrixXcd w_horizontal_fft(img_rows_,img_cols_);
  Eigen::MatrixXcd w_vertical_fft(img_rows_,img_cols_);

  fft_2dim(w_horizontal_fft,w_horizontal_);
  fft_2dim(w_vertical_fft,w_vertical_);

  Eigen::MatrixXcd u_fft(img_rows_,img_cols_);
  for(int r=0;r<img_rows_;++r){
    for(int c=0;c<img_cols_;++c){
      std::complex<double> numer = 
        std::conj(D_horizontal_fft_(r,c))*w_horizontal_fft(r,c)+
        std::conj(D_vertical_fft_(r,c))*w_vertical_fft(r,c)+
        (mu_/beta_)*std::conj(K_fft_(r,c))*observed_img_fft_(r,c);

      std::complex<double> denom = 
        std::conj(D_horizontal_fft_(r,c))*D_horizontal_fft_(r,c)+
        std::conj(D_vertical_fft_(r,c))*D_vertical_fft_(r,c)+
        (mu_/beta_)*std::conj(K_fft_(r,c))*K_fft_(r,c);

      u_fft(r,c) = numer/denom;
    }
  }

}

void ImageReconstructor::fft_2dim(Eigen::MatrixXcd &dst_mat,const Eigen::MatrixXcd &src_mat,bool forward){
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

