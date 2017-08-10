#include <opencv2/imgproc.hpp>
#include <unsupported/Eigen/FFT>
#include <complex>
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

Eigen::Vector2d ImageReconstructor::compute_grad(const Eigen::MatrixXd &mat,int r,int c){
  Eigen::Vector2d grad;
  grad << compute_horizontal_diff(mat,r,c) ,
          compute_vertical_diff(mat,r,c);
  return grad;
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

      std::complex<double> denom;

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


