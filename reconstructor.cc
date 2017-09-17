#include <Eigen/Dense>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <unsupported/Eigen/FFT>
#include <complex>
#include <iostream>
#include "./reconstructor.h"

using namespace imagereconstruction;

void ImageReconstructor::operator()(const cv::Mat &src_img,cv::Mat &dst_img){
  img_rows_ = src_img.rows;
  img_cols_ = src_img.cols;

  std::cout << img_rows_ << "," << img_cols_ << std::endl;

  w_horizontal_ = Eigen::MatrixXd::Zero(img_rows_,img_cols_);
  w_vertical_ = Eigen::MatrixXd::Zero(img_rows_,img_cols_);

  K_ = Eigen::MatrixXd::Zero(img_rows_,img_cols_);
  D_horizontal_ = Eigen::MatrixXd::Zero(img_rows_,img_cols_);
  D_vertical_ = Eigen::MatrixXd::Zero(img_rows_,img_cols_);

  K_fft_ = Eigen::MatrixXd::Zero(img_rows_,img_cols_);
  D_horizontal_fft_ = Eigen::MatrixXd::Zero(img_rows_,img_cols_);
  D_vertical_fft_ = Eigen::MatrixXd::Zero(img_rows_,img_cols_);
  denom_fft_ = Eigen::MatrixXd::Zero(img_rows_,img_cols_);

  observed_img_fft_ = Eigen::MatrixXd::Zero(img_rows_,img_cols_);

  for(int i=0;i<gaussian_size_;++i){
    for(int j=0;j<gaussian_size_;++j){
      int r = get_r(i-gaussian_size_/2,img_rows_);
      int c = get_c(j-gaussian_size_/2,img_cols_);
      K_(r,c) = gaussian_filter_(i,j);
    }
  }

  D_horizontal_(0,0) = -1;
  D_horizontal_(0,1) = 1;

  D_vertical_(0,0) = -1;
  D_vertical_(1,0) = 1;
 
  fft_2dim(K_fft_,K_);
  fft_2dim(D_horizontal_fft_,D_horizontal_);
  fft_2dim(D_vertical_fft_,D_vertical_);

  cv::cv2eigen(src_img,observed_img_);
  observed_img_ /= 255.0;
  fft_2dim(observed_img_fft_,observed_img_);
  
  u_ = observed_img_;
  beta_ = beta0_;
  cv::namedWindow("differential horizontal image", CV_WINDOW_AUTOSIZE);
  cv::namedWindow("differential vertical image", CV_WINDOW_AUTOSIZE);
  cv::namedWindow("denoised image", CV_WINDOW_AUTOSIZE);
  while(beta_<=max_beta_){

    denom_fft_ = (beta_/mu_)*(D_horizontal_fft_.cwiseProduct(D_horizontal_fft_.conjugate())
                +D_vertical_fft_.cwiseProduct(D_vertical_fft_.conjugate()))
                +K_fft_.cwiseProduct(K_fft_.conjugate());

    for(cnt_=0;cnt_<max_cnt_&&!check_stop_criterion();++cnt_){
      compute_w();
      compute_u();

      Eigen::MatrixXd u_tmp(img_rows_,img_cols_);

      cv::Mat tmp_img;
      cv::eigen2cv(u_,tmp_img);
      tmp_img.convertTo(dst_img,CV_8UC1,255);

      cv::Mat diff_horizontal_img;
      cv::eigen2cv(w_horizontal_,diff_horizontal_img);

      cv::Mat diff_vertical_img;
      cv::eigen2cv(w_vertical_,diff_vertical_img);

      cv::imshow("differential horizontal image", diff_horizontal_img);
      cv::imshow("differential vertical image", diff_vertical_img);
      cv::imshow("denoised image", dst_img);


      int k = cv::waitKey(10);
      if(k==13) break;
    }
    beta_ *= 2.0;
  }

}

void ImageReconstructor::set_gaussian(int size,double sigma){
  gaussian_size_ = size;
  gaussian_filter_ = Eigen::MatrixXd::Zero(gaussian_size_,gaussian_size_);

  double sum = 0;
  for(int i=0;i<gaussian_size_;++i){
    for(int j=0;j<gaussian_size_;++j){
      int x = j-gaussian_size_/2;
      int y = i-gaussian_size_/2;
      gaussian_filter_(i,j) = std::exp(-(x*x+y*y)/(2.0*sigma*sigma));
      sum += gaussian_filter_(i,j);
    }
  }
  gaussian_filter_ /= sum;
}

double ImageReconstructor::compute_forward_horizontal_diff(const Eigen::MatrixXd &mat,int r,int c)const{
  int fc = get_c(c+1,img_cols_);
  int bc = get_c(c,img_cols_);
  return mat(r,fc)-mat(r,bc);
}

void ImageReconstructor::compute_forward_horizontal_diff(Eigen::MatrixXd &dst_mat,const Eigen::MatrixXd &src_mat)const{
  int rows = src_mat.rows();
  int cols = src_mat.cols();

  for(int r=0;r<rows;++r){
    for(int c=0;c<cols;++c){
      dst_mat(r,c) = compute_forward_horizontal_diff(src_mat,r,c);
    }
  }
}


double ImageReconstructor::compute_forward_vertical_diff(const Eigen::MatrixXd &mat,int r,int c)const{
  int fr = get_r(r+1,img_rows_);
  int br = get_r(r,img_rows_);
  return mat(fr,c)-mat(br,c);
}

void ImageReconstructor::compute_forward_vertical_diff(Eigen::MatrixXd &dst_mat,const Eigen::MatrixXd &src_mat)const{
  int rows = src_mat.rows();
  int cols = src_mat.cols();

  for(int r=0;r<rows;++r){
    for(int c=0;c<cols;++c){
      dst_mat(r,c) = compute_forward_vertical_diff(src_mat,r,c);
    }
  }
}

Eigen::Vector2d ImageReconstructor::compute_grad(const Eigen::MatrixXd &mat,int r,int c)const{
  Eigen::Vector2d grad;
  grad << compute_forward_horizontal_diff(mat,r,c)
         ,compute_forward_vertical_diff(mat,r,c);
  return grad;
}

double ImageReconstructor::compute_backward_horizontal_diff(const Eigen::MatrixXd &mat,int r,int c)const{
  int fc = get_c(c,img_cols_);
  int bc = get_c(c-1,img_cols_);
  return mat(r,fc)-mat(r,bc);
}

void ImageReconstructor::compute_backward_horizontal_diff(Eigen::MatrixXd &dst_mat,const Eigen::MatrixXd &src_mat)const{
  int rows = src_mat.rows();
  int cols = src_mat.cols();

  for(int r=0;r<rows;++r){
    for(int c=0;c<cols;++c){
      dst_mat(r,c) = compute_backward_horizontal_diff(src_mat,r,c);
    }
  }
}

double ImageReconstructor::compute_backward_vertical_diff(const Eigen::MatrixXd &mat,int r,int c)const{
  int fr = get_r(r,img_rows_);
  int br = get_r(r-1,img_rows_);
  return mat(fr,c)-mat(br,c);
}

void ImageReconstructor::compute_backward_vertical_diff(Eigen::MatrixXd &dst_mat,const Eigen::MatrixXd &src_mat)const{
  int rows = src_mat.rows();
  int cols = src_mat.cols();

  for(int r=0;r<rows;++r){
    for(int c=0;c<cols;++c){
      dst_mat(r,c) = compute_backward_vertical_diff(src_mat,r,c);
    }
  }

}

double ImageReconstructor::blur(const Eigen::MatrixXd &src_mat,int cr,int cc)const{
  double blurred_val = 0;
  for(int i=0;i<gaussian_size_;++i){
    for(int j=0;j<gaussian_size_;++j){
      int r = get_r(cr+i-gaussian_size_/2,img_rows_);
      int c = get_c(cc+j-gaussian_size_/2,img_cols_);

      blurred_val += src_mat(r,c)*gaussian_filter_(i,j);
    }
  }
  return blurred_val;
}


void ImageReconstructor::blur(Eigen::MatrixXd &dst_mat,const Eigen::MatrixXd &src_mat)const{
  int rows = src_mat.rows();
  int cols = src_mat.cols();

  for(int r=0;r<rows;++r){
    for(int c=0;c<cols;++c){
      dst_mat(r,c) = blur(src_mat,r,c);
    }
  }
}

bool ImageReconstructor::check_stop_criterion()const{
  double max_r1_norm = -1.0e10 , max_r2 = -1.0e10;

  Eigen::MatrixXd diff_horizontal_u(img_rows_,img_cols_);
  Eigen::MatrixXd sub_horizontal(img_rows_,img_cols_);

  Eigen::MatrixXd diff_vertical_u(img_rows_,img_cols_);
  Eigen::MatrixXd sub_vertical(img_rows_,img_cols_);

  for(int r=0;r<img_rows_;++r){
    for(int c=0;c<img_cols_;++c){
      Eigen::Vector2d Du = compute_grad(u_,r,c);
      diff_horizontal_u(r,c) = Du[0];
      sub_horizontal(r,c) = diff_horizontal_u(r,c)-w_horizontal_(r,c);

      diff_vertical_u(r,c) = Du[1];
      sub_vertical(r,c) = diff_vertical_u(r,c)-w_vertical_(r,c);

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

  Eigen::MatrixXd diff_horizontal_sub_horizontal(img_rows_,img_cols_);
  compute_backward_horizontal_diff(diff_horizontal_sub_horizontal,sub_horizontal);
  
  Eigen::MatrixXd diff_vertical_sub_vertical(img_rows_,img_cols_);
  compute_backward_vertical_diff(diff_vertical_sub_vertical,sub_vertical);
  
  Eigen::MatrixXd blurred_u(img_rows_,img_cols_);
  blur(blurred_u,u_);
  Eigen::MatrixXd sub_blurred = blurred_u - observed_img_;
  Eigen::MatrixXd blurred_sub_blurred(img_rows_,img_cols_);
  blur(blurred_sub_blurred,sub_blurred);
  
  Eigen::MatrixXd r3 = beta_/mu_*(diff_horizontal_sub_horizontal+diff_vertical_sub_vertical)
                      +blurred_sub_blurred;
  
  double max_r3_infinity_norm = r3.lpNorm<Eigen::Infinity>();
  
  double max_r = max(max_r1_norm,max_r2,max_r3_infinity_norm);
  std::cout << beta_ << " " << max_r1_norm << " " << max_r2 << " " << max_r3_infinity_norm << std::endl;
  return max_r < epsilon_;
}

void ImageReconstructor::compute_w(){
  for(int r=0;r<img_rows_;++r){
    for(int c=0;c<img_cols_;++c){
      Eigen::Vector2d Du(compute_grad(u_,r,c));

      double Du_norm = Du.norm();
      if(Du_norm==0) Du_norm = 1.0;
      double coef = Du_norm-1.0/beta_;
      if(coef > 0){
        w_horizontal_(r,c) = coef*Du[0]/Du_norm;
        w_vertical_(r,c) = coef*Du[1]/Du_norm;
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

  Eigen::MatrixXcd numer_fft = (beta_/mu_)*(w_horizontal_fft.cwiseProduct(D_horizontal_fft_.conjugate())
                  +w_vertical_fft.cwiseProduct(D_vertical_fft_.conjugate()))
                  +observed_img_fft_.cwiseProduct(K_fft_.conjugate());

  u_fft = numer_fft.cwiseQuotient(denom_fft_);
  Eigen::MatrixXcd u_complex(img_rows_,img_cols_);

  fft_2dim(u_complex,u_fft,false);
  u_ = u_complex.real();
}

void ImageReconstructor::fft_2dim(Eigen::MatrixXcd &dst_mat,const Eigen::MatrixXcd &src_mat,bool forward)const{
  Eigen::FFT<double> fft;
  int rows = src_mat.rows();
  int cols = src_mat.cols();

  Eigen::MatrixXcd work_mat(rows,cols);
  work_mat = src_mat;
  
  Eigen::MatrixXcd tmp_mat(rows,cols);
  for(int c=0;c<cols;++c){
    Eigen::VectorXcd src_vec = work_mat.col(c);
    Eigen::VectorXcd tmp_vec(rows);
    if(forward){
      fft.fwd(tmp_vec,src_vec);
    }else{
      fft.inv(tmp_vec,src_vec);
    }
    tmp_mat.col(c) = tmp_vec;
  }

  for(int r=0;r<rows;++r){
    Eigen::VectorXcd tmp_vec = tmp_mat.row(r);
    Eigen::VectorXcd dst_vec(cols);
    if(forward){
      fft.fwd(dst_vec,tmp_vec);
    }else{
      fft.inv(dst_vec,tmp_vec);
    }
    dst_mat.row(r) = dst_vec;
  }
}

