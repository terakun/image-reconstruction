#include <opencv2/imgproc.hpp>
#include <eigen3/Eigen/Dense>
#include "./reconstructor.h"

void ImageReconstructor::operator()(const cv::Mat &src_img,cv::Mat &dst_img){


  while(check_stop_criterion()){
    compute_w();
    compute_u();
  }

}

bool ImageReconstructor::check_stop_criterion(){
  return true;
}

void ImageReconstructor::compute_w(){
}

void ImageReconstructor::compute_u(){
}
