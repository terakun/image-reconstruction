#include <iostream>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include "./reconstructor.h"

int main(int argc, char *argv[]){
  if(argc < 2){
    std::cerr << argv[0] << " [image file]" << std::endl;
    return 0;
  }

  cv::Mat src_img = cv::imread(argv[1],0);
  if(src_img.empty()) return -1;

  imagereconstruction::ImageReconstructor rc;
  rc.set_epsilon(1.0e-3);
  cv::Mat dst_img;
  rc(src_img,dst_img);

  cv::namedWindow("source image", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
  cv::imshow("source image", src_img);
  cv::namedWindow("denoised image", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
  cv::imshow("denoised image", dst_img);
   
  cv::waitKey(0);
  return 0;
}
