#include <iostream>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include "./reconstructor.h"

int main(int argc, char *argv[]){
  if(argc < 8){
    std::cerr << argv[0] << " [image file] [hsize] [sigma] [beta0] [max beta] [mu] [max count]" << std::endl;
    return 0;
  }

  cv::Mat src_img = cv::imread(argv[1],0);
  if(src_img.empty()) return -1;

  imagereconstruction::ImageReconstructor rc;
  rc.set_epsilon(2.0e-3);
  rc.set_max_count(std::atoi(argv[7]));
  rc.set_gaussian(std::atoi(argv[2]),std::atof(argv[3]));
  rc.set_mu(std::atof(argv[6]));
  rc.set_beta0(std::atof(argv[4]));
  rc.set_max_beta(std::atof(argv[5]));
  cv::Mat dst_img;
  rc(src_img,dst_img);

  cv::namedWindow("noised image", CV_WINDOW_AUTOSIZE);
  cv::imshow("noised image", src_img);
  cv::imwrite("dst.png",dst_img);
   
  cv::waitKey(0);
  return 0;
}
