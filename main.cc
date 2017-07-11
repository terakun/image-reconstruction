#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "./denoiser.h"

int main(int argc, char *argv[]){

  if(argc < 2){
    std::cerr << argv[0] << " [image file]" << std::endl;
    return 0;
  }

  cv::Mat src_img = cv::imread(argv[1]);
  if(src_img.empty()) return -1;

  // processing
  denoiser dn;
  cv::Mat dst_img;
  dn(src_img,dst_img);

  cv::namedWindow("source image", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
  cv::imshow("source image", src_img);
  cv::namedWindow("denoised image", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
  cv::imshow("denoised image", dst_img);
   
  cv::waitKey(0);
  return 0;
}
