#ifndef DENOISER_H
#define DENOISER_H
#include <opencv2/core/core.hpp>

class denoiser{

  public:

  void operator()(const cv::Mat &src_img,cv::Mat &dst_img);

};

#endif
