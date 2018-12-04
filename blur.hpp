#ifndef METRIC_BLUR_H_
#define METRIC_BLUR_H_

#include "opencv2/opencv.hpp"

namespace metric {

template <typename Derived>
class BlurMetric {
 public:
  double Estimate(const cv::Mat& img) {
    return static_cast<Derived*>(this)->Estimate(img);
  }

  double Estimate(const std::string& filename) {
    cv::Mat img = cv::imread(filename, 1); // here 1 indicate BGR channel
    return Estimate(img);
  }
};

}  // end of namespace metric


#endif  // end of METRIC_BLUR_H_
