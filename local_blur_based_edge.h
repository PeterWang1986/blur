#ifndef METRIC_LOCAL_BLUR_BASED_EDGE_H_
#define METRIC_LOCAL_BLUR_BASED_EDGE_H_
// ref to paper: "A NO-REFERENCE PERCEPTUAL BLUR METRIC"

#include "blur.hpp"

namespace metric {

class LocalBlurBasedEdge : public BlurMetric<LocalBlurBasedEdge> {
 public:
  // if the img.channels()==3, we assume it is BGR mode
  // and the storage of img MUST be continuous
  double Estimate(const cv::Mat& img);

 private:
  void CalcStationaryPoint(const unsigned char* data,
                           cv::Mat& local_stationary);

 private:
  static const int kMinState = -1; 
  static const int kMaxState = 1; 
  static const int kNormalState = 0; 
};

}  // end of namespace metric
#endif  // end of METRIC_LOCAL_BLUR_BASED_EDGE_H_
