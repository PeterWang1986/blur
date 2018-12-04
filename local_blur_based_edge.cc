
#include "local_blur_based_edge.h"

namespace metric {
using cv::Mat;

double LocalBlurBasedEdge::Estimate(const Mat& img) {
  Mat gray_img;
  if (img.channels() == 3) {
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
  } else {
    gray_img = img;
  }

  Mat sobel_edge_img;
  cv::Sobel(gray_img, sobel_edge_img, -1, 1, 0);  //sobel vertical edge detection

  int64_t blur = 0;
  int64_t edge_pixel_count = 0;
  const int rows = gray_img.rows;
  const int cols = gray_img.cols;
  Mat local_stationary(5, cols, CV_32SC1);
  for (int i = 0; i < rows; ++i) {
    const unsigned char* data = gray_img.ptr(i);
    CalcStationaryPoint(data, local_stationary);
    for (int j = 0; j < cols; ++j) {
      const int val = static_cast<int>(sobel_edge_img.at<unsigned char>(i, j));
      if (val > 0) { // represent it is a vertical edge
        const int left_min = local_stationary.at<int32_t>(1, j);
        const int left_max = local_stationary.at<int32_t>(2, j);
        const int right_min = local_stationary.at<int32_t>(3, j);
        const int right_max = local_stationary.at<int32_t>(4, j);
        int min_max_blur = cols;
        int max_min_blur = cols;
        if (left_min > 0 && right_max > 0) {
          min_max_blur = right_max - left_min + 1;
        }
        if (left_max > 0 && right_min > 0) {
          max_min_blur = right_min - left_max + 1;
        }
        const int local_blur = std::min(min_max_blur, max_min_blur);
        if (local_blur < cols) {
          ++edge_pixel_count;
          blur += local_blur;
        }
      }
    }
  }
  return blur / static_cast<double>(edge_pixel_count);
}

// here local_stationary has 5 * cols layout
// first row represent pixel state(0: normal, -1: minimum, 1: maximum)
// second row represent the position of left minimum of current pixel
// third row represent the position of left maximum of current pixel
// fouth row represent the position of right minimum of current pixel
// fifth row represent the position of right maximum of current pixel
void LocalBlurBasedEdge::CalcStationaryPoint(
  const unsigned char* data,
  Mat& local_stationary) {
  const int cols = local_stationary.cols;

  //for left side
  local_stationary.at<int32_t>(0, 0) = kNormalState;
  local_stationary.at<int32_t>(1, 0) = -1;
  local_stationary.at<int32_t>(2, 0) = -1;
  const int last_index = cols - 1;
  for (int i = 1; i < last_index; ++i) {
    const int previous_index = i - 1;
    const int left = static_cast<int>(data[previous_index]);
    const int middle = static_cast<int>(data[i]);
    const int right = static_cast<int>(data[i + 1]);
    if (middle < left && middle <= right) {
      local_stationary.at<int32_t>(0, i) = kMinState; // local minimum
      const int32_t state = local_stationary.at<int32_t>(0, previous_index);
      if (state == kMaxState) {
        local_stationary.at<int32_t>(1, i) = local_stationary.at<int32_t>(1, previous_index);
        local_stationary.at<int32_t>(2, i) = previous_index;
      } else {
        local_stationary.at<int32_t>(1, i) = local_stationary.at<int32_t>(1, previous_index);
        local_stationary.at<int32_t>(2, i) = local_stationary.at<int32_t>(2, previous_index);
      }
    } else if (middle >= left && middle > right) {
      local_stationary.at<int32_t>(0, i) = kMaxState; // local maximum
      const int32_t state = local_stationary.at<int32_t>(0, previous_index);
      if (state == kMinState) {
        local_stationary.at<int32_t>(1, i) = previous_index;
        local_stationary.at<int32_t>(2, i) = local_stationary.at<int32_t>(2, previous_index);
      } else {
        local_stationary.at<int32_t>(1, i) = local_stationary.at<int32_t>(1, previous_index);
        local_stationary.at<int32_t>(2, i) = local_stationary.at<int32_t>(2, previous_index);
      }
    } else {
      local_stationary.at<int32_t>(0, i) = kNormalState; // normal
      const int32_t state = local_stationary.at<int32_t>(0, previous_index);
      if (state == kMinState) {
        local_stationary.at<int32_t>(1, i) = previous_index;
        local_stationary.at<int32_t>(2, i) = local_stationary.at<int32_t>(2, previous_index);
      } else if (state == kMaxState) {
        local_stationary.at<int32_t>(1, i) = local_stationary.at<int32_t>(1, previous_index);
        local_stationary.at<int32_t>(2, i) = previous_index;
      } else {
        local_stationary.at<int32_t>(1, i) = local_stationary.at<int32_t>(1, previous_index);
        local_stationary.at<int32_t>(2, i) = local_stationary.at<int32_t>(2, previous_index);
      }
    }
  }
  local_stationary.at<int32_t>(0, last_index) = kNormalState; // normal
  const int32_t state = local_stationary.at<int32_t>(0, last_index - 1);
  if (state == kMinState) {
    local_stationary.at<int32_t>(1, last_index) = last_index - 1;
    local_stationary.at<int32_t>(2, last_index) = local_stationary.at<int32_t>(2, last_index - 1);
  } else if (state == kMaxState) {
    local_stationary.at<int32_t>(1, last_index) = local_stationary.at<int32_t>(1, last_index - 1);
    local_stationary.at<int32_t>(2, last_index) = last_index - 1;
  } else {
    local_stationary.at<int32_t>(1, last_index) = local_stationary.at<int32_t>(1, last_index - 1);
    local_stationary.at<int32_t>(2, last_index) = local_stationary.at<int32_t>(2, last_index - 1);
  }

  // for right side
  local_stationary.at<int32_t>(3, last_index) = -1;
  local_stationary.at<int32_t>(4, last_index) = -1;
  for (int i = last_index - 1; i >= 0; --i) {
    const int32_t state = local_stationary.at<int32_t>(0, i + 1);
    if (state == kMinState) {
      local_stationary.at<int32_t>(3, i) = i + 1;
      local_stationary.at<int32_t>(4, i) = local_stationary.at<int32_t>(4, i + 1);
    } else if (state == kMaxState) {
      local_stationary.at<int32_t>(3, i) = local_stationary.at<int32_t>(3, i + 1);
      local_stationary.at<int32_t>(4, i) = i + 1;
    } else {
      local_stationary.at<int32_t>(3, i) = local_stationary.at<int32_t>(3, i + 1);
      local_stationary.at<int32_t>(4, i) = local_stationary.at<int32_t>(4, i + 1);
    }
  }
}

}  // end of namespace metric
