# A NO-REFERENCE PERCEPTUAL BLUR METRIC

## Introduction
the file of local_blur_based_edge.cc implement the algoritm which described by https://ieeexplore.ieee.org/abstract/document/1038902, and we can use this algorithm to do blur detection.

### Dependencies
* OpenCV
* Cmake

### Installation
1. git clone https://github.com/PeterWang1986/blur.git
2. mkdir build && cd build && cmake ..
3. make

### Test
run build/local_blur_based_edge_test, you will see:
* testdata/zy1.jpg blur= 19.694
* testdata/zy2.jpg blur= 7.21562
