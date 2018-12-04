
#include <string>
#include <iostream>

#include "local_blur_based_edge.h"

using namespace std;

int main() {
  string filenames[2] = {"testdata/zy1.jpg", "testdata/zy2.jpg"};
  metric::BlurMetric<metric::LocalBlurBasedEdge> estimator;
  for (int i = 0; i < 2; ++i) {
    const string file = filenames[i];
    const double blur = estimator.Estimate(file);
    cout << file << " blur= " << blur << endl;
  }
  return 0;
}

