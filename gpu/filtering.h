#ifndef __morphing__h
#define __morphing__h

#include <cmath>
#include <iostream>

#include "basicImageManipulation.h"
#include "Image.h"

using namespace std;

class Filter {

public:
    //Constructor
    Filter(const vector<float> &fData, int fWidth, int fHeight);
    Filter(int fWidth, int fHeight); // kernel with all zero

    // Destructor. Because there is no explicit memory management here, this doesn't do anything
    ~Filter();

    // function to convolve your filter with an image
    Image convolve(const Image &im, bool clamp=true);

    // Accessors of the filter values
    const float & operator()(int x, int y) const;
    float & operator()(int x, int y);

private:
    std::vector<float> kernel;
    int width;
    int height;
};

// GPU bilateral filtering
Image bilateralGpuBasic(const Image & im, float sigmaRange = 0.1, float sigmaDomain = 1.0, float truncateDomain = 3.0);

#endif
