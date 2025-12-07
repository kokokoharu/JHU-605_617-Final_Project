#include "filtering.h"
#include <cmath>
#include <cassert>

using namespace std;

// ------------- FILTER CLASS -----------------------
Filter::Filter(const vector<float> &fData, int fWidth, int fHeight)
  : kernel(fData), width(fWidth), height(fHeight)
{
    assert(fWidth*fHeight == (int) fData.size());
}


Filter::Filter(int fWidth, int fHeight)
  : kernel(std::vector<float>(fWidth*fHeight,0)), width(fWidth), height(fHeight)
{}

Image Filter::convolve(const Image &im, bool clamp)
{
    Image imFilter(im.width(), im.height(), im.channels());

    int sideW = int((width-1.0)/2.0);
    int sideH = int((height-1.0)/2.0);
    float accum;

    // For every pixel in the image
    for (int z = 0; z < imFilter.channels(); z++) {
        for (int y = 0; y < imFilter.height(); y++) {
            for (int x = 0; x < imFilter.width(); x++) {
                accum = 0.0;
                for (int yFilter=0; yFilter<height; yFilter++) {
                    for (int xFilter=0; xFilter<width; xFilter++) {
                        // Sum the image pixel values weighted by the filter
                        // flipped kernel, xFilter, yFilter have different signs in filter
                        // and im
                        accum += operator()(xFilter, yFilter) *
                            im.smartAccessor(x-xFilter+sideW,y-yFilter+sideH,z,clamp);
                    }
                }
                // Assign the pixel the value from convolution
                imFilter(x,y,z) = accum;
           }
        }
    }
    return imFilter;
}

Filter::~Filter()
{}


const float & Filter::operator()(int x, int y) const
{
    if (x < 0 || x >= width)
        throw OutOfBoundsException();
    if ( y < 0 || y >= height)
        throw OutOfBoundsException();

    return kernel[x + y*width];
}


float & Filter::operator()(int x, int y)
{
    if (x < 0 || x >= width)
        throw OutOfBoundsException();
    if ( y < 0 || y >= height)
        throw OutOfBoundsException();

    return kernel[x +y*width];
}
// --------- END FILTER CLASS -----------------------
