#include "filtering.h"
#include <cmath>
#include <cassert>

using namespace std;

// Image bilateral(const Image &im, float sigmaRange, float sigmaDomain, float truncateDomain, bool clamp)
// {
//     Image imFilter(im.width(), im.height(), im.channels());

//     // calculate the filter size
//     int offset   = int(ceil(truncateDomain * sigmaDomain));
//     int sizeFilt = 2*offset + 1;
//     float accum,
//           tmp,
//           range_dist,
//           normalizer,
//           factorDomain,
//           factorRange;

//     // for every pixel in the image
//     for (int z=0; z<imFilter.channels(); z++)
//     for (int y=0; y<imFilter.height(); y++)
//     for (int x=0; x<imFilter.width(); x++)
//     {
//         // initilize normalizer and sum value to 0 for every pixel location
//         normalizer = 0.0f;
//         accum      = 0.0f;

//         // sum over the filter's support
//         for (int yFilter=0; yFilter<sizeFilt; yFilter++)
//         for (int xFilter=0; xFilter<sizeFilt; xFilter++)
//         {
//             // calculate the distance between the 2 pixels (in range)
//             range_dist = 0.0f; // |R-R1|^2 + |G-G1|^2 + |B-B1|^2
//             for (int z1 = 0; z1 < imFilter.channels(); z1++) {
//                 tmp  = im.smartAccessor(x,y,z1,clamp); // center pixel
//                 tmp -= im.smartAccessor(x+xFilter-offset,y+yFilter-offset,z1,clamp); // neighbor
//                 tmp *= tmp; // square
//                 range_dist += tmp;
//             }

//             // calculate the exponential weight from the domain and range
//             factorDomain = exp( - ((xFilter-offset)*(xFilter-offset) +  (yFilter-offset)*(yFilter-offset) )/ (2.0 * sigmaDomain*sigmaDomain ) );
//             factorRange  = exp( - range_dist / (2.0 * sigmaRange*sigmaRange) );

//             normalizer += factorDomain * factorRange;
//             accum += factorDomain * factorRange * im.smartAccessor(x+xFilter-offset,y+yFilter-offset,z,clamp);
//         }

//         // set pixel in filtered image to weighted sum of values in the filter region
//         imFilter(x,y,z) = accum/normalizer;
//     }

//     return imFilter;
// }

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
