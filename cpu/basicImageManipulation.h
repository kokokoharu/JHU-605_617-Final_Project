#ifndef __basicImageManipulation__h
#define __basicImageManipulation__h

#include "Image.h"
#include <iostream>
#include <math.h>

using namespace std;

std::vector<Image> lumiChromi(const Image &im);
Image lumiChromi2rgb(const vector<Image> & lc);
Image gamma_code(const Image &im, float gamma);
Image color2gray(const Image &im,
                 const std::vector<float> &weights = std::vector<float>{0.299, 0.587, 0.114});
#endif
