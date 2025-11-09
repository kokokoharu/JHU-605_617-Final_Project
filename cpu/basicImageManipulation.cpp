#include "basicImageManipulation.h"
using namespace std;

// For this function, we want two outputs, a single channel luminance image
// and a three channel chrominance image. Return them in a vector with luminance first
std::vector<Image> lumiChromi(const Image &im) {
    // // Create the luminance image
    // // Create the chrominance image
    // // Create the output vector as (luminance, chrominance)

    // Create the luminance
    Image im_luminance = color2gray(im);

    // Create chrominance images
    // We copy the input as starting point for the chrominance
    Image im_chrominance = im;
    for (int c = 0 ; c < im.channels(); c++ ) {
        for (int y = 0 ; y < im.height(); y++) {
            for (int x = 0 ; x < im.width(); x++) {
                im_chrominance(x,y,c) = im_chrominance(x,y,c) / im_luminance(x,y);
            }
        }
    }

    // Stack luminance and chrominance in the output vector, luminance first
    return std::vector<Image>{im_luminance, im_chrominance};
}

Image lumiChromi2rgb(const vector<Image> & lc) {
    // luminance is lc[0]
    // chrominance is lc[1]

    // Create chrominance images
    // We copy the input as starting point for the chrominance
    Image im = Image(lc[1].width(), lc[1].height(), lc[1].channels());
    for (int c = 0 ; c < im.channels(); c++ ) {
      for (int y = 0 ; y < im.height(); y++) {
        for (int x = 0 ; x < im.width(); x++) {
            im(x,y,c) = lc[1](x,y,c) * lc[0](x,y);
        }
      }
    }
    return im;
}

Image gamma_code(const Image &im, float gamma) {
    Image output = Image(im.width(), im.height(), im.channels());
    for (int i = 0; i < im.number_of_elements(); ++i){
        output(i) = pow(im(i), (1/gamma));
    }
    return output;
}

Image color2gray(const Image &im, const std::vector<float> &weights) {
    Image output(im.width(), im.height(), 1);
    for (int i = 0 ; i < im.width(); i++ ) {
        for (int j = 0 ; j < im.height(); j++ ) {
            output(i,j,0) = im(i,j,0) * weights[0] + im(i,j,1) * weights[1] + im(i,j,2) *weights[2];
        }
    }
    return output;
}