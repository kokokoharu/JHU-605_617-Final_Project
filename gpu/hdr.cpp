#include "hdr.h"
#include "filtering.h"
#include "cuda_utils.h"
#include <math.h>
#include <algorithm>
#include <iostream>

using namespace std;

float computeFactor(const Image &im1, const Image &w1, const Image &im2, const Image &w2){
    // Compute the multiplication factor between a pair of images. This
    // gives us the relative exposure between im1 and im2. It is computed as
    // the median of im2/(im1+eps) for some small eps, taking into account
    // pixels that are valid in both images.
    float factor;
    Image new_im1 = im1 + pow(10,-10);    // no one wants a division by 0
    vector<float> useful_pixels;
    int length;

    for (int k=0; k<im2.channels(); k++) {
        for (int j=0; j<im2.height(); j++) {
            for (int i=0; i<im2.width(); i++) {
                if ((w1(i,j,k)==1.0) && (w2(i,j,k)==1.0)) useful_pixels.push_back(im2(i,j,k)/new_im1(i,j,k));
            }
        }
    }

    length = useful_pixels.size();
    if (length==0) {
        cout << "no useful pixel, returning 0" << endl;
        return 0.0;
    }

    sort(useful_pixels.begin(),useful_pixels.end());

    if (fmod(length,2)==0.0) {
        factor = 0.5*(useful_pixels[length/2-1]+useful_pixels[length/2]);
    }
    else {
        factor = useful_pixels[(length-1)/2];
    }

    return factor;

}

/**************************************************************
 //                      TONE MAPPING                        //
 *************************************************************/


Image toneMap(const Image &im, float targetBase, float detailAmp, float sigmaRange) {
    // tone map an hdr image
    // - Split the image into its luminance-chrominance components.
    // - Work in the log10 domain for the luminance

    // Timing
    TimePoint t_start_total = now();
    TimePoint t_start, t_end;

    // Step 1: lumiChromi
    t_start = now();
    vector<Image> lc = lumiChromi(im);
    Image lumi = lc[0];
    Image chromi = lc[1];
    t_end = now();
    double t_lumiChromi = elapsed_ms(t_start, t_end);

    // Step 2: convert to log domain
    t_start = now();
    Image log_lumi = log10Image(lumi);
    t_end = now();
    double t_log10Image = elapsed_ms(t_start, t_end);

    // Step 3: find domain standard deviation
    t_start = now();
    int largest;
    im.channels()<im.width() ? largest=im.width() : largest=im.channels();
    im.width()<im.height() ? largest=im.height() : largest=im.width();
    float sigmaDomain = largest/(float)50;
    t_end = now();
    double t_sigmaDomain = elapsed_ms(t_start, t_end);

    // Step 4: find base (low freq)
    t_start = now();
    Image base(log_lumi.width(),log_lumi.height(),log_lumi.channels());
    base = bilateralGpuBasic(log_lumi, sigmaRange, sigmaDomain);
    t_end = now();
    double t_bilateral = elapsed_ms(t_start, t_end);

    // Step 5: find detail (high freq)
    t_start = now();
    Image detail = log_lumi - base;
    t_end = now();
    double t_detail = elapsed_ms(t_start, t_end);

    // Step 6: find scale factor
    t_start = now();
    float k;    // k = lg(target)/(baseMax-baseMin)
    k = log10(targetBase)/(base.max()-base.min());
    t_end = now();
    double t_scaleFactor = elapsed_ms(t_start, t_end);

    // Step 7: new log-luminance
    t_start = now();
    Image new_log_lumi = detailAmp*detail + k*(base-base.max());
    t_end = now();
    double t_newLogLumi = elapsed_ms(t_start, t_end);

    // Step 8: convert back to linear domain
    t_start = now();
    Image new_lumi = exp10Image(new_log_lumi);
    t_end = now();
    double t_exp10Image = elapsed_ms(t_start, t_end);

    // Step 9: back to rgb
    t_start = now();
    Image result = lumiChromi2rgb(vector<Image>{new_lumi, chromi});
    t_end = now();
    double t_lumiChromi2rgb = elapsed_ms(t_start, t_end);

    // Timing output
    TimePoint t_end_total = now();
    double t_total = elapsed_ms(t_start_total, t_end_total);

    cout << "=== toneMap (bilateral) ===" << endl;
    cout << "  lumiChromi: " << t_lumiChromi << " ms" << endl;
    cout << "  log10Image: " << t_log10Image << " ms" << endl;
    cout << "  sigmaDomain calc: " << t_sigmaDomain << " ms" << endl;
    cout << "  bilateral (GPU): " << t_bilateral << " ms" << endl;
    cout << "  detail computation: " << t_detail << " ms" << endl;
    cout << "  scale factor k: " << t_scaleFactor << " ms" << endl;
    cout << "  new_log_lumi: " << t_newLogLumi << " ms" << endl;
    cout << "  exp10Image: " << t_exp10Image << " ms" << endl;
    cout << "  lumiChromi2rgb: " << t_lumiChromi2rgb << " ms" << endl;
    cout << "Total toneMap (bilateral): " << t_total << " ms" << endl;
    cout << endl;

    return result;

}


/*********************************************************************
 *                       Tone mapping helpers                        *
 *********************************************************************/

// image --> log10Image
Image log10Image(const Image &im) {
    // Taking a linear image im, transform to log10 scale.
    // To avoid infinity issues, make any 0-valued pixel be equal the the minimum
    // non-zero value. See image_minnonzero(im).
    float eps = image_minnonzero(im);
    Image log_im(im.width(),im.height(),im.channels());

    for (int k=0; k<im.channels(); k++) {
        for (int j=0; j<im.height(); j++) {
            for (int i=0; i<im.width(); i++) {
                if (im(i,j,k) == 0) {log_im(i,j,k) = log10(eps);}
                else {log_im(i,j,k) = log10(im(i,j,k));}
            }
        }
    }

    return log_im;

}

// Image --> 10^Image
Image exp10Image(const Image &im) {
    // take an image in log10 domain and transform it back to linear domain.
    // see pow(a, b)
    Image lin_im(im.width(), im.height(), im.channels());

    for (int k=0; k<im.channels(); k++) {
        for (int j=0; j<im.height(); j++) {
            for (int i=0; i<im.width(); i++) {
                lin_im(i,j,k) = pow(10,im(i,j,k));
            }
        }
    }

    return lin_im;

}

// min non-zero pixel value of image
float image_minnonzero(const Image &im) {
    // return the smallest value in the image that is non-zeros (across all
    // channels too)

    float smallest = 1.0;   // get it into the loop
    for (int k=0; k<im.channels(); k++) {
        for (int j=0; j<im.height(); j++) {
            for (int i=0; i<im.width(); i++) {
                if (im(i,j,k) < smallest) smallest = im(i,j,k);
            }
        }
    }

    return smallest;

}
