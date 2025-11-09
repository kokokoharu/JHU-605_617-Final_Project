#include "hdr.h"
#include "filtering.h"
#include <math.h>
#include <algorithm>


using namespace std;

/**************************************************************
 //                       HDR MERGING                        //
 *************************************************************/

Image computeWeight(const Image &im, float epsilonMini, float epsilonMaxi){
    // Generate a weight image that indicates which pixels are good to use in
    // HDR, i.e. weight=1 when the pixel value is in [epsilonMini, epsilonMaxi].
    // The weight is per pixel, per channel.
    Image weight(im.width(),im.height(),im.channels());
    for (int k=0; k<im.channels(); k++) {
        for (int j=0; j<im.height(); j++) {
            for (int i=0; i<im.width(); i++) {
                (im(i,j,k)<=epsilonMaxi) && (im(i,j,k)>=epsilonMini) ? weight(i,j,k)=1.0 : weight(i,j,k)=0.0;
            }
        }
    }
    return weight;
}


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


Image makeHDR(vector<Image> &imSeq, float epsilonMini, float epsilonMaxi){
    // Merge images to make a single hdr image
    // For each image in the sequence, compute the weight map (special cases
    // for the first and last images).
    // Compute the exposure factor for each consecutive pair of image.
    // Write the valid pixel to your hdr output, taking care of rescaling them
    // properly using the factor.

    // if only one image in imSeq, prompt and return that image
    if (imSeq.size() == 1) {
        cout << "provide more than one image" << endl;
        return imSeq[0];
    }

    Image hdr(imSeq[0].width(), imSeq[0].height(), imSeq[0].channels());
    Image first = imSeq[0];
    Image last = imSeq[imSeq.size()-1];

    // compute weight of the first image
    Image weight_first(first.width(),first.height(),first.channels());
    for (int k=0; k<first.channels(); k++) {
        for (int j=0; j<first.height(); j++) {
            for (int i=0; i<first.width(); i++) {
                first(i,j,k)>=epsilonMini ? weight_first(i,j,k)=1.0 : weight_first(i,j,k)=0.0;
            }
        }
    }

    // compute weight of the last image
    Image weight_last(last.width(), last.height(), last.channels());
    for (int k=0; k<last.channels(); k++) {
        for (int j=0; j<last.height(); j++) {
            for (int i=0; i<last.width(); i++) {
                last(i,j,k)<=epsilonMaxi ? weight_last(i,j,k)=1.0 : weight_last(i,j,k)=0.0;
            }
        }
    }


    // vector to store sequence of weight maps and factors
    vector<Image> weight_maps;
    vector<float> factors;

    // push the first element
    weight_maps.push_back(weight_first);
    factors.push_back(1.0);   // assume k0=1, first image reference, others scaled to it

    Image current_weight = weight_first;
    float current_ratio;

    // compute all weight maps and factors
    for (int im=1; im<(int)imSeq.size()-1; im++) {
        Image w_current = computeWeight(imSeq[im]);
        weight_maps.push_back(w_current);

        if (im == 1) {
            Image w1 = computeWeight(imSeq[1]);
            current_ratio = computeFactor(first,weight_first,imSeq[1],w1);  // k_1/k_0
        }
        else {
            Image w_prev = computeWeight(imSeq[im-1]);
            Image w_curr = computeWeight(imSeq[im]);
            current_ratio = computeFactor(imSeq[im-1],w_prev,imSeq[im],w_curr);    // k_i/k_(i-1)
        }
        factors.push_back(current_ratio*factors[im-1]);   // k_i
    }

    // push the last element
    weight_maps.push_back(weight_last);

    Image w_second_last = computeWeight(imSeq[imSeq.size()-2]);
    float final_factor = computeFactor(imSeq[imSeq.size()-2],w_second_last,last,weight_last);
    factors.push_back(factors[factors.size()-1]*final_factor);


    // merging1, loop through all images for each pixel
    float weight_accum;

    for (int k=0; k<first.channels(); k++) {
        for (int j=0; j<first.height(); j++) {
            for (int i=0; i<first.width(); i++) {
                weight_accum = 0.0;   // reset accmulate weight before processing a pixel
                for (int im=0; im<(int)imSeq.size(); im++) {
                    weight_accum += weight_maps[im](i,j,k);
                    hdr(i,j,k) = hdr(i,j,k) + weight_maps[im](i,j,k)*(1/factors[im])*imSeq[im](i,j,k) ;
                }
                if (weight_accum == 0.0) {hdr(i,j,k) = first(i,j,k);}
                else {hdr(i,j,k) = hdr(i,j,k)/weight_accum;}
            }
        }
    }

    // // merging2, loop through all pixels for each image
    // Image current_im = imSeq[0];
    // Image weight_accum(weight_maps[0].width(),weight_maps[0].height(),weight_maps[0].channels());
    // float current_factor;
    //
    // for (int im=0; im<imSeq.size(); im++) {
    //     current_weight = weight_maps[im];
    //     current_factor = factors[im];
    //     current_im = imSeq[im];
    //     for (int k=0; k<hdr.channels(); k++) {
    //         for (int j=0; j<hdr.height(); j++) {
    //             for (int i=0; i<hdr.width(); i++) {
    //                 hdr(i,j,k) = hdr(i,j,k) + 1/current_factor*current_weight(i,j,k)*current_im(i,j,k);
    //                 weight_accum(i,j,k) = weight_accum(i,j,k) + current_weight(i,j,k);
    //             }
    //         }
    //     }
    // }
    //
    // for (int k=0; k<hdr.channels(); k++) {
    //     for (int j=0; j<hdr.height(); j++) {
    //         for (int i=0; i<hdr.width(); i++) {
    //             if (weight_accum(i,j,k) == 0) {
    //                 hdr(i,j,k) = first(i,j,k);
    //             }
    //             else {
    //                 hdr(i,j,k) = hdr(i,j,k)/weight_accum(i,j,k);
    //             }
    //         }
    //     }
    // }

    return hdr;

}

/**************************************************************
 //                      TONE MAPPING                        //
 *************************************************************/


Image toneMap(const Image &im, float targetBase, float detailAmp, float sigmaRange) {
    // tone map an hdr image
    // - Split the image into its luminance-chrominance components.
    // - Work in the log10 domain for the luminance

    // Step 1: lumiChromi
    vector<Image> lc = lumiChromi(im);
    Image lumi = lc[0];
    Image chromi = lc[1];

    // Step 2: convert to log domain
    Image log_lumi = log10Image(lumi);

    // Step 3: find domain standard deviation
    int largest;
    im.channels()<im.width() ? largest=im.width() : largest=im.channels();
    im.width()<im.height() ? largest=im.height() : largest=im.width();
    float sigmaDomain = largest/(float)50;

    // Step 4: find base (low freq)
    Image base(log_lumi.width(),log_lumi.height(),log_lumi.channels());
    base = bilateralGpuBasic(log_lumi, sigmaRange, sigmaDomain);

    // Step 5: find detail (high freq)
    Image detail = log_lumi - base;

    // Step 6: find scale factor
    float k;    // k = lg(target)/(baseMax-baseMin)
    k = log10(targetBase)/(base.max()-base.min());

    // Step 7: new log-luminance
    Image new_log_lumi = detailAmp*detail + k*(base-base.max());

    // Step 8: convert back to linear domain
    Image new_lumi = exp10Image(new_log_lumi);

    // Step 9: back to rgb
    Image result = lumiChromi2rgb(vector<Image>{new_lumi, chromi});

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
