#include "Image.h"
#include "basicImageManipulation.h"
#include "hdr.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>

using namespace std;

void testComputeWeight() {
    // load in image and invert gamma correction
    Image im1("../Input/ante1-1.png");
    im1 = gamma_code(im1, 1.0/2.2);

    // compute the weight image and save it
    Image weight = computeWeight(im1, 0.01, 0.99);
    weight.write("./Output/weight.png");
}

void testComputeFactor() {
    // load 2 images
    Image im1("../Input/ante2-1.png");
    Image im2("../Input/ante2-2.png");

    // invert gamma correction
    im1 = gamma_code(im1, 1.0/2.2);
    im2 = gamma_code(im2, 1.0/2.2);

    // compute weight images and save them
    Image w1 = computeWeight(im1);
    Image w2 = computeWeight(im2);
    w1.write("./Output/ante2-w1.png");
    w2.write("./Output/ante2-w2.png");

    // compute the factor
    float factor = computeFactor(im1, w1, im2, w2);
    cout << "factor: " << factor << endl;
}


void testMakeHDR() {
    // load an image sequence
    vector<Image> imSeq;
    imSeq.push_back(gamma_code(Image("../Input/design-1.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/design-2.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/design-3.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/design-4.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/design-5.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/design-6.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/design-7.png"), 1.0/2.2));

    // generate an hdr image
    Image hdr = makeHDR(imSeq);
    //hdr.write("./Output/hdr_image");

    // save out images clipped to different ranges.
    float maxVal = hdr.max();
    Image hdrScale0 = gamma_code(hdr/maxVal, 2.2);
    hdrScale0.write("./Output/scaledHDR_design_0.png");
    Image hdrScale2 = gamma_code((2e2)*hdr/maxVal, 2.2);
    hdrScale2.write("./Output/scaledHDR_design_2.png");
    Image hdrScale4 = gamma_code((2e4)*hdr/maxVal, 2.2);
    hdrScale4.write("./Output/scaledHDR_design_4.png");
    Image hdrScale6 = gamma_code((2e6)*hdr/maxVal, 2.2);
    hdrScale6.write("./Output/scaledHDR_design_6.png");
    Image hdrScale8 = gamma_code((2e8)*hdr/maxVal, 2.2);
    hdrScale8.write("./Output/scaledHDR_design_8.png");
    Image hdrScale10 = gamma_code((2e10)*hdr/maxVal, 2.2);
    hdrScale10.write("./Output/scaledHDR_design_10.png");

}

void testToneMapping_ante1() {

    // load images
    vector<Image> imSeq;
    imSeq.push_back(gamma_code(Image("../Input/ante1-1.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/ante1-2.png"), 1.0/2.2));

    // create hdr image
    auto start = chrono::high_resolution_clock::now();
    Image hdr = makeHDR(imSeq);
    auto end = chrono::high_resolution_clock::now();
    double makeHDR_time = chrono::duration<double, milli>(end - start).count();

    // tone map with bilaterial
    start = chrono::high_resolution_clock::now();
    Image tm = toneMap(hdr, 100, 3, 0.1);
    end = chrono::high_resolution_clock::now();
    double toneMap_bilateral_time = chrono::duration<double, milli>(end - start).count();
    tm = gamma_code(tm, 2.2);
    tm.write("./Output/ante1-tonedHDRsimple-bilateral.png");

    cout << "========================================" << endl;
    cout << "Overall timing for testToneMapping_ante1:" << endl;
    cout << "  makeHDR: " << makeHDR_time << " ms" << endl;
    cout << "  toneMap (bilateral): " << toneMap_bilateral_time << " ms" << endl;
    cout << "========================================" << endl;

}

// HDR and Tone Mapping on Ante2 images
void testToneMapping_ante2() {

    // load images
    vector<Image> imSeq;
    imSeq.push_back(gamma_code(Image("../Input/ante2-1.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/ante2-2.png"), 1.0/2.2));

    // create hdr image
    auto start = chrono::high_resolution_clock::now();
    Image hdr = makeHDR(imSeq);
    auto end = chrono::high_resolution_clock::now();
    double makeHDR_time = chrono::duration<double, milli>(end - start).count();

    // tone map with bilaterial
    start = chrono::high_resolution_clock::now();
    Image tm = toneMap(hdr, 100, 3, 0.1);
    end = chrono::high_resolution_clock::now();
    double toneMap_bilateral_time = chrono::duration<double, milli>(end - start).count();
    tm = gamma_code(tm, 2.2);
    tm.write("./Output/ante2-tonedHDRsimple-bilateral.png");

    cout << "========================================" << endl;
    cout << "Overall timing for testToneMapping_ante2:" << endl;
    cout << "  makeHDR: " << makeHDR_time << " ms" << endl;
    cout << "  toneMap (bilateral): " << toneMap_bilateral_time << " ms" << endl;
    cout << "========================================" << endl;

}

// HDR and Tone Mapping on Ante3 images
void testToneMapping_ante3() {

    // load images
    vector<Image> imSeq;
    imSeq.push_back(gamma_code(Image("../Input/ante3-1.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/ante3-2.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/ante3-3.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/ante3-4.png"), 1.0/2.2));

    // create hdr image
    auto start = chrono::high_resolution_clock::now();
    Image hdr = makeHDR(imSeq);
    auto end = chrono::high_resolution_clock::now();
    double makeHDR_time = chrono::duration<double, milli>(end - start).count();

    // tone map with bilaterial
    start = chrono::high_resolution_clock::now();
    Image tm = toneMap(hdr, 100, 3, 0.1);
    end = chrono::high_resolution_clock::now();
    double toneMap_bilateral_time = chrono::duration<double, milli>(end - start).count();
    tm = gamma_code(tm, 2.2f);
    tm.write("./Output/ante3-tonedHDRsimple-bilateral.png");

    cout << "========================================" << endl;
    cout << "Overall timing for testToneMapping_ante3:" << endl;
    cout << "  makeHDR: " << makeHDR_time << " ms" << endl;
    cout << "  toneMap (bilateral): " << toneMap_bilateral_time << " ms" << endl;
    cout << "========================================" << endl;

}

// HDR and Tone Mapping on Boston Images
void testToneMapping_boston() {

    // load images
    vector<Image> imSeq;
    imSeq.push_back(gamma_code(Image("../Input/boston-1.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/boston-2.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/boston-3.png"), 1.0/2.2));

    // create hdr image
    auto start = chrono::high_resolution_clock::now();
    Image hdr = makeHDR(imSeq);
    auto end = chrono::high_resolution_clock::now();
    double makeHDR_time = chrono::duration<double, milli>(end - start).count();

    // tone map with bilaterial
    start = chrono::high_resolution_clock::now();
    Image tm = toneMap(hdr, 100, 3, 0.1);
    end = chrono::high_resolution_clock::now();
    double toneMap_bilateral_time = chrono::duration<double, milli>(end - start).count();
    tm = gamma_code(tm, 2.2);
    tm.write("./Output/boston-tonedHDRsimple-bilateral.png");

    cout << "========================================" << endl;
    cout << "Overall timing for testToneMapping_boston:" << endl;
    cout << "  makeHDR: " << makeHDR_time << " ms" << endl;
    cout << "  toneMap (bilateral): " << toneMap_bilateral_time << " ms" << endl;
    cout << "========================================" << endl;
}

void testToneMapping_design() {

    // load images
    vector<Image> imSeq;
    imSeq.push_back(gamma_code(Image("../Input/design-1.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/design-2.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/design-3.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/design-4.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/design-5.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/design-6.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/design-7.png"), 1.0/2.2));

    // create hdr image
    auto start = chrono::high_resolution_clock::now();
    Image hdr = makeHDR(imSeq);
    auto end = chrono::high_resolution_clock::now();
    double makeHDR_time = chrono::duration<double, milli>(end - start).count();

    // Note: bilaterial filtering these images takes a very long time. It is not
    // necessary to attempt this for testing
    // tone map with bilaterial
    start = chrono::high_resolution_clock::now();
    Image tm = toneMap(hdr, 100, 3, 0.1);
    end = chrono::high_resolution_clock::now();
    double toneMap_bilateral_time = chrono::duration<double, milli>(end - start).count();
    tm = gamma_code(tm, 2.2);
    tm.write("./Output/design-tonedHDRsimple-bilateral.png");

    cout << "========================================" << endl;
    cout << "Overall timing for testToneMapping_design:" << endl;
    cout << "  makeHDR: " << makeHDR_time << " ms" << endl;
    cout << "  toneMap (bilateral): " << toneMap_bilateral_time << " ms" << endl;
    cout << "========================================" << endl;
}

void testToneMapping_horse() {

    // load images
    vector<Image> imSeq;
    imSeq.push_back(gamma_code(Image("../Input/horse-1.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/horse-2.png"), 1.0/2.2));

    // create hdr image
    auto start = chrono::high_resolution_clock::now();
    Image hdr = makeHDR(imSeq);
    auto end = chrono::high_resolution_clock::now();
    double makeHDR_time = chrono::duration<double, milli>(end - start).count();

    // Note: bilaterial filtering these images takes a very long time. It is not
    // necessary to attempt this for testing
    // tone map with bilaterial
    start = chrono::high_resolution_clock::now();
    Image tm = toneMap(hdr, 100, 3, 0.1);
    end = chrono::high_resolution_clock::now();
    double toneMap_bilateral_time = chrono::duration<double, milli>(end - start).count();
    tm = gamma_code(tm, 2.2);
    tm.write("./Output/horse-tonedHDRsimple-bilateral.png");

    cout << "========================================" << endl;
    cout << "Overall timing for testToneMapping_horse:" << endl;
    cout << "  makeHDR: " << makeHDR_time << " ms" << endl;
    cout << "  toneMap (bilateral): " << toneMap_bilateral_time << " ms" << endl;
    cout << "========================================" << endl;
}

void testToneMapping_nyc() {

    // load images
    vector<Image> imSeq;
    imSeq.push_back(gamma_code(Image("../Input/nyc-1.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/nyc-2.png"), 1.0/2.2));

    // create hdr image
    auto start = chrono::high_resolution_clock::now();
    Image hdr = makeHDR(imSeq);
    auto end = chrono::high_resolution_clock::now();
    double makeHDR_time = chrono::duration<double, milli>(end - start).count();

    // Note: bilaterial filtering these images takes a very long time. It is not
    // necessary to attempt this for testing
    // tone map with bilaterial
    start = chrono::high_resolution_clock::now();
    Image tm = toneMap(hdr, 100, 3, 0.1);
    end = chrono::high_resolution_clock::now();
    double toneMap_bilateral_time = chrono::duration<double, milli>(end - start).count();
    tm = gamma_code(tm, 2.2);
    tm.write("./Output/nyc-tonedHDRsimple-bilateral.png");

    cout << "========================================" << endl;
    cout << "Overall timing for testToneMapping_nyc:" << endl;
    cout << "  makeHDR: " << makeHDR_time << " ms" << endl;
    cout << "  toneMap (bilateral): " << toneMap_bilateral_time << " ms" << endl;
    cout << "========================================" << endl;
}

void testToneMapping_sea() {

    // load images
    vector<Image> imSeq;
    imSeq.push_back(gamma_code(Image("../Input/sea-1.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/sea-2.png"), 1.0/2.2));

    // create hdr image
    auto start = chrono::high_resolution_clock::now();
    Image hdr = makeHDR(imSeq);
    auto end = chrono::high_resolution_clock::now();
    double makeHDR_time = chrono::duration<double, milli>(end - start).count();

    // Note: bilaterial filtering these images takes a very long time. It is not
    // necessary to attempt this for testing
    // tone map with bilaterial
    start = chrono::high_resolution_clock::now();
    Image tm = toneMap(hdr, 100, 3, 0.1);
    end = chrono::high_resolution_clock::now();
    double toneMap_bilateral_time = chrono::duration<double, milli>(end - start).count();
    tm = gamma_code(tm, 2.2);
    tm.write("./Output/sea-tonedHDRsimple-bilateral.png");

    cout << "========================================" << endl;
    cout << "Overall timing for testToneMapping_sea:" << endl;
    cout << "  makeHDR: " << makeHDR_time << " ms" << endl;
    cout << "  toneMap (bilateral): " << toneMap_bilateral_time << " ms" << endl;
    cout << "========================================" << endl;
}

void testToneMapping_vine() {

    // load images
    vector<Image> imSeq;
    imSeq.push_back(gamma_code(Image("../Input/vine-1.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/vine-2.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/vine-3.png"), 1.0/2.2));

    // create hdr image
    auto start = chrono::high_resolution_clock::now();
    Image hdr = makeHDR(imSeq);
    auto end = chrono::high_resolution_clock::now();
    double makeHDR_time = chrono::duration<double, milli>(end - start).count();

    // Note: bilaterial filtering these images takes a very long time. It is not
    // necessary to attempt this for testing
    // tone map with bilaterial
    start = chrono::high_resolution_clock::now();
    Image tm = toneMap(hdr, 100, 3, 0.1);
    end = chrono::high_resolution_clock::now();
    double toneMap_bilateral_time = chrono::duration<double, milli>(end - start).count();
    tm = gamma_code(tm, 2.2);
    tm.write("./Output/vine-tonedHDRsimple-bilateral.png");

    cout << "========================================" << endl;
    cout << "Overall timing for testToneMapping_vine:" << endl;
    cout << "  makeHDR: " << makeHDR_time << " ms" << endl;
    cout << "  toneMap (bilateral): " << toneMap_bilateral_time << " ms" << endl;
    cout << "========================================" << endl;
}

int main() {

    testToneMapping_ante1();
    testToneMapping_ante2();
    testToneMapping_ante3();
    testToneMapping_boston();
    testToneMapping_design();
    //testToneMapping_horse();
    //testToneMapping_nyc();
    //testToneMapping_sea();
    //testToneMapping_vine();

    return 0;
}
