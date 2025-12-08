#include "Image.h"
#include "basicImageManipulation.h"
#include "hdr.h"
#include "cuda_utils.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>

using namespace std;

// HDR and Tone Mapping on Ante1 images
void testToneMapping_ante1() {

    cout << "\n========== testToneMapping_ante1 ==========" << endl;
    cout << endl;

    // load images
    vector<Image> imSeq;
    imSeq.push_back(gamma_code(Image("../Input/ante1-1.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/ante1-2.png"), 1.0/2.2));

    // create hdr image
    TimePoint t_start_hdr = now();
    Image hdr = makeHdrGpuBasic(imSeq);
    TimePoint t_end_hdr = now();
    double t_hdr = elapsed_ms(t_start_hdr, t_end_hdr);

    // tone map with bilaterial
    TimePoint t_start_tm = now();
    Image tm = toneMap(hdr, 100, 3, 0.1);
    TimePoint t_end_tm = now();
    double t_tm = elapsed_ms(t_start_tm, t_end_tm);

    tm = gamma_code(tm, 2.2);
    tm.write("./Output/ante1-tonedHDRsimple-bilateral.png");

    cout << "========================================" << endl;
    cout << "Overall timing for testToneMapping_ante1:" << endl;
    cout << "  makeHDR: " << t_hdr << " ms" << endl;
    cout << "  toneMap: " << t_tm << " ms" << endl;
    cout << "========================================" << endl;

}

// HDR and Tone Mapping on Ante2 images
void testToneMapping_ante2() {

    cout << "\n========== testToneMapping_ante2 ==========" << endl;
    cout << endl;

    // load images
    vector<Image> imSeq;
    imSeq.push_back(gamma_code(Image("../Input/ante2-1.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/ante2-2.png"), 1.0/2.2));

    // create hdr image
    TimePoint t_start_hdr = now();
    Image hdr = makeHdrGpuBasic(imSeq);
    TimePoint t_end_hdr = now();
    double t_hdr = elapsed_ms(t_start_hdr, t_end_hdr);

    // tone map with bilaterial
    TimePoint t_start_tm = now();
    Image tm = toneMap(hdr, 100, 3, 0.1);
    TimePoint t_end_tm = now();
    double t_tm = elapsed_ms(t_start_tm, t_end_tm);

    tm = gamma_code(tm, 2.2);
    tm.write("./Output/ante2-tonedHDRsimple-bilateral.png");

    cout << "========================================" << endl;
    cout << "Overall timing for testToneMapping_ante2:" << endl;
    cout << "  makeHDR: " << t_hdr << " ms" << endl;
    cout << "  toneMap: " << t_tm << " ms" << endl;
    cout << "========================================" << endl;

}

// HDR and Tone Mapping on Ante3 images
void testToneMapping_ante3() {

    cout << "\n========== testToneMapping_ante3 ==========" << endl;
    cout << endl;

    // load images
    vector<Image> imSeq;
    imSeq.push_back(gamma_code(Image("../Input/ante3-1.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/ante3-2.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/ante3-3.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/ante3-4.png"), 1.0/2.2));

    // create hdr image
    TimePoint t_start_hdr = now();
    Image hdr = makeHdrGpuBasic(imSeq);
    TimePoint t_end_hdr = now();
    double t_hdr = elapsed_ms(t_start_hdr, t_end_hdr);

    // tone map with bilaterial
    TimePoint t_start_tm = now();
    Image tm = toneMap(hdr, 100, 3, 0.1);
    TimePoint t_end_tm = now();
    double t_tm = elapsed_ms(t_start_tm, t_end_tm);

    tm = gamma_code(tm, 2.2f);
    tm.write("./Output/ante3-tonedHDRsimple-bilateral.png");

    cout << "========================================" << endl;
    cout << "Overall timing for testToneMapping_ante3:" << endl;
    cout << "  makeHDR: " << t_hdr << " ms" << endl;
    cout << "  toneMap: " << t_tm << " ms" << endl;
    cout << "========================================" << endl;

}

// HDR and Tone Mapping on Boston Images
void testToneMapping_boston() {

    cout << "\n========== testToneMapping_boston ==========" << endl;
    cout << endl;

    // load images
    vector<Image> imSeq;
    imSeq.push_back(gamma_code(Image("../Input/boston-1.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/boston-2.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/boston-3.png"), 1.0/2.2));

    // create hdr image
    TimePoint t_start_hdr = now();
    Image hdr = makeHdrGpuBasic(imSeq);
    TimePoint t_end_hdr = now();
    double t_hdr = elapsed_ms(t_start_hdr, t_end_hdr);

    // tone map with bilaterial
    TimePoint t_start_tm = now();
    Image tm = toneMap(hdr, 100, 3, 0.1);
    TimePoint t_end_tm = now();
    double t_tm = elapsed_ms(t_start_tm, t_end_tm);

    tm = gamma_code(tm, 2.2);
    tm.write("./Output/boston-tonedHDRsimple-bilateral.png");

    cout << "========================================" << endl;
    cout << "Overall timing for testToneMapping_boston:" << endl;
    cout << "  makeHDR: " << t_hdr << " ms" << endl;
    cout << "  toneMap: " << t_tm << " ms" << endl;
    cout << "========================================" << endl;
}

void testToneMapping_design() {

    cout << "\n========== testToneMapping_design ==========" << endl;
    cout << endl;

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
    TimePoint t_start_hdr = now();
    Image hdr = makeHdrGpuBasic(imSeq);
    TimePoint t_end_hdr = now();
    double t_hdr = elapsed_ms(t_start_hdr, t_end_hdr);

    // Note: bilaterial filtering these images takes a very long time. It is not
    // necessary to attempt this for testing
    TimePoint t_start_tm = now();
    Image tm = toneMap(hdr, 100, 3, 0.1);
    TimePoint t_end_tm = now();
    double t_tm = elapsed_ms(t_start_tm, t_end_tm);

    tm = gamma_code(tm, 2.2);
    tm.write("./Output/design-tonedHDRsimple-bilateral.png");

    cout << "========================================" << endl;
    cout << "Overall timing for testToneMapping_design:" << endl;
    cout << "  makeHDR: " << t_hdr << " ms" << endl;
    cout << "  toneMap: " << t_tm << " ms" << endl;
    cout << "========================================" << endl;
}

void testToneMapping_horse() {

    cout << "\n========== testToneMapping_horse ==========" << endl;
    cout << endl;

    // load images
    vector<Image> imSeq;
    imSeq.push_back(gamma_code(Image("../Input/horse-1.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/horse-2.png"), 1.0/2.2));

    // create hdr image
    TimePoint t_start_hdr = now();
    Image hdr = makeHdrGpuBasic(imSeq);
    TimePoint t_end_hdr = now();
    double t_hdr = elapsed_ms(t_start_hdr, t_end_hdr);

    // Note: bilaterial filtering these images takes a very long time. It is not
    // necessary to attempt this for testing
    TimePoint t_start_tm = now();
    Image tm = toneMap(hdr, 100, 3, 0.1);
    TimePoint t_end_tm = now();
    double t_tm = elapsed_ms(t_start_tm, t_end_tm);

    tm = gamma_code(tm, 2.2);
    tm.write("./Output/horse-tonedHDRsimple-bilateral.png");

    cout << "========================================" << endl;
    cout << "Overall timing for testToneMapping_horse:" << endl;
    cout << "  makeHDR: " << t_hdr << " ms" << endl;
    cout << "  toneMap: " << t_tm << " ms" << endl;
    cout << "========================================" << endl;
}

void testToneMapping_nyc() {

    cout << "\n========== testToneMapping_nyc ==========" << endl;
    cout << endl;

    // load images
    vector<Image> imSeq;
    imSeq.push_back(gamma_code(Image("../Input/nyc-1.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/nyc-2.png"), 1.0/2.2));

    // create hdr image
    TimePoint t_start_hdr = now();
    Image hdr = makeHdrGpuBasic(imSeq);
    TimePoint t_end_hdr = now();
    double t_hdr = elapsed_ms(t_start_hdr, t_end_hdr);

    // Note: bilaterial filtering these images takes a very long time. It is not
    // necessary to attempt this for testing
    TimePoint t_start_tm = now();
    Image tm = toneMap(hdr, 100, 3, 0.1);
    TimePoint t_end_tm = now();
    double t_tm = elapsed_ms(t_start_tm, t_end_tm);

    tm = gamma_code(tm, 2.2);
    tm.write("./Output/nyc-tonedHDRsimple-bilateral.png");

    cout << "========================================" << endl;
    cout << "Overall timing for testToneMapping_nyc:" << endl;
    cout << "  makeHDR: " << t_hdr << " ms" << endl;
    cout << "  toneMap: " << t_tm << " ms" << endl;
    cout << "========================================" << endl;
}

void testToneMapping_sea() {

    cout << "\n========== testToneMapping_sea ==========" << endl;
    cout << endl;

    // load images
    vector<Image> imSeq;
    imSeq.push_back(gamma_code(Image("../Input/sea-1.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/sea-2.png"), 1.0/2.2));

    // create hdr image
    TimePoint t_start_hdr = now();
    Image hdr = makeHdrGpuBasic(imSeq);
    TimePoint t_end_hdr = now();
    double t_hdr = elapsed_ms(t_start_hdr, t_end_hdr);

    // Note: bilaterial filtering these images takes a very long time. It is not
    // necessary to attempt this for testing
    TimePoint t_start_tm = now();
    Image tm = toneMap(hdr, 100, 3, 0.1);
    TimePoint t_end_tm = now();
    double t_tm = elapsed_ms(t_start_tm, t_end_tm);

    tm = gamma_code(tm, 2.2);
    tm.write("./Output/sea-tonedHDRsimple-bilateral.png");

    cout << "========================================" << endl;
    cout << "Overall timing for testToneMapping_sea:" << endl;
    cout << "  makeHDR: " << t_hdr << " ms" << endl;
    cout << "  toneMap: " << t_tm << " ms" << endl;
    cout << "========================================" << endl;
}

void testToneMapping_vine() {

    cout << "\n========== testToneMapping_vine ==========" << endl;
    cout << endl;

    // load images
    vector<Image> imSeq;
    imSeq.push_back(gamma_code(Image("../Input/vine-1.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/vine-2.png"), 1.0/2.2));
    imSeq.push_back(gamma_code(Image("../Input/vine-3.png"), 1.0/2.2));

    // create hdr image
    TimePoint t_start_hdr = now();
    Image hdr = makeHdrGpuBasic(imSeq);
    TimePoint t_end_hdr = now();
    double t_hdr = elapsed_ms(t_start_hdr, t_end_hdr);

    // Note: bilaterial filtering these images takes a very long time. It is not
    // necessary to attempt this for testing
    TimePoint t_start_tm = now();
    Image tm = toneMap(hdr, 100, 3, 0.1);
    TimePoint t_end_tm = now();
    double t_tm = elapsed_ms(t_start_tm, t_end_tm);

    tm = gamma_code(tm, 2.2);
    tm.write("./Output/vine-tonedHDRsimple-bilateral.png");

    cout << "========================================" << endl;
    cout << "Overall timing for testToneMapping_vine:" << endl;
    cout << "  makeHDR: " << t_hdr << " ms" << endl;
    cout << "  toneMap: " << t_tm << " ms" << endl;
    cout << "========================================" << endl;
}

int main() {

    testToneMapping_ante1();
    testToneMapping_ante2();
    testToneMapping_ante3();
    // testToneMapping_boston();
    // testToneMapping_design();
    // testToneMapping_horse();
    // testToneMapping_nyc();
    // testToneMapping_sea();
    // testToneMapping_vine();

    return 0;
}
