#ifndef __morphing__h
#define __morphing__h

#include <cmath>
#include <iostream>

#include "basicImageManipulation.h"
#include "Image.h"

using namespace std;

// Bilaterial Filtering
Image bilateral(const Image &im, float sigmaRange=0.1, float sigmaDomain=1.0, float truncateDomain=3.0, bool clamp=true);

#endif
