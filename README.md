# High Dynamic Range (HDR) Image Processing

**JHU-605/617 GPU Programming Course - Final Project**

## Overview

This project implements High Dynamic Range (HDR) imaging algorithms that merge multiple exposure photographs into a single HDR image, followed by tone mapping for display on standard monitors. The implementation features two versions for performance comparison:

- **CPU Implementation** (`/cpu`): Host-only baseline implementation using standard C++
- **GPU Implementation** (`/gpu`): CUDA-accelerated version for performance optimization

The dual implementation approach allows direct performance comparison between CPU and GPU execution, with detailed timing instrumentation for both versions.

## Repository Structure

```
JHU-605_617-Final_Project/
├── README.md              # This file
├── CLAUDE.md              # Project documentation
├── Input/                 # Shared input images (27 PNG files across 9 sequences)
│   ├── ante1-1.png, ante1-2.png        # 2 exposures
│   ├── ante2-1.png, ante2-2.png        # 2 exposures
│   ├── ante3-1.png ... ante3-4.png     # 4 exposures
│   ├── boston-1.png ... boston-3.png   # 3 exposures
│   ├── design-1.png ... design-7.png   # 7 exposures (large images)
│   ├── horse-1.png, horse-2.png        # 2 exposures
│   ├── nyc-1.png, nyc-2.png            # 2 exposures
│   ├── sea-1.png, sea-2.png            # 2 exposures
│   └── vine-1.png ... vine-3.png       # 3 exposures
├── cpu/                   # CPU baseline implementation
└── gpu/                   # GPU-accelerated implementation
```

## CPU Implementation (`/cpu`)

### Purpose
The CPU implementation serves as the baseline reference implementation, providing:
- Correctness validation for algorithms
- Performance baseline for GPU comparison
- Standard C++ image processing operations

### Directory Structure

```
cpu/
├── main.cpp                        # Test cases and entry point
├── Image.h / Image.cpp             # Core image class (multi-dimensional)
├── hdr.h / hdr.cpp                 # HDR merging and tone mapping algorithms
├── filtering.h / filtering.cpp     # Filtering operations (Gaussian, bilateral)
├── basicImageManipulation.h/cpp    # Color conversions and gamma correction
├── lodepng.h / lodepng.cpp         # PNG I/O library
├── ImageException.h                # Error handling
├── Makefile                        # Build system
├── _build/                         # Build artifacts (generated)
├── main                            # Executable (generated)
└── Output/                         # Output images (generated)
```

### Build Instructions

```bash
cd cpu/

# Build the executable
make

# Build and run all tests
make run

# Clean build artifacts
make clean
```

**Requirements:**
- C++ compiler: g++
- C++ Standard: C++11
- No CUDA required

## GPU Implementation (`/gpu`)

### Purpose
The GPU implementation provides CUDA-accelerated versions of computationally intensive operations:
- Weight map computation (GPU kernels)
- HDR image merging (GPU kernels)
- Bilateral filtering (GPU kernels)
- Performance profiling with CUDA events

### Target GPU
- **Model:** Nvidia Tesla T4
- **Architecture:** Turing
- **Compute Capability:** 7.5
- **Makefile Setting:** `CUDA_ARCH := sm_75`

### Directory Structure

```
gpu/
├── main.cpp                        # Test cases with GPU calls
├── Image.h / Image.cpp             # Core image class (shared with CPU)
├── hdr.h / hdr.cpp                 # HDR host-side code
├── hdr.cu                          # HDR CUDA kernels (computeWeightGpu, hdrMergeGpu)
├── filtering.h / filtering.cpp     # Filtering host-side code
├── filtering.cu                    # Bilateral filtering CUDA kernel
├── cuda_utils.h                    # CUDA utilities (error checking, timing, device functions)
├── basicImageManipulation.h/cpp    # Color conversions and gamma correction
├── lodepng.h / lodepng.cpp         # PNG I/O library
├── ImageException.h                # Error handling
├── Makefile                        # CUDA build system
├── _build/                         # Build artifacts (generated)
├── main                            # Executable (generated)
└── Output/                         # Output images (generated)
```

### Build Instructions

```bash
cd gpu/

# Build the executable with CUDA
make

# Build and run all tests
make run

# Clean build artifacts
make clean
```

**Requirements:**
- CUDA Toolkit (nvcc compiler)
- C++ compiler: g++
- C++ Standard: C++11
- CUDA-capable GPU (tested on Nvidia T4)

## Important Assumptions

### Input Image Requirements

**⚠️ Critical:** The HDR algorithm makes the following assumptions about input images:

1. **Linear Encoding:**
   - Input images should be in **linear light space** (not gamma-encoded)
   - If images are in sRGB space (gamma-encoded), they must be linearized using `gamma_code(im, 1.0/2.2)` before HDR processing
   - Output images must be converted back to sRGB using `gamma_code(result, 2.2)` before saving

2. **Exposure Ordering:**
   - Image sequence **MUST** be ordered from **least exposed (darkest) first** to **most exposed (brightest) last**
   - Example correct order: `[underexposed.png, normal.png, overexposed.png]`
   - This ordering is critical for the weight map computation and factor estimation algorithms

**Why This Matters:**
The HDR merging algorithm computes different weight maps for the first and last images in the sequence. Incorrect ordering will result in improper weighting and poor HDR quality.

## Features and Capabilities

### 1. HDR Merging
**CPU:** `makeHDR()`
**GPU:** `makeHdrGpuBasic()`

- Combines multiple exposure images into a single HDR image
- Automatic exposure ratio detection using median-based factor computation
- Weight-based merging that excludes over/under-exposed pixels

### 2. Tone Mapping
**Both:** `toneMap()` (uses CPU or GPU bilateral filter)

- Compresses high dynamic range to displayable range (LDR)
- Log-domain base/detail separation
- Edge-preserving bilateral filtering for detail enhancement
- Adjustable parameters:
  - `targetBase`: Brightness level (default: 100)
  - `detailAmp`: Detail amplification factor (default: 3)
  - `sigmaRange`: Bilateral filter range parameter (default: 0.1)

### 3. Performance Profiling
- **CPU:** `std::chrono` for host-side timing
- **GPU:** CUDA events for kernel timing + chrono for total time
- Detailed breakdown of HDR merging and tone mapping stages
- Timing output format matches between CPU and GPU for easy comparison

## HDR Processing Pipeline

```
1. Load Images
   └─> Read multiple exposure images from Input/

2. Linearization
   └─> gamma_code(im, 1.0/2.2)  # Convert sRGB to linear space

3. Compute Weights
   └─> Identify well-exposed pixels for each image
       - First image: pixels >= epsilonMini
       - Last image: pixels <= epsilonMaxi
       - Middle images: epsilonMini <= pixels <= epsilonMaxi

4. Estimate Exposure Factors
   └─> Compute relative exposure ratios between consecutive images
       Using median of pixel ratios in valid regions

5. Merge HDR
   └─> Weighted average combining all exposures
       - GPU: computeWeightGpu + hdrMergeGpu kernels
       - CPU: Nested loop implementation

6. Tone Mapping
   └─> Compress dynamic range for display
       - Luminance/chrominance separation
       - Log-domain processing
       - Bilateral filtering (GPU: bilateral_basic kernel)
       - Detail enhancement

7. Display Encoding
   └─> gamma_code(result, 2.2)  # Convert back to sRGB

8. Save Output
   └─> Write tone-mapped image to Output/
```

## Test Cases

The project includes 9 test sequences with varying exposure counts and image sizes:

| Test Function | Images | Exposures | Notes |
|--------------|--------|-----------|-------|
| `testToneMapping_ante1()` | ante1-1, ante1-2 | 2 | Small, fast |
| `testToneMapping_ante2()` | ante2-1, ante2-2 | 2 | Small, fast |
| `testToneMapping_ante3()` | ante3-1 through ante3-4 | 4 | Medium |
| `testToneMapping_boston()` | boston-1 through boston-3 | 3 | Medium |
| `testToneMapping_design()` | design-1 through design-7 | 7 | **Large, slow** |
| `testToneMapping_horse()` | horse-1, horse-2 | 2 | Medium |
| `testToneMapping_nyc()` | nyc-1, nyc-2 | 2 | Medium |
| `testToneMapping_sea()` | sea-1, sea-2 | 2 | Medium |
| `testToneMapping_vine()` | vine-1 through vine-3 | 3 | Medium |

## Usage Examples

### CPU Implementation

```cpp
#include "Image.h"
#include "hdr.h"
#include "basicImageManipulation.h"
#include <vector>

// Load exposure sequence
vector<Image> imSeq;
imSeq.push_back(gamma_code(Image("../Input/boston-1.png"), 1.0/2.2));
imSeq.push_back(gamma_code(Image("../Input/boston-2.png"), 1.0/2.2));
imSeq.push_back(gamma_code(Image("../Input/boston-3.png"), 1.0/2.2));

// Create HDR image (CPU)
Image hdr = makeHDR(imSeq);

// Tone map with bilateral filtering
// Parameters: targetBase=100, detailAmp=3, sigmaRange=0.1
Image tm = toneMap(hdr, 100, 3, 0.1);

// Convert to sRGB and save
tm = gamma_code(tm, 2.2);
tm.write("./Output/boston-hdr-tonemapped.png");
```

### GPU Implementation

```cpp
#include "Image.h"
#include "hdr.h"
#include "basicImageManipulation.h"
#include <vector>

// Load exposure sequence (same as CPU)
vector<Image> imSeq;
imSeq.push_back(gamma_code(Image("../Input/boston-1.png"), 1.0/2.2));
imSeq.push_back(gamma_code(Image("../Input/boston-2.png"), 1.0/2.2));
imSeq.push_back(gamma_code(Image("../Input/boston-3.png"), 1.0/2.2));

// Create HDR image (GPU)
Image hdr = makeHdrGpuBasic(imSeq);

// Tone map with GPU bilateral filtering
Image tm = toneMap(hdr, 100, 3, 0.1);

// Convert to sRGB and save
tm = gamma_code(tm, 2.2);
tm.write("./Output/boston-hdr-tonemapped.png");
```

## Performance and Timing Output

Both CPU and GPU implementations produce detailed timing output for performance comparison.

### Example Output Format

```
========== testToneMapping_ante2 ==========

=== makeHDR ===
  Total weight calculations (GPU): 40.5468 ms
  Total factor calculations: 31.1561 ms
  Merging (GPU): 71.5817 ms
Total makeHDR: 143.285 ms

=== toneMap (bilateral) ===
  lumiChromi: 24.703 ms
  log10Image: 8.84294 ms
  sigmaDomain calc: 0.00016 ms
  bilateral (GPU): 32969.4 ms
  detail computation: 3.78256 ms
  scale factor k: 3.98365 ms
  new_log_lumi: 13.9948 ms
  exp10Image: 7.4282 ms
  lumiChromi2rgb: 16.5523 ms
Total toneMap (bilateral): 33048.7 ms

========================================
Overall timing for testToneMapping_ante2:
  makeHDR: 149.642 ms
  toneMap: 33049 ms
========================================
```

## Build Artifacts and Output Files

### Generated During Build

**CPU and GPU:**
- `_build/*.o` - Object files
- `main` - Executable
- `Output/` - Directory for output images

## Key Implementation Details

### Image Class
- Multi-dimensional support (1D, 2D, or 3D: width × height × channels)
- Internal storage: `std::vector<float>` with values typically in [0.0, 1.0]
- Overloaded operators for element-wise operations (+, -, *, /)
- Safe accessors with optional clamping
- PNG I/O via lodepng library

### CUDA Utilities (`cuda_utils.h` - GPU only)
- `CUDA_CHECK()` - Macro for CUDA error checking
- `CUDA_CHECK_KERNEL()` - Macro for kernel launch error checking
- `CudaTimer` class - GPU kernel timing using CUDA events
- `TimePoint`, `now()`, `elapsed_ms()` - Host-side chrono timing
- Device functions: `im_smartAccessClamp()`, `im_smartSetterClamp()`, etc.

### GPU Kernels
- `computeWeightGpu()` - Weight map generation on GPU
- `hdrMergeGpu()` - HDR image merging on GPU
- `bilateral_basic()` - Edge-preserving bilateral filtering on GPU

**Thread Configuration:**
- 16×16 thread blocks (256 threads per block)
- 2D grid covering entire image dimensions

## Acknowledgments
- PNG I/O: lodepng library
- Target GPU: Nvidia Tesla T4 (AWS g4dn instance)
