# some variables
BUILD_DIR  :=_build
EXECUTABLE := main
OUTPUT := Output

#contain a list of test
TESTS = $(patsubst %.cpp, %, $(wildcard test*.cpp))

# CUDA compiler (nvcc) and host C++ compiler
NVCC := nvcc
CXX := g++

# Target GPU: Nvidia T4 (Turing architecture, compute capability 7.5)
CUDA_ARCH := sm_75

# CUDA compilation flags
NVCC_FLAGS := -arch=$(CUDA_ARCH) -std=c++11 -g -G -Xcompiler -Wall

# Host C++ compiler flags (for .cpp files compiled with g++)
CXX_FLAGS := -Wall -g3 -ggdb -std=c++11

# ------------------------------------------------------------------------------

# 'make' or 'make all' runs the default target 'all' which requires that
# the EXECUTABLE must be available; if the EXECUTABLE is already available
# then nothing happens. see the rules for EXECUTABLE

all: $(EXECUTABLE)

# ------------------------------------------------------------------------------

# 'make run' runs the target 'run' which calls for running the executable
# if the executable is not available, then the rules for creating it are run
# this is a superset of the target 'all' because it creates the executable
# and also runs it

run: $(EXECUTABLE)
	./$(EXECUTABLE)

# ------------------------------------------------------------------------------

# 'make clean' runs the target 'clean' which in turn removes the
# intermediate .o files and the executable

clean:
	rm -rf $(BUILD_DIR) $(EXECUTABLE) $(OUTPUT) $(TESTS)

# ------------------------------------------------------------------------------

# make tests' compiles the test cases
tests: $(TESTS)

# ------------------------------------------------------------------------------

# rule for creating the executable: link .o files using nvcc
# nvcc can link both CUDA and C++ object files
# If .o files are not available, then the rules for creating .o files are run.

$(EXECUTABLE): $(BUILD_DIR)/main.o $(BUILD_DIR)/hdr.o $(BUILD_DIR)/basicImageManipulation.o $(BUILD_DIR)/filtering.o $(BUILD_DIR)/filtering_cu.o $(BUILD_DIR)/Image.o $(BUILD_DIR)/lodepng.o
	$(NVCC) $(NVCC_FLAGS) $(BUILD_DIR)/main.o $(BUILD_DIR)/hdr.o $(BUILD_DIR)/basicImageManipulation.o $(BUILD_DIR)/filtering.o $(BUILD_DIR)/filtering_cu.o $(BUILD_DIR)/Image.o $(BUILD_DIR)/lodepng.o -o $(EXECUTABLE)
	mkdir -p $(OUTPUT)

# ------------------------------------------------------------------------------

# rules for creating the .o files:
# nvcc can compile both .cpp and .cu files
# For .cpp files, nvcc will use the host compiler (g++) internally
# For .cu files, nvcc will compile CUDA kernels

$(BUILD_DIR)/%.o: %.cpp
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cu
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Explicit rule for filtering.cu with dependency on cuda_utils.h
$(BUILD_DIR)/filtering_cu.o: filtering.cu cuda_utils.h filtering.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c filtering.cu -o $@

# ------------------------------------------------------------------------------

# rule for creating the tests: link test .o files using nvcc
# It creates each test .o file and links it to all the other ones.

$(TESTS) : $(BUILD_DIR)/hdr.o $(BUILD_DIR)/basicImageManipulation.o $(BUILD_DIR)/filtering.o $(BUILD_DIR)/filtering_cu.o $(BUILD_DIR)/Image.o $(BUILD_DIR)/lodepng.o
	$(NVCC) $(NVCC_FLAGS) -c $(@).cpp -o $(BUILD_DIR)/$@.o
	$(NVCC) $(NVCC_FLAGS) $(BUILD_DIR)/$(@).o $(BUILD_DIR)/hdr.o $(BUILD_DIR)/filtering.o $(BUILD_DIR)/filtering_cu.o $(BUILD_DIR)/basicImageManipulation.o $(BUILD_DIR)/Image.o $(BUILD_DIR)/lodepng.o -o $(@)
	mkdir -p $(OUTPUT)
