objects = matrixmul.cu

all: $(objects)
	nvcc -arch=sm_30 $(objects) -Xptxas="-v" --cubin
