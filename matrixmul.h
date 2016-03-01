extern "C"{

  #ifndef MATRIX_MUL_GPU
  #define MATRIX_MUL_GPU

  #define TILE_WIDTH 32

  struct MatrixMulOp {
    int Width, _padding;
    float *Md, *Nd, *Pd;
  };

  __global__ void MatrixMulKernel(MatrixMulOp *mat);
  __global__ void MatrixMulKernelSh(MatrixMulOp *mat);

  #endif
}
