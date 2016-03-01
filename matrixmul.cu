extern "C"{

  #include "matrixmul.h"

  #define TILE_WIDTH 32

  __global__ void MatrixMulKernel(MatrixMulOp *mat)
  {
      int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
      int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;

      float Pvalue = 0;
      for(int k=0; k<mat->Width; ++k)
          Pvalue += mat->Md[Row * mat->Width + k] * mat->Nd[k * mat->Width + Col];

      mat->Pd[Row * mat->Width + Col] = Pvalue;
  }

  __global__ void MatrixMulKernelSh(MatrixMulOp *mat)
  {
      __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
      __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

      int bx = blockIdx.x;
      int by = blockIdx.y;
      int tx = threadIdx.x;
      int ty = threadIdx.y;

      int Row = by * TILE_WIDTH + ty;
      int Col = bx * TILE_WIDTH + tx;

      float Pvalue = 0;
      for(int m=0; m<mat->Width/TILE_WIDTH; ++m){
          Mds[ty][tx] = mat->Md[Row*mat->Width + (m*TILE_WIDTH + tx)];
          Nds[ty][tx] = mat->Nd[(m*TILE_WIDTH + ty)*mat->Width + Col];
          __syncthreads();

          for(int k=0; k<TILE_WIDTH; ++k)
              Pvalue += Mds[ty][k] * Nds[k][tx];
          __syncthreads();
      }

      mat->Pd[Row*mat->Width + Col] = Pvalue;
  }

}
