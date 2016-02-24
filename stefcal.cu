#include <stdio.h>
#include <cuComplex.h>
#include "cblas.h"
#include "lapacke_utils.h"
#include "cublas_v2.h"
#include "Defines.h"
#include "CArray.h"
extern cublasHandle_t g_cublasHandle;

int MyGEMM(CArrayBase out, CArrayBase inA, CArrayBase inB)
{
   CBLAS_TRANSPOSE transA = inA.cblas_trans();
   CBLAS_TRANSPOSE transB = inB.cblas_trans();
   cuComplex alpha;
   alpha.x = 1.0; alpha.y=0.0;
   cuComplex beta;
   beta.x = 0.0; beta.y=0.0;
   if (inA.GPUData)
   {
      if (!inB.GPUData || !out.GPUData) 
      {
         fprintf(stderr, "ERROR! All matrices passed to MyGEMM must be either GPU or CPU arrays.\n"
                         "Please use mempcy member functions to move arrays\n");
         return 1;
      }
      if (inA.nrows != out.nrows || 
          inB.rowlen != out.rowlen  )
      {
         fprintf(stderr, "ERROR! Output array must have dimensions %d x %d\n", 
                                 inA.nrows, inB.rowlen);
         return 1;
      }
      if (inA.rowlen != inB.nrows)
      {
         fprintf(stderr, "ERROR! Inner dimensions do not match\n");
         return 1;
      }
      cublasCgemm(g_cublasHandle, inA.cublas_trans(), inB.cublas_trans(), inA.nrows, inB.rowlen, inB.nrows,
                  &alpha, inA.data, inA.rowlen, inB.data, inB.rowlen, &beta, out.data,
                  inB.rowlen);
   } else {
      cblas_cgemm(CblasRowMajor, transA, transB, inA.nrows, inB.rowlen, inB.nrows, 
                  &alpha, inA.data, inA.rowlen, inB.data, inB.rowlen, &beta, out.data, 
                  inB.rowlen);
   }
   return 0;
}
__device__ void warp_reduce(double &in, int sz = 16) {
   if (16<sz) sz=16;
   for(int s = sz; s>0;s/=2) {
      in += __shfl_down(in,s);
   }
}
__device__ void warp_reduce(float &in, int sz = 16) {
   if (16<sz) sz=16;
   for(int s = sz; s>0;s/=2) {
      in += __shfl_down(in,s);
   }
}
__device__ void warp_reduce2(float &in, int sz = 32) {
   if (32<sz) sz=32;
   for(int s=1; s<sz; s*=2) {
      in += __shfl_down(in,s);
   }
}
__device__ void warp_reduce2(double &in, int sz = 32) {
   if (32<sz) sz=32;
   for(int s=1; s<sz; s*=2) {
      in += __shfl_down(in,s);
   }
}
__global__ void stefcal_kernel(Datatype* RM, float* MM, Datatype* g, Datatype* glast,
                          Datatype* d_num, float* d_denom, int dim, bool do_avg)
{
   Datatype num; num.x=0.0; num.y=0.0;
   float denom = 0.0;
   //snum has the following rows
   //  0. real       part of the numerator
   //  1. imaginary     "
   //  2. denominator
#if BLOCK_WID > 32
   __shared__ float snum[3][BLOCK_WID];
#endif

   RM += blockIdx.x * dim;
   MM += blockIdx.x * dim;
   for(int z = threadIdx.x; z<dim; z+=blockDim.x)
   {
      //num = num + RM[z]/**glast[z]*/;
      num.x += RM[z].x*glast[z].x - RM[z].y*glast[z].y;
      num.y += RM[z].x*glast[z].y + RM[z].y*glast[z].x;
      denom = denom + MM[z]*(glast[z].x*glast[z].x+glast[z].y*glast[z].y);
   }
   //Reduce num and denom over the threadblock
#if BLOCK_WID > 32
   snum[0][threadIdx.x] = num.x;
   snum[1][threadIdx.x] = num.y;
   snum[2][threadIdx.x] = denom;
   
   __syncthreads();
   for(int s=BLOCK_WID/2;s>16;s/=2) {
      if(threadIdx.x < s) snum[0][threadIdx.x] += snum[0][threadIdx.x+s]; 
      if(threadIdx.x < s) snum[1][threadIdx.x] += snum[1][threadIdx.x+s]; 
      if(threadIdx.x < s) snum[2][threadIdx.x] += snum[2][threadIdx.x+s]; 
      __syncthreads();
   }
   if (threadIdx.x < 32) {
      num.x = snum[0][threadIdx.x];
      num.y = snum[1][threadIdx.x];
      denom = snum[2][threadIdx.x];
   }
#endif
   if (threadIdx.x < 32)
   {
      warp_reduce(num.x);
      warp_reduce(num.y);
      warp_reduce(denom);
   }
   if (do_avg)
   {
      if (0==threadIdx.x) g[blockIdx.x].x = (num.x/denom + glast[blockIdx.x].x)/2;
      if (0==threadIdx.x) g[blockIdx.x].y = (num.y/denom + glast[blockIdx.x].y)/2;
   } else {
      if (0==threadIdx.x) g[blockIdx.x].x = num.x/denom;
      if (0==threadIdx.x) g[blockIdx.x].y = num.y/denom;
   }
}
int main(void) 
{
   cublasCreate(&g_cublasHandle);
   srand(2541617);
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   
   CArray sig_in(SOURCES, TIMESTEPS);
   CArray sig_out(ELEMENTS, TIMESTEPS);
   CArray response(ELEMENTS, SOURCES);
   sig_in.random();
   sig_out.random();
   for (int i=0;i<ELEMENTS;i++)
   for (int j=0;j<TIMESTEPS;j++)
      sig_out[i][j] = sig_out[i][j]/sqrt(1.0*TIMESTEPS);
   response.random();

   CArray Rhat = sig_out*sig_out.hermitian();

   CArray M = response*sig_in*sig_in.hermitian()*response.hermitian();

   float MM[ELEMENTS*ELEMENTS];
   CArray RM(ELEMENTS,ELEMENTS);

#if 1
   for (int i=0;i<ELEMENTS;i++)
   for (int j=0;j<ELEMENTS;j++)
   {
      //TODO MM should be real
      MM[i*ELEMENTS+j] = (M[i][j]*conj(M[i][j])).x;  
      RM[i][j] = conj(Rhat[j][i])*M[i][j];
   }
#endif
   
   Datatype g2[2][ELEMENTS];
   Datatype *g, *glast;
   Datatype foo;
   foo.x=1.0;
   foo.y=0.0;
   g=g2[0];
   glast=g2[1];
   for (int i=0; i<ELEMENTS; i++) g[i] = foo;
   for (int i=0; i<ELEMENTS; i++) glast[i] = foo;
   foo.x = foo.y=0.0;
   for (int j = 0; j < ELEMENTS; j++) 
   for (int i = 0; i < ELEMENTS; i++) 
   {
      Datatype tmp = Rhat[i][j] - g[i]*conj(g[j])*M[i][j];
      //Datatype tmp = Rhat[i][j];
      foo  = foo + tmp*conj(tmp);
   }
   printf("first resid = %e, %e\n", foo.x, foo.y);
#ifdef __GPU
   RM.memcpyH2D();
   Datatype *d_g2, *d_g, *d_glast, *d_num;
   float *d_denom, *d_MM;
   cudaMalloc(&d_g2, sizeof(Datatype)*ELEMENTS*2);
   cudaMalloc(&d_num, sizeof(Datatype)*ELEMENTS);
   cudaMalloc(&d_denom, sizeof(float)*ELEMENTS);
   cudaMalloc(&d_MM, sizeof(float)*ELEMENTS*ELEMENTS);
   cudaMemcpy(d_g2, g2, sizeof(Datatype)*ELEMENTS*2, cudaMemcpyHostToDevice);
   cudaMemcpy(d_MM, MM, sizeof(float)*ELEMENTS*ELEMENTS, cudaMemcpyHostToDevice);
   d_g = d_g2;
   d_glast = d_g2+ELEMENTS;
   
#endif

   cudaEventRecord(start,0);
   for (int iter=0;iter<ITERATIONS;iter++) 
   {
#ifdef __GPU
      Datatype *tmp = d_g;
      d_g = d_glast;
      d_glast = tmp;
      stefcal_kernel<<<ELEMENTS, BLOCK_WID>>>(RM.data, d_MM, d_g, d_glast, 
                                              d_num, d_denom, ELEMENTS, 1==iter%2);
#else
      Datatype *tmp = g;
      g = glast;
      glast = tmp;
      float num_x, num_y;
      float denom;
      for (int j = 0; j < ELEMENTS; j++) 
      {
         num_x = num_y = 0.0;
         denom = 0.0;
         #pragma omp parallel for reduction(+:num_x) reduction(+:num_y) reduction(+:denom)
         for (int i = 0; i < ELEMENTS; i++) 
         {
            num_x += (RM[j][i].x*glast[i].x-RM[j][i].y*glast[i].y);
            num_y += (RM[j][i].y*glast[i].x+RM[j][i].x*glast[i].y);
            denom += MM[j*ELEMENTS+i]*(glast[i].x*glast[i].x+glast[i].y*glast[i].y);
         }
         g[j].x = num_x/denom;
         g[j].y = num_y/denom;
      }
      //TODO stopping criteria
#ifdef __DEBUG
      float resid = 0.0;
      for (int j = 0; j < ELEMENTS; j++) 
      for (int i = 0; i < ELEMENTS; i++) 
      {
         Datatype tmp = Rhat[i][j] - g[i]*conj(g[j])*M[i][j];
         resid  = resid + (tmp*conj(tmp)).x;
      }
      printf("resid = %e\n", resid);
#endif
      if (1==iter%2) {
        for (int i=0;i<ELEMENTS;i++) g[i] = (g2[0][i] + g2[1][i])/2; 
      }
#endif
   }
   cudaDeviceSynchronize();
   cudaEventRecord(stop,0);
   float elapsed;
   cudaEventElapsedTime(&elapsed, start, stop);
#ifdef __GPU
   cudaMemcpy(g, d_g, sizeof(Datatype)*ELEMENTS, cudaMemcpyDeviceToHost);
   cudaFree(d_g2);
   cudaFree(d_num);
   cudaFree(d_denom);
#endif
   float resid = 0.0;
   for (int j = 0; j < ELEMENTS; j++) 
   for (int i = 0; i < ELEMENTS; i++) 
   {
      Datatype tmp = Rhat[i][j] - g[i]*conj(g[j])*M[i][j];
      resid  = resid + (tmp*conj(tmp)).x;
   }
   printf("Gains:\n");
   for (int i=0; i < ELEMENTS; i++)
      printf("    %f, %f\n", g[i].x, g[i].y);
   printf("Final resid = %e\n", resid);
   printf("Compute time: %f ms\n", elapsed);
   
   cublasDestroy(g_cublasHandle);
   printf("Destroyded cublasHandle\n"); fflush(0);
}
