/* Copyright 2016, NVIDIA Corporation 

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
  * Neither the name of NVIDIA CORPORATION nor the names of its
    contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */

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
   RM+=blockIdx.y*dim*dim;
   MM+=blockIdx.y*dim*dim;
   g+=blockIdx.y*2*dim;
   glast+=blockIdx.y*2*dim;
   d_num+=blockIdx.y*dim;
   d_denom+=blockIdx.y*dim;
   //snum has the following rows
   //  0. real       part of the numerator
   //  1. imaginary     "
   //  2. denominator
#if BLOCK_WID > 32
   __shared__ float snum[3][BLOCK_HEIGHT][BLOCK_WID];
#endif

   if (blockIdx.x * BLOCK_HEIGHT + threadIdx.y >= dim) return;
   RM += dim * (blockIdx.x * BLOCK_HEIGHT + threadIdx.y);
   MM += dim * (blockIdx.x * BLOCK_HEIGHT + threadIdx.y);
   for(int z = threadIdx.x; z<dim; z+=blockDim.x)
   {
      float thisg_x = __ldg(&glast[z].x);
      float thisg_y = __ldg(&glast[z].y);
      num.x += RM[z].x*thisg_x - RM[z].y*thisg_y;
      num.y += RM[z].x*thisg_y + RM[z].y*thisg_x;
      denom = denom + MM[z]*(thisg_x*thisg_x+thisg_y*thisg_y);
   }
   //Reduce num and denom over the threadblock
#if BLOCK_WID > 32
   snum[0][threadIdx.y][threadIdx.x] = num.x;
   snum[1][threadIdx.y][threadIdx.x] = num.y;
   snum[2][threadIdx.y][threadIdx.x] = denom;
   
   __syncthreads();
   for(int s=BLOCK_WID/2;s>16;s/=2) {
      if(threadIdx.x < s) snum[0][threadIdx.y][threadIdx.x] += 
                             snum[0][threadIdx.y][threadIdx.x+s]; 
      if(threadIdx.x < s) snum[1][threadIdx.y][threadIdx.x] += 
                             snum[1][threadIdx.y][threadIdx.x+s]; 
      if(threadIdx.x < s) snum[2][threadIdx.y][threadIdx.x] += 
                             snum[2][threadIdx.y][threadIdx.x+s]; 
      __syncthreads();
   }
   if (threadIdx.x < 32) {
      num.x = snum[0][threadIdx.y][threadIdx.x];
      num.y = snum[1][threadIdx.y][threadIdx.x];
      denom = snum[2][threadIdx.y][threadIdx.x];
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
      if (0==threadIdx.x) g[blockIdx.x*BLOCK_HEIGHT+threadIdx.y].x = 
                   (num.x/denom + glast[blockIdx.x*BLOCK_HEIGHT+threadIdx.y].x)/2;
      if (0==threadIdx.x) g[blockIdx.x*BLOCK_HEIGHT+threadIdx.y].y = 
                   (num.y/denom + glast[blockIdx.x*BLOCK_HEIGHT+threadIdx.y].y)/2;
   } else {
      if (0==threadIdx.x) g[blockIdx.x*BLOCK_HEIGHT+threadIdx.y].x = num.x/denom;
      if (0==threadIdx.x) g[blockIdx.x*BLOCK_HEIGHT+threadIdx.y].y = num.y/denom;
   }
}

int main(void) 
{
   cublasCreate(&g_cublasHandle);
   srand(2541617);
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   
   float *MM  = (float*)malloc(sizeof(float)*ELEMENTS*ELEMENTS*N_POL);
   CArray RM(ELEMENTS*N_POL,ELEMENTS);

   CArray sig_in(SOURCES, TIMESTEPS);
   CArray sig_out(ELEMENTS, TIMESTEPS);
   CArray response(ELEMENTS,SOURCES);
   CArray Rhat_save(ELEMENTS,ELEMENTS);
   CArray M_save(ELEMENTS,ELEMENTS);
   for (int q=0;q<N_POL;q++) {
      sig_in.random();
      sig_out.random();
      for (int i=0;i<ELEMENTS;i++)
      for (int j=0;j<TIMESTEPS;j++)
         sig_out[i][j] = sig_out[i][j]/sqrt(1.0*TIMESTEPS);
      response.random();
      CArray Rhat = sig_out*sig_out.hermitian();

      CArray M = response*sig_in*sig_in.hermitian()*response.hermitian();

#if 1
      for (int i=0;i<ELEMENTS;i++)
      for (int j=0;j<ELEMENTS;j++)
      {
         //TODO MM should be real
         MM[i*ELEMENTS+j+q*ELEMENTS*ELEMENTS] = (M[i][j]*conj(M[i][j])).x;  
         RM[i+ELEMENTS*q][j] = conj(Rhat[j][i])*M[i][j];
         if (N_POL-1 == q) {
            Rhat_save[j][i] = Rhat[j][i];
            M_save[i][j] = M[i][j];
         }
      }
#endif
      
   }
   
   Datatype g2[2*N_POL][ELEMENTS];
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
      Datatype tmp = Rhat_save[i][j] - g[i]*conj(g[j])*M_save[i][j];
      //Datatype tmp = Rhat[i][j];
      foo  = foo + tmp*conj(tmp);
   }
   printf("first resid = %e, %e\n", foo.x, foo.y);
#ifdef __GPU
   RM.memcpyH2D();
   Datatype *d_g2, *d_g, *d_glast, *d_num;
   float *d_denom, *d_MM;
   cudaMalloc(&d_g2, sizeof(Datatype)*ELEMENTS*2*N_POL);
   cudaMalloc(&d_num, sizeof(Datatype)*ELEMENTS*N_POL);
   cudaMalloc(&d_denom, sizeof(float)*ELEMENTS*N_POL);
   cudaMalloc(&d_MM, sizeof(float)*ELEMENTS*ELEMENTS*N_POL);
   cudaMemcpy(d_g2, g, sizeof(Datatype)*ELEMENTS*2*N_POL, cudaMemcpyHostToDevice);
   cudaMemcpy(d_MM, MM, sizeof(float)*ELEMENTS*ELEMENTS*N_POL, cudaMemcpyHostToDevice);
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
      stefcal_kernel<<<dim3(ELEMENTS/BLOCK_HEIGHT,N_POL), 
                       dim3(BLOCK_WID,BLOCK_HEIGHT)>>>
                                   (RM.data, d_MM, d_g, d_glast, 
                                    d_num, d_denom, ELEMENTS, 1==iter%2);
      cudaDeviceSynchronize();
      cudaError_t err = cudaGetLastError();
      if (err) fprintf(stderr, "Error in call to stefcal_kernel:\n"
                                              "%s\n", cudaGetErrorString(err));
#else
      Datatype *tmp = g;
      g = glast;
      glast = tmp;
      float num_x, num_y;
      float denom;
      Datatype *RM_p = RM.data;
      float *MM_p = MM;
      for (int j = 0; j < ELEMENTS; j++) 
      {
         num_x = num_y = 0.0;
         denom = 0.0;
         #pragma omp parallel for reduction(+:num_x) reduction(+:num_y) reduction(+:denom)
         for (int i = 0; i < ELEMENTS; i++) 
         {
            num_x += (RM_p[i].x*glast[i].x-RM_p[i].y*glast[i].y);
            num_y += (RM_p[i].y*glast[i].x+RM_p[i].x*glast[i].y);
            denom += MM_p[i]*(glast[i].x*glast[i].x+glast[i].y*glast[i].y);
         }
         g[j].x = num_x/denom;
         g[j].y = num_y/denom;
         RM_p += ELEMENTS;
         MM_p += ELEMENTS;
      }
      //TODO stopping criteria
#ifdef __DEBUG
      float resid = 0.0;
      for (int j = 0; j < ELEMENTS; j++) 
      for (int i = 0; i < ELEMENTS; i++) 
      {
         Datatype tmp = Rhat_save[i][j] - g[i]*conj(g[j])*M_save[i][j];
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
      Datatype tmp = Rhat_save[i][j] - g[i]*conj(g[j])*M_save[i][j];
      resid  = resid + (tmp*conj(tmp)).x;
   }
   printf("Gains:\n");
   for (int i=0; i < ELEMENTS; i++)
      printf("    %f, %f\n", g[i].x, g[i].y);
   printf("Final resid = %e\n", resid);
   printf("Compute time: %f ms\n", elapsed);
   
   free(MM);
   cublasDestroy(g_cublasHandle);
   printf("Destroyded cublasHandle\n"); fflush(0);
}
