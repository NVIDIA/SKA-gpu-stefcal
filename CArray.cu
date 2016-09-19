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

#include "stdio.h"
#include <cuComplex.h>
#include "cblas.h"
#include "lapacke_utils.h"
#include "cublas_v2.h"
#include "Defines.h"
#include "CArray.h"
cublasHandle_t g_cublasHandle;
/*********** Some basic functions for CUDA complex **************/
template <class T>
T conj(T in)
{
   return in;
}
cuComplex conj(cuComplex in)
{
   cuComplex ret = in;
   ret.y *= -1;
   return ret;
}
cuDoubleComplex conj(cuDoubleComplex in)
{
   cuDoubleComplex ret = in;
   ret.y *= -1;
   return ret;
}
float2 cont(float2 in)
{
   float2 ret = in;
   ret.y *= -1;
   return ret;
}
template <class T>
void zero(T& inout)
{
   inout = 0;
}
void zero(cuComplex& inout)
{
   inout.x=0;
   inout.y=0;
}
void zero(cuDoubleComplex& inout)
{
   inout.x=0;
   inout.y=0;
}
template <class T>
void accum(T& inout, const T in)
{
   inout += in;
}
void accum(cuComplex& inout, const cuComplex in)
{
   inout.x += in.x;
   inout.y += in.y;
}
void accum(cuDoubleComplex& inout, const cuDoubleComplex in)
{
   inout.x += in.x;
   inout.y += in.y;
}
__host__ __device__ cuComplex operator+(const cuComplex inA, const cuComplex inB) 
{
  cuComplex ret;
  ret.x = inA.x + inB.x;
  ret.y = inA.y + inB.y;
  return ret;
}
__host__ __device__ cuDoubleComplex operator+(const cuDoubleComplex inA, const cuDoubleComplex inB) 
{
  cuDoubleComplex ret;
  ret.x = inA.x + inB.x;
  ret.y = inA.y + inB.y;
  return ret;
}
__host__ __device__ cuComplex operator-(const cuComplex inA, const cuComplex inB) 
{
  cuComplex ret;
  ret.x = inA.x - inB.x;
  ret.y = inA.y - inB.y;
  return ret;
}
__host__ __device__ cuDoubleComplex operator-(const cuDoubleComplex inA, const cuDoubleComplex inB) 
{
  cuDoubleComplex ret;
  ret.x = inA.x - inB.x;
  ret.y = inA.y - inB.y;
  return ret;
}
__host__ __device__ cuComplex operator*(const cuComplex inA, const cuComplex inB) 
{
  cuComplex ret;
  ret.x = inA.x * inB.x - inA.y * inB.y;
  ret.y = inA.x * inB.y + inA.y * inB.x;
  return ret;
}
__host__ __device__ cuDoubleComplex operator*(const cuDoubleComplex inA, const cuDoubleComplex inB) 
{
  cuDoubleComplex ret;
  ret.x = inA.x * inB.x - inA.y - inB.y;
  ret.y = inA.x * inB.y + inA.y * inB.x;
  return ret;
}
__host__ __device__ cuComplex operator/(const cuComplex inA, const float inB) 
{
  cuComplex ret;
  ret.x = inA.x / inB;
  ret.y = inA.y / inB;
  return ret;
}
__host__ __device__ cuDoubleComplex operator/(const cuDoubleComplex inA, const double inB) 
{
  cuDoubleComplex ret;
  ret.x = inA.x / inB;
  ret.y = inA.y / inB;
  return ret;
}
template <class T>
T mult(const T inA, const T inB)
{
   return inA * inB;
}
cuComplex mult(const cuComplex inA, const cuComplex inB) {
   cuComplex ret;
   ret.x = inA.x*inB.x-inA.y*inB.y;
   ret.y = inA.x*inB.y+inA.y*inB.x;
   return ret;
}
/**************  CArray type and supporting classes ******************/
Datatype TransformedRow::operator[](int colnum)
{   
   if (stride==0)
   {
      fprintf(stderr,"ERROR! Looks like an illegal access\n");
   }
#ifdef __DEBUG
   if (colnum>=len)
   {
      fprintf(stderr,"ERROR! column number %d exceeds the dimension of the array\n",colnum);
   }
#endif
   Datatype retval;
   retval=*(data+stride*colnum);
   if (conjugate) return conj(retval);
   else return retval;
}

CArrayBase CArrayBase::trans()
{
   CArrayBase ret;
   ret.data = data;
   ret.rowlen = nrows;
   ret.nrows = rowlen;
   ret.GPUData = GPUData;
   ret.conjugate = conjugate;
   ret.transpose = !transpose;
   return ret;
}
CArrayBase CArrayBase::conj()
{
   CArrayBase ret;
   ret.rowlen = rowlen;
   ret.nrows = nrows;
   ret.data = data;
   ret.GPUData = GPUData;
   ret.conjugate = !conjugate;
   ret.transpose = transpose;
   return ret;
}
CArrayBase CArrayBase::hermitian()
{
   CArrayBase ret;
   ret.rowlen = nrows;
   ret.nrows = rowlen;
   ret.data = data;
   ret.GPUData = GPUData;
   ret.conjugate = !conjugate;
   ret.transpose = !transpose;
   return ret;
}
TransformedRow CArrayBase::operator[](int rownum) 
{
#ifdef __DEBUG
   if (rownum > nrows)
   {
      fprintf(stderr, "ERROR! row number %d exceeds matrix dimension (%d)\n", rownum, nrows);
   }
#endif
   if(GPUData) {
      fprintf(stderr,"ERROR! Access to GPU data via array index [] is not supported\n");
      TransformedRow ret;
      ret.data = data;
      ret.stride=0;
      return ret;
   }
   TransformedRow ret;
   ret.conjugate = conjugate;
   if (transpose) {
      ret.data = data+rownum;
      ret.stride = nrows;
      ret.len = rowlen;
   } else {
      ret.data = data+rowlen*rownum;
      ret.stride = 1;
      ret.len = rowlen;
   }
   return ret;
}
char CArrayBase::lapack_trans()
{
   if (!transpose && !conjugate) return 'N';
   if (transpose && !conjugate) return 'T';
   if (transpose && conjugate) return 'C';
   fprintf(stderr, "ERROR. Don't know how to pass a conjugate, "
                   "but not Hermitian transposed matrix into LAPACK\n");
   return '0';
}
CBLAS_TRANSPOSE CArrayBase::cblas_trans()
{
   if (!transpose && !conjugate) return CblasNoTrans; 
   if (transpose && !conjugate) return CblasTrans;
   if (transpose && conjugate) return CblasConjTrans;
   fprintf(stderr, "ERROR. Don't know how to pass a conjugate, "
                   "but not Hermitian transposed matrix into CBLAS\n");
   return CblasNoTrans;
}
cublasOperation_t CArrayBase::cublas_trans()
{
   if (!transpose && !conjugate) return CUBLAS_OP_N;
   if (transpose && !conjugate) return CUBLAS_OP_T;
   if (transpose && conjugate) return CUBLAS_OP_C;
   fprintf(stderr, "ERROR. Don't know how to pass a conjugate, "
                   "but not Hermitian transposed matrix into CBLAS\n");
   return CUBLAS_OP_N;
}
void CArrayBase::GPUAlloc() 
{
   if (GPUData) return;
   Datatype* tmp;
   cudaMalloc(&tmp, sizeof(Datatype)*nrows*rowlen);
   free(data);
   data=tmp; 
   GPUData = true;
#ifdef __DEBUG
   printf("Malloc on GPU\n");
#endif
}
void CArrayBase::CPUAlloc()
{
   if (!GPUData) return;
   Datatype* tmp =(Datatype*) malloc(sizeof(Datatype)*nrows*rowlen);
   cudaFree(data);
   data=tmp; 
   GPUData = false;
}
void CArrayBase::memcpyD2H()
{
   if (!GPUData) return;
   Datatype* tmp =(Datatype*) malloc(sizeof(Datatype)*nrows*rowlen);
   cudaMemcpy(tmp, data, sizeof(Datatype)*nrows*rowlen, cudaMemcpyDeviceToHost);
   cudaFree(data);
   data=tmp; 
   GPUData = false;
}
void CArrayBase::memcpyH2D()
{
   if (GPUData) return;
   Datatype* tmp;
   cudaMalloc(&tmp, sizeof(Datatype)*nrows*rowlen);
   cudaMemcpy(tmp, data, sizeof(Datatype)*nrows*rowlen, cudaMemcpyHostToDevice);
   free(data);
   data=tmp; 
#ifdef __DEBUG
   printf("Moving to GPU\n");
#endif
   GPUData = true;
}
void CArrayBase::copy(CArrayBase in)
{
   if(GPUData) {
      if (!in.GPUData)
      {
         fprintf(stderr, "Can't copy from host to device using .copy()\n"
                         "Both source and dest must be on either host or device\n");
      }
      if (nrows*rowlen != in.nrows*in.rowlen) 
      {
         cudaFree(data); 
         cudaMalloc(&data, sizeof(Datatype)*in.nrows*in.rowlen);
      }
      cudaMemcpy(data, in.data, sizeof(Datatype)*in.rowlen*in.nrows, cudaMemcpyDeviceToDevice);
      transpose = in.transpose;
      conjugate = in.conjugate;
      nrows = in.nrows;
      rowlen = in.rowlen;
   } else {
      CPUAlloc();
      if (nrows*rowlen != in.nrows*in.rowlen)
      {
         data = (Datatype*)realloc(data, sizeof(Datatype)*in.rowlen*in.nrows);
      }
      for (int r=0; r<in.nrows;r++)
      for (int c=0; c<in.rowlen;c++)
      {
         data[r*in.rowlen+c] = in[r][c];
      }
      transpose = conjugate = false;
      nrows = in.nrows;
      rowlen = in.rowlen;
   }
}
Datatype* CArray::operator[](int rownum) 
{
#ifdef __DEBUG
   if (rownum > nrows)
   {
      fprintf(stderr, "ERROR! row number %d exceeds matrix dimension (%d)\n", rownum, nrows);
   }
#endif
   return data+rowlen*rownum;
}
void CArray::random()
{
    for (int i=0;i<nrows*rowlen;i++) {
        Datatype foo;
        foo.x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        foo.y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        data[i] = foo;
    }
}

void display(const CArrayBase& A)
{
   printf("Size = %d x %d\n", A.nrows, A.rowlen);
   printf("Memory = %p - %p\n", A.data, A.data+A.nrows*A.rowlen-1);
}
CArray operator*(CArrayBase A, CArrayBase B)
{
   if (A.rowlen != B.nrows) {
      fprintf(stderr,"ERROR! Inner dimensions do not match!\n");
      return CArray(0,0);
   }
   CArray *C = new CArray(A.nrows,B.rowlen);
   CBLAS_TRANSPOSE transA = A.cblas_trans();
   CBLAS_TRANSPOSE transB = B.cblas_trans();
   if ( true || 
       A.transpose != CblasNoTrans ||
       B.transpose != CblasNoTrans)
   {
      
      cuComplex alpha;
      alpha.x = 1.0; alpha.y=0.0;
      cuComplex beta;
      beta.x = 0.0; beta.y=0.0;
      if (A.GPUData && B.GPUData)
      {
         C->GPUAlloc();
#ifdef __DEBUG
         printf("cublasCgemm. transA = %d, transB = %d\n", transA, transB);
#endif
         cublasCgemm(g_cublasHandle, A.cublas_trans(), B.cublas_trans(), A.nrows, B.rowlen, B.nrows,
                     &alpha, A.data, A.transpose ? A.nrows : A.rowlen, 
                     B.data, B.transpose ? B.nrows : B.rowlen, &beta, C->data,
                     B.rowlen);
      } else {
#ifdef __DEBUG
         printf("cblas_cgemm. transA = %d, transB = %d\n", transA, transB);
#endif
         cblas_cgemm(CblasRowMajor, transA, transB, A.nrows, B.rowlen, B.nrows, 
                     &alpha, A.data, A.transpose ? A.nrows : A.rowlen, 
                     B.data, B.transpose ? B.nrows : B.rowlen, &beta, C->data, 
                     B.rowlen);
      }
      
   } else {
#ifdef __DEBUG
      printf("Standard matrix multiply\n");
#endif
      for (int k=0;k<A.nrows;k++) 
      for (int j=0;j<B.rowlen;j++)
      {
         zero((*C)[k][j]);
         for (int i=0;i<A.rowlen;i++)
         {   
            Datatype foo = mult(A[k][i],B[i][j]);
            accum((*C)[k][j],foo);
         }
       }
   }
   return *C;        
}
CArray operator+(CArrayBase A, CArrayBase B)
{
   if (A.rowlen != B.rowlen || A.nrows != B.nrows)
   {
      fprintf(stderr,"ERROR! Dimensions do not match\n");
   }
   CArray *C = new CArray(A.nrows, B.rowlen);
   for (int r=0; r<A.nrows;r++) 
   for (int c=0; r<A.rowlen;r++) 
   {
      (*C)[r][c] = A[r][c] + B[r][c];
   }
   return *C;
}
CArray operator-(CArrayBase A, CArrayBase B)
{
   if (A.rowlen != B.rowlen || A.nrows != B.nrows)
   {
      fprintf(stderr,"ERROR! Dimensions do not match\n");
   }
   CArray *C = new CArray(A.nrows, B.rowlen);
   for (int r=0; r<A.nrows;r++) 
   for (int c=0; r<A.rowlen;r++) 
   {
      (*C)[r][c] = A[r][c] - B[r][c];
   }
   return *C;
}
bool operator==(CArrayBase A, CArrayBase B)
{
   bool ret = true;
   for (int r=0; r<A.nrows;r++) 
   for (int c=0; r<A.rowlen;r++) 
   {
      if (fabs(A[r][c].x - B[r][c].x)/A[r][c].x > 0.00001) ret = false;
      if (fabs(A[r][c].y - B[r][c].y)/A[r][c].y > 0.00001) ret = false;
   }
   return ret;
}
bool operator!=(CArrayBase A, CArrayBase B)
{
   return !(A==B);
}
CArray pointOpBinary(CArrayBase A, CArrayBase B, 
                     Datatype(*op)(const Datatype, const Datatype))
{
   CArray *C = new CArray(A.nrows, A.rowlen);
   for (int r=0; r<A.nrows;r++) 
   for (int c=0; r<A.rowlen;r++) 
   {
      (*C)[r][c] = op(A[r][c], B[r][c]);
   }
   return *C;
}

