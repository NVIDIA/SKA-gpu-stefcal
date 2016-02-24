#ifndef __CARRAY_H
#define __CARRAY_H
#include "stdlib.h"
#include "cuda.h"
#include "Defines.h"

template <class T>
T conj(T in);
cuComplex conj(cuComplex in);
cuDoubleComplex conj(cuDoubleComplex in);
template <class T>
void zero(T& inout);
template <class T>
void accum (T& inout, const T in);
__host__ __device__ cuComplex operator+(const cuComplex inA, const cuComplex inB);
__host__ __device__ cuComplex operator-(const cuComplex inA, const cuComplex inB); 
__host__ __device__ cuComplex operator*(const cuComplex inA, const cuComplex inB); 
__host__ __device__ cuDoubleComplex operator+(const cuDoubleComplex inA, const cuDoubleComplex inB); 
__host__ __device__ cuDoubleComplex operator-(const cuDoubleComplex inA, const cuDoubleComplex inB); 
__host__ __device__ cuDoubleComplex operator*(const cuDoubleComplex inA, const cuDoubleComplex inB); 
__host__ __device__ cuComplex operator/(const cuComplex inA, const float inB); 
__host__ __device__ cuDoubleComplex operator/(const cuDoubleComplex inA, const double inB); 
template <class T>
T mult (const T inA, const T inB);
struct TransformedRow
{
   Datatype* data;
   int stride;
   bool conjugate;
   int len;
   Datatype operator[](int colnum); 
};
struct CArrayBase {
   //The base is borrowing a data reference. The child class CArray
   //has allocation/deallocation
   Datatype* data;
   int rowlen;
   int nrows;
   bool transpose;
   bool conjugate;
   bool GPUData;
   CArrayBase trans();
   CArrayBase conj();
   CArrayBase hermitian();
   TransformedRow operator[](int rownum); 
   char lapack_trans();
   CBLAS_TRANSPOSE cblas_trans();
   cublasOperation_t cublas_trans();
   void GPUAlloc();
   void CPUAlloc();
   void memcpyD2H();
   void memcpyH2D();
   void copy(CArrayBase in);
   CArrayBase()
   {
       transpose = conjugate = GPUData = false;
   }
   ~CArrayBase()
   {
   }
};
struct CArray : public CArrayBase {
   
   Datatype* operator[](int rownum); 
   void random();
   CArray(int rows_in, int cols_in) 
   {
       rowlen=cols_in;
       nrows=rows_in;
       data = (Datatype*)malloc(sizeof(Datatype)*rows_in*cols_in);
       GPUData = false;
#ifdef __DEBUG
       printf("Alloc %d x %d (%p)\n", rows_in, cols_in, data);
#endif
   }
   ~CArray()
   {
#ifdef __DEBUG
       printf("Free %d x %d (%p).\n", nrows, rowlen, data);
       if (GPUData) printf("GPUData.\n");
#endif
       if (GPUData) cudaFree(data);
       else free(data);
   }
};
void display(const CArrayBase& A);
CArray operator*(CArrayBase A, CArrayBase B);
CArray operator+(CArrayBase A, CArrayBase B);
CArray operator-(CArrayBase A, CArrayBase B);
bool operator==(CArrayBase A, CArrayBase B);
bool operator!=(CArrayBase A, CArrayBase B);
CArray pointOpBinary(CArrayBase A, CArrayBase B, 
                     Datatype(*op)(const Datatype, const Datatype));




#endif
