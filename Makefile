
ifeq (${DEBUG}, 1)
	CFLAGS += -g -G -lineinfo -D__DEBUG
endif
ifeq (${GPU}, 0)
else
	CFLAGS += -D__GPU
endif

all: stefcal

CArray.o: CArray.cu CArray.h Defines.h
	nvcc -o CArray.o -c -arch=sm_35 CArray.cu ${CFLAGS}
stefcal: stefcal.cu CArray.o Defines.h
	nvcc -arch=sm_35 -o stefcal stefcal.cu CArray.o ${CFLAGS} -Xcompiler -fopenmp -L/usr/lib/lapacke -llapacke -L/usr/lib/ -lblas -lcublas
dgetrs_example: dgetrs_example.cc
	g++ -o dgetrs_example dgetrs_example.cc -L/usr/lib/lapacke -llapacke
