#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <ctime>
using namespace std;
int **Allocate2DArray( int nRows, int nCols) //allocate 2d array
{
    int **array = new int*[nRows];
    for( int i = 0; i < nRows; ++i)
    {
        array[i] = new int[nCols];
    }
    return array;
}

void Free2DArray(int** Array, int nRows, int nCols) //free 2d array
{
	for(int i=0; i<nRows; i++) {
  		delete[] Array[i];
 	}
 	delete[] Array;
}

#define THRESHOLD  32768 /* product size below which matmultleaf is used */  //32^3=32768

void matmultleaf(int ml, int nl, int pl, int **A, int **B, int **C) // Normal matMul by cpu
{
	// A: ml*pl
	// B: pl*nl
	for (int i = 0; i < ml; i++)
		for (int j = 0; j < nl; j++) {
			C[i][j] = 0.0;
			for (int k = 0; k < pl; k++)
				C[i][j] += A[i][k]*B[k][j];
		}
}

void copyQtrMatrix(int **X, int m, int **Y, int mf, int nf) // m表示新矩阵的列数,mf表示原矩阵的开始行数,nf表示原矩阵开始列数
{
	for (int i = 0; i < m; i++)
		X[i] = &Y[mf+i][nf];
}

void AddMatBlocks(int **T, int m, int n, int **X, int **Y)
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			T[i][j] = X[i][j] + Y[i][j];
}

void SubMatBlocks(int **T, int m, int n, int **X, int **Y)
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			T[i][j] = X[i][j] - Y[i][j];
}


void strassenMMult(int ml, int nl, int pl, int **A, int **B, int **C)
{

	if (((float)ml)*((float)nl)*((float)pl) <= THRESHOLD) //使用Normal的方法
		matmultleaf(ml, nl, pl, A, B, C);

	else {
		int m2 = ml/2;
		int n2 = nl/2;
		int p2 = pl/2;

		int **M1 = Allocate2DArray(m2, n2);
		int **M2 = Allocate2DArray(m2, n2);
		int **M3 = Allocate2DArray(m2, n2);
		int **M4 = Allocate2DArray(m2, n2);
		int **M5 = Allocate2DArray(m2, n2);
		int **M6 = Allocate2DArray(m2, n2);
		int **M7 = Allocate2DArray(m2, n2);

		int **wAM1 = Allocate2DArray(m2, p2);
		int **wBM1 = Allocate2DArray(p2, n2);
		int **wAM2 = Allocate2DArray(m2, p2);
		int **wBM3 = Allocate2DArray(p2, n2);
		int **wBM4 = Allocate2DArray(p2, n2);
		int **wAM5 = Allocate2DArray(m2, p2);
		int **wAM6 = Allocate2DArray(m2, p2);
		int **wBM6 = Allocate2DArray(p2, n2);
		int **wAM7 = Allocate2DArray(m2, p2);
		int **wBM7 = Allocate2DArray(p2, n2);

		int **A11 = new int*[m2];
		int **A12 = new int*[m2];
		int **A21 = new int*[m2];
		int **A22 = new int*[m2];

		int **B11 = new int*[p2];
		int **B12 = new int*[p2];
		int **B21 = new int*[p2];
		int **B22 = new int*[p2];

		int **C11 = new int*[m2];
		int **C12 = new int*[m2];
		int **C21 = new int*[m2];
		int **C22 = new int*[m2];

		copyQtrMatrix(A11, m2, A,  0,  0);
		copyQtrMatrix(A12, m2, A,  0, p2);
		copyQtrMatrix(A21, m2, A, m2,  0);
		copyQtrMatrix(A22, m2, A, m2, p2);

		copyQtrMatrix(B11, p2, B,  0,  0);
		copyQtrMatrix(B12, p2, B,  0, n2);
		copyQtrMatrix(B21, p2, B, p2,  0);
		copyQtrMatrix(B22, p2, B, p2, n2);

		copyQtrMatrix(C11, m2, C,  0,  0);
		copyQtrMatrix(C12, m2, C,  0, n2);
		copyQtrMatrix(C21, m2, C, m2,  0);
		copyQtrMatrix(C22, m2, C, m2, n2);

		#pragma omp task
		{
			// M1 = (A11 + A22)*(B11 + B22)
			AddMatBlocks(wAM1, m2, p2, A11, A22);
			AddMatBlocks(wBM1, p2, n2, B11, B22);
			strassenMMult(m2, n2, p2, wAM1, wBM1, M1);
		}

		#pragma omp task
		{
			//M2 = (A21 + A22)*B11
			AddMatBlocks(wAM2, m2, p2, A21, A22);
			strassenMMult(m2, n2, p2, wAM2, B11, M2);
		}

		#pragma omp task
		{
			//M3 = A11*(B12 - B22)
			SubMatBlocks(wBM3, p2, n2, B12, B22);
			strassenMMult(m2, n2, p2, A11, wBM3, M3);
		}

		#pragma omp task
		{
			//M4 = A22*(B21 - B11)
			SubMatBlocks(wBM4, p2, n2, B21, B11);
			strassenMMult(m2, n2, p2, A22, wBM4, M4);
		}

		#pragma omp task
		{
			//M5 = (A11 + A12)*B22
			AddMatBlocks(wAM5, m2, p2, A11, A12);
			strassenMMult(m2, n2, p2, wAM5, B22, M5);
		}

		#pragma omp task
		{
			//M6 = (A21 - A11)*(B11 + B12)
			SubMatBlocks(wAM6, m2, p2, A21, A11);
			AddMatBlocks(wBM6, p2, n2, B11, B12);
			strassenMMult(m2, n2, p2, wAM6, wBM6, M6);
		}

		#pragma omp task
		{
			//M7 = (A12 - A22)*(B21 + B22)
			SubMatBlocks(wAM7, m2, p2, A12, A22);
			AddMatBlocks(wBM7, p2, n2, B21, B22);
			strassenMMult(m2, n2, p2, wAM7, wBM7, M7);
		}
		#pragma omp taskwait
		for (int i = 0; i < m2; i++)
			for (int j = 0; j < n2; j++) {
				C11[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
				C12[i][j] = M3[i][j] + M5[i][j];
				C21[i][j] = M2[i][j] + M4[i][j];
				C22[i][j] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
			}

		Free2DArray(M1, m2, n2);
		Free2DArray(M2, m2, n2);
		Free2DArray(M3, m2, n2);
		Free2DArray(M4, m2, n2);
		Free2DArray(M5, m2, n2);
		Free2DArray(M6, m2, n2);
		Free2DArray(M7, m2, n2);

		Free2DArray(wAM1, m2, p2);
		Free2DArray(wBM1, p2, n2);
		Free2DArray(wAM2, m2, p2);
		Free2DArray(wBM3, p2, n2);
		Free2DArray(wBM4, p2, n2);
		Free2DArray(wAM5, m2, p2);
		Free2DArray(wAM6, m2, p2);
		Free2DArray(wBM6, p2, n2);
		Free2DArray(wAM7, m2, p2);
		Free2DArray(wBM7, p2, n2);
		delete[] A11; delete[] A12; delete[] A21; delete[] A22;
		delete[] B11; delete[] B12; delete[] B21; delete[] B22;
		delete[] C11; delete[] C12; delete[] C21; delete[] C22;
	}
}

void matmultS(int m, int n, int p, int **A, int **B, int **C)
{
	#pragma omp parallel
	{
		#pragma omp single // single表示由一个线程执行
		{
    		strassenMMult(m, n, p, A, B, C);
	  	}
  	}
}

int main(int argc, char* argv[])
{
	int sizes[] = {64, 128, 256, 512, 1024, 2048};
	for(int s=0; s<6; s++){
		int size = sizes[s];
	
		//THRESHOLD = atoi(argv[2]);
		int threads_num = 1024; // threads_num

		double start, end;

		int **A = Allocate2DArray(size, size);
		int **B = Allocate2DArray(size, size);
		int **C = Allocate2DArray(size, size);

		int i, j;


	    for (int i = 0; i < size; ++i)
	        for (int j = 0; j < size; ++j){
		        B[i][j] = 0;
		        A[i][j] = 1;
	        }

	    for (int i = 0; i < size; ++i)
	        B[i][i] = 1;

		omp_set_dynamic(0);     // Explicitly disable dynamic teams
		omp_set_num_threads(threads_num);

		start = omp_get_wtime();

		matmultS(size, size, size, A, B, C);

		end = omp_get_wtime();

		double elapsed_secs = end - start;
		/*checking result if correct*/

	    bool correctness = true;
	    for (int i = 0; i < size; ++i)
	        for (int j = 0; j < size; ++j)
	            if(C[i][j] != A[i][j]) correctness = false;
	    if(correctness)
	        printf("OpenMP - Correct: matrix size = %d, numbers of thread = %d,  time = %lf sec \n",  size, omp_get_max_threads(), elapsed_secs );
	        
	    else
	        printf("OpenMP - Incorrect: matrix size = %d, numbers of thread = %d,  time = %lf sec \n",  size, omp_get_max_threads(), elapsed_secs );
	        
		Free2DArray(A, size, size);
		Free2DArray(B, size, size);
		Free2DArray(C, size, size);

	}
}
