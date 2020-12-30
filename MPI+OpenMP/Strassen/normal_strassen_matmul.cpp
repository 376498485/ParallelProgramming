#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
using namespace std;
//Multiplies Two Matrices recrusively.
int Strassen(int n, int** MatrixA, int ** MatrixB, int ** MatrixC);
//Adds two Matrices, and places the result in another Matrix
int ADD(int** MatrixA, int** MatrixB, int** MatrixResult, int length );
//subtracts two Matrices , and places  the result in another Matrix
int SUB(int** MatrixA, int** MatrixB, int** MatrixResult, int length );
//Multiplies two matrices in conventional way.
int MUL(int** MatrixA, int** MatrixB, int** MatrixResult, int length );
//Fills Matrices with random numbers.
void FillMatrix( int** matrix1, int** matrix2, int length);
//prints the Matrix content.
void PrintMatrix( int **MatrixA, int MatrixSize );
 
int main()
{
	int MatrixSizes[6] = {64, 128, 256, 512, 1024, 2048};
	for(int m=0; m<6; m++){
		int MatrixSize = MatrixSizes[m];
	    int** MatrixA;
	    int** MatrixB;
	    int** MatrixC;
	 
	    clock_t start_normal ;
	    clock_t end_normal ;
	 
	    clock_t start_strassen ;
	    clock_t end_strassen ;

	    srand(time(0));
	 
	    MatrixA = new int *[MatrixSize];
	    MatrixB = new int *[MatrixSize];
	    MatrixC = new int *[MatrixSize];
	 
	    for (int i = 0; i < MatrixSize; i++)
	    {
	        MatrixA[i] = new int [MatrixSize];
	        MatrixB[i] = new int [MatrixSize];
	        MatrixC[i] = new int [MatrixSize];
	    }
	 
	    FillMatrix(MatrixA,MatrixB,MatrixSize);
	 
	  //*******************conventional multiplication test
	    start_normal = clock();

		MUL(MatrixA,MatrixB,MatrixC,MatrixSize);

	    end_normal = clock();

		//cout<<"\nMatrix Result... \n";
	    //PrintMatrix(MatrixC,MatrixSize);
	 
	  //*******************Strassen multiplication test
	    start_strassen = clock();

		Strassen( MatrixSize, MatrixA, MatrixB, MatrixC );

		end_strassen = clock();
	 	
	 	double time_normal = (double)(end_normal - start_normal)/CLOCKS_PER_SEC;
	 	double time_strassen = (double)(end_strassen - start_strassen)/CLOCKS_PER_SEC;
	 
		//cout<<"\nMatrix Result... \n";
		//PrintMatrix(MatrixC,MatrixSize);
	 
		printf("Matrix size %d, Normal mode time %lf sec, Strassen mode time %lf sec\n", MatrixSize, time_normal, time_strassen);
	}
}

int Strassen(int N, int **MatrixA, int **MatrixB, int **MatrixC)
{
        int HalfSize = N/2;
        int newSize = N/2;
 
        if ( N <= 64 )//choosing the threshhold is extremely important
        {
            MUL(MatrixA,MatrixB,MatrixC,N);
        }
        else
        {
			int** A11;
			int** A12;
			int** A21;
			int** A22;
 
			int** B11;
			int** B12;
			int** B21;
			int** B22;
 
			int** C11;
			int** C12;
			int** C21;
			int** C22;
 
			int** M1;
			int** M2;
			int** M3;
			int** M4;
			int** M5;
			int** M6;
			int** M7;
			int** AResult;
			int** BResult;
 
            //making a 1 diminsional pointer based array.
			A11 = new int *[newSize];
			A12 = new int *[newSize];
			A21 = new int *[newSize];
			A22 = new int *[newSize];
 
			B11 = new int *[newSize];
			B12 = new int *[newSize];
			B21 = new int *[newSize];
			B22 = new int *[newSize];
 
			C11 = new int *[newSize];
			C12 = new int *[newSize];
			C21 = new int *[newSize];
			C22 = new int *[newSize];
 
			M1 = new int *[newSize];
			M2 = new int *[newSize];
			M3 = new int *[newSize];
			M4 = new int *[newSize];
			M5 = new int *[newSize];
			M6 = new int *[newSize];
			M7 = new int *[newSize];
 
			AResult = new int *[newSize];
			BResult = new int *[newSize];
 
			int newLength = newSize;
 
            //making that 1 diminsional pointer based array , 
            //a 2D pointer based array
			for ( int i = 0; i < newSize; i++)
			{
				A11[i] = new int[newLength];
				A12[i] = new int[newLength];
				A21[i] = new int[newLength];
				A22[i] = new int[newLength];
 
				B11[i] = new int[newLength];
				B12[i] = new int[newLength];
				B21[i] = new int[newLength];
				B22[i] = new int[newLength];
 
				C11[i] = new int[newLength];
				C12[i] = new int[newLength];
				C21[i] = new int[newLength];
				C22[i] = new int[newLength];
 
				M1[i] = new int[newLength];
				M2[i] = new int[newLength];
				M3[i] = new int[newLength];
				M4[i] = new int[newLength];
				M5[i] = new int[newLength];
				M6[i] = new int[newLength];
				M7[i] = new int[newLength];
 
				AResult[i] = new int[newLength];
				BResult[i] = new int[newLength];
 
 
			}
			//splitting input Matrixes, into 4 submatrices each.
            for (int i = 0; i < N / 2; i++)
            {
                for (int j = 0; j < N / 2; j++)
                {
                    A11[i][j] = MatrixA[i][j];
                    A12[i][j] = MatrixA[i][j + N / 2];
                    A21[i][j] = MatrixA[i + N / 2][j];
                    A22[i][j] = MatrixA[i + N / 2][j + N / 2];
 
                    B11[i][j] = MatrixB[i][j];
                    B12[i][j] = MatrixB[i][j + N / 2];
                    B21[i][j] = MatrixB[i + N / 2][j];
                    B22[i][j] = MatrixB[i + N / 2][j + N / 2];
 
                }
            }
            //here we calculate M1..M7 matrices .
            //M1=(A11+A22)(B11+B22)
            ADD( A11,A22,AResult, HalfSize);
            ADD( B11,B22,BResult, HalfSize);
            Strassen( HalfSize, AResult, BResult, M1 );
 
 
            //M2=(A21+A22)B11
            ADD( A21,A22,AResult, HalfSize);
            Strassen(HalfSize, AResult, B11, M2);
 
            //M3=A11(B12-B22)
            SUB( B12,B22,BResult, HalfSize);
            Strassen(HalfSize, A11, BResult, M3);
 
            //M4=A22(B21-B11)
            SUB( B21, B11, BResult, HalfSize);
            Strassen(HalfSize, A22, BResult, M4);
 
            //M5=(A11+A12)B22
            ADD( A11, A12, AResult, HalfSize);
            Strassen(HalfSize, AResult, B22, M5);
 
            //M6=(A21-A11)(B11+B12)
            SUB( A21, A11, AResult, HalfSize);
            ADD( B11, B12, BResult, HalfSize);
            Strassen( HalfSize, AResult, BResult, M6);
 
            //M7=(A12-A22)(B21+B22)
            SUB(A12, A22, AResult, HalfSize);
            ADD(B21, B22, BResult, HalfSize);
            Strassen(HalfSize, AResult, BResult, M7);
 
            //C11 = M1 + M4 - M5 + M7;
            ADD( M1, M4, AResult, HalfSize);
            SUB( M7, M5, BResult, HalfSize);
            ADD( AResult, BResult, C11, HalfSize);
 
            //C12 = M3 + M5;
            ADD( M3, M5, C12, HalfSize);
 
            //C21 = M2 + M4;
            ADD( M2, M4, C21, HalfSize);
 
            //C22 = M1 + M3 - M2 + M6;
            ADD( M1, M3, AResult, HalfSize);
            SUB( M6, M2, BResult, HalfSize);
            ADD( AResult, BResult, C22, HalfSize);
 
 
            //at this point , we have calculated the c11..c22 matrices, and now we are going to
            //put them together and make a unit matrix which would describe our resulting Matrix.
            for (int i = 0; i < N/2 ; i++)
            {
                for (int j = 0 ; j < N/2 ; j++)
                {
                    MatrixC[i][j] = C11[i][j];
                    MatrixC[i][j + N / 2] = C12[i][j];
                    MatrixC[i + N / 2][j] = C21[i][j];
                    MatrixC[i + N / 2][j + N / 2] = C22[i][j];
                }
            }
 
            // dont forget to free the space we alocated for matrices,
			for (int i = 0; i < newLength; i++)
			{
				delete[] A11[i];delete[] A12[i];delete[] A21[i];
				delete[] A22[i];
 
				delete[] B11[i];delete[] B12[i];delete[] B21[i];
				delete[] B22[i];
				delete[] C11[i];delete[] C12[i];delete[] C21[i];
				delete[] C22[i];
				delete[] M1[i];delete[] M2[i];delete[] M3[i];delete[] M4[i];
				delete[] M5[i];delete[] M6[i];delete[] M7[i];
				delete[] AResult[i];delete[] BResult[i] ;
			}
				delete[] A11;delete[] A12;delete[] A21;delete[] A22;
				delete[] B11;delete[] B12;delete[] B21;delete[] B22;
				delete[] C11;delete[] C12;delete[] C21;delete[] C22;
				delete[] M1;delete[] M2;delete[] M3;delete[] M4;delete[] M5;
				delete[] M6;delete[] M7;
				delete[] AResult;
				delete[] BResult ;
 
 
        }//end of else
 
 
	return 0;
}
 
int ADD(int** MatrixA, int** MatrixB, int** MatrixResult, int MatrixSize )
{
    for ( int i = 0; i < MatrixSize; i++)
    {
        for ( int j = 0; j < MatrixSize; j++)
        {
            MatrixResult[i][j] =  MatrixA[i][j] + MatrixB[i][j];
        }
    }
	return 0;
}
 
int SUB(int** MatrixA, int** MatrixB, int** MatrixResult, int MatrixSize )
{
    for ( int i = 0; i < MatrixSize; i++)
    {
        for ( int j = 0; j < MatrixSize; j++)
        {
            MatrixResult[i][j] =  MatrixA[i][j] - MatrixB[i][j];
        }
    }
	return 0;
}
 
int MUL( int** MatrixA, int** MatrixB, int** MatrixResult, int MatrixSize )
{
    for (int i=0;i<MatrixSize ;i++)
        {
              for (int j=0;j<MatrixSize ;j++)
              {
                   MatrixResult[i][j]=0;
                   for (int k=0;k<MatrixSize ;k++)
                   {
                          MatrixResult[i][j]=MatrixResult[i][j]+MatrixA[i][k]*MatrixB[k][j];
                   }
              }
        }
	return 0;
}
 
void FillMatrix( int** MatrixA, int** MatrixB, int length)
{
    for(int row = 0; row<length; row++)
    {
        for(int column = 0; column<length; column++)
        {
           MatrixB[row][column] = (MatrixA[row][column] = rand() %5);
        }
 
    }
}
void PrintMatrix(int **MatrixA,int MatrixSize)
{
	cout<<endl;
	   for(int row = 0; row<MatrixSize; row++)
		{
			for(int column = 0; column<MatrixSize; column++)
			{
				cout<<MatrixA[row][column]<<"\t";
				if ((column+1)%((MatrixSize)) == 0)
					cout<<endl;
			}
 
		}
	   cout<<endl;
}