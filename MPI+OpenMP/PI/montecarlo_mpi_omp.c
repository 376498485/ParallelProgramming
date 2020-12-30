#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include <math.h>
#include <time.h>
#define N 1E6
#define d 1E-6

int main(int argc, char *argv[]) {
	int numprocs,rank, sum=0;
	int thread_nums = 4;
	double pi=0.0, begin=0.0, end=0.0, x, y;
	long long int i;
	long long count = 0;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Barrier(MPI_COMM_WORLD);
    begin = MPI_Wtime();
	int low = rank * (N / numprocs);
	int up = low + N / numprocs - 1;
	#pragma omp parallel num_threads(thread_nums)
    {
        unsigned seed = time(NULL);
        #pragma omp for private(x, y) reduction(+:count)
        for(i = low; i < up; i++){
            x = (double)rand_r(&seed) / RAND_MAX;
            y = (double)rand_r(&seed) / RAND_MAX;
            if(x*x + y*y <= 1) 
                count++;
        }
    }
	MPI_Reduce(&count, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    //Caculate and print PI
    if (rank==0)
    {
        pi=4*d*sum;
        printf("numprocs=%2d \t Time=%fs \t PI=%0.4f\n",numprocs, end-begin, pi);
    }
    MPI_Finalize();
}
