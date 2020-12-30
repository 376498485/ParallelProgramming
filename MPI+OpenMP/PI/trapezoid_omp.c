#include<stdio.h>
#include<omp.h> 
#include<time.h>

int main(int argc, char* argv[]) 
{
	long long int total = atoi(argv[1]);
	long long int i;
	int thread_nums = 4;
	double x, pi, step, sum = 0.0;
	step = 1.0 / (double)total;
	omp_set_num_threads(thread_nums);
	double begin = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum) private(x) 
	for (i = 0; i < total; i++) {
		x = (i+0.5)*step;
		sum = sum + 4.0 / (1.0 + x * x);
	}
	double end = omp_get_wtime();
	pi = step * sum;
	printf("total=%lld,thread_nums=%2d,Time=%fs,PI=%0.4f\n",total, thread_nums, end-begin, pi);
	return 0;
}