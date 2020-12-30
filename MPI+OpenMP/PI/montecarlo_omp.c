#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

int main(int argc, char* argv[]) {
    long long int total = atoi(argv[1]);  // 默认100万个样本
    int thread_nums = 4;             // 默认24个线程
    long long count = 0;
    double x, y;
    long long int i;

    double begin = omp_get_wtime();
    #pragma omp parallel num_threads(thread_nums)
    {
        unsigned seed = time(NULL);

        #pragma omp for private(x, y) reduction(+:count)
        
        for(i = 0; i < total; i++) {
            x = (double)rand_r(&seed) / RAND_MAX;
            y = (double)rand_r(&seed) / RAND_MAX;
            if(x*x + y*y <= 1) {
                count++;
            }
        }
    }
    double end = omp_get_wtime();
    double pi = 4 * (double)count / total;

    printf("total=%lld, thread_nums=%2d;    Time=%fs;    PI=%0.4f\n",total, thread_nums, end-begin, pi);
    return 0;
}
