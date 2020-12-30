#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

int main(int argc, char* argv[]) {
    //###################
    // montecarlo
    long long int total = atoi(argv[1]); 
    int thread_nums = 4;
    long long count = 0;
    double x, y;
    long long int i;
    unsigned seed = time(NULL);
    clock_t begin = clock();
    for(i = 0; i < total; i++) {
        x = (double)rand_r(&seed) / RAND_MAX;
        y = (double)rand_r(&seed) / RAND_MAX;
        if(x*x + y*y <= 1) {
            count++;
        }
    }
    clock_t end = clock();
    double pi = 4 * (double)count / total;
    printf("montecarlo: total=%lld, thread_nums=%2d;    Time=%fs;    PI=%0.4f\n",total, thread_nums, (double)(end-begin)/CLOCKS_PER_SEC, pi);

    //####################
    //trapezoid



    double step, sum = 0.0;
    step = 1.0 / (double)total;
    begin = clock();

    for (i = 0; i < total; i++) {
        x = (i+0.5)*step;
        sum +=  4.0 / (1.0 + x * x);
    }
    end = clock();
    pi = step * sum;
    
    printf("trapezoid: total=%lld, thread_nums=%2d;    Time=%fs;    PI=%0.4f\n",total, thread_nums, (double)(end-begin)/CLOCKS_PER_SEC, pi);


}
