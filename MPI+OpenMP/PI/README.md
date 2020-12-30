# 串行计算PI的命令  
gcc serial.c -o serial  
./serial  

# MPI的程序运行命令  
mpicc montecarlo_mpi.c -o montecarlo_mpi  
mpirun -np 3 ./montecarlo_mpi  
mpicc trapezoid_mpi.c -o trapezoid_mpi  
mpirun -np 3 ./trapezoid_mpi  

# OpenMP的程序运行命令  
gcc montecarlo_omp.c -o montecarlo_omp -fopenmp  
./montecarlo_omp  
gcc trapezoid_omp.c -o trapezoid_omp -fopenmp  
./trapezoid_omp  

# MPI+OpenMP的程序运行命令  
mpicc montecarlo_mpi_omp.c -o montecarlo_mpi_omp -fopenmp  
./montecarlo_mpi_omp  
mpicc trapezoid_mpi_omp.c -o trapezoid_mpi_omp -fopenmp  
./trapezoid_mpi_omp  

