# 串行  
g++ serial_knn.cpp -o serial_knn  
./serial_knn  

# OpenMP  
g++ omp_knn.cpp -o omp_knn -fopenmp  
./omp_knn  

# MPI  
mpic++ mpi_knn.cpp -o mpi_knn  
mpirun -np 5 ./mpi_knn  

# MPI+OpenMP
mpic++ omp_mpi_knn.cpp -o omp_mpi_knn -fopenmp  
mpirun -np 5 ./omp_mpi_knn  
