# 串行  
g++ normal_strassen_matmul.cpp -o sernormal_strassen_matmulial_knn  
./normal_strassen_matmul  

# OpenMP  
g++ omp_strassen.cpp -o omp_strassen -fopenmp  
./omp_strassen  

# MPI  
mpic++ mpi_strassen.cpp -o mpi_strassen  
mpirun -np 5 ./mpi_strassen  

# MPI+OpenMP
mpic++ mpi_omp_strassen.cpp -o mpi_omp_strassen -fopenmp  
mpirun -np 5 ./mpi_omp_strassen
