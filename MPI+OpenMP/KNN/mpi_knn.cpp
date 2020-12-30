#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<time.h>
#include<mpi.h>
using namespace std;
#define N 150
#define TEST_SIZE 0.2 // the test set accounted for 20% of the data set
#define RANDOM_STATE 0  // Random seed is 0
#define FEATURES_NUM 4  // number of features
#define K 3
#define THREAD_NUM 1024
float features[N][FEATURES_NUM]; // features
int labels[N]; // labels
void file_read(){ // load data
    FILE *fpRead=fopen("iris.txt","r");
    if(fpRead==NULL)
        return ;
    char label[100];
    int No;
    int i=0;
    char temp[20];
    while(fgets (label, 90, fpRead)!=NULL )
    {
        if(i!=0)
        {
            sscanf(label,"\"%d\" %f %f %f %f \"%s\"",&No, &features[i-1][0], &features[i-1][1], &features[i-1][2], &features[i-1][3], temp);
            int len = strlen(temp);
            temp[len-1]='\0';
            if(strcmp(temp,"setosa")==0)
                labels[i-1] = 1;
            else if(strcmp(temp,"versicolor")==0)
                labels[i-1] = 2;
            else if(strcmp(temp,"virginica")==0)
                labels[i-1] = 3;
        }
        i++;
    }
}

void genRandArray(int *index, int random_state){//generate unique random numbers
    srand(random_state);
    for(int i=0; i<N; i++){
        index[i] = i+1;
    }
    for(int i=0; i<N;i++)
    {
        int w=(int)(rand()%(N-i)+i);
        int t=index[i];
        index[i]=index[w];
        index[w]=t;
    }
}

void train_test_split(float **X_train, int *y_train, float **X_test, int *y_test, float X[N][FEATURES_NUM], int *y, float test_size, int random_state){//partition data set
    int index[N];
    genRandArray(index, random_state);
    int test_len = N * test_size;
    int train_len = N - test_len;
    for(int i=0; i<train_len; i++)
        X_train[i] = X[index[i]];
    for(int k=0,i=train_len; i<N; k++,i++)
        X_test[k] = X[index[i]];
    for(int i=0; i<train_len; i++)
        y_train[i] = y[index[i]];
    for(int k=0,i=train_len; i<N; k++,i++)
        y_test[k] = y[index[i]];
}

float distance(float *a, float *b){ //calculate the distance between two vectors
    float dis=0.0;
    for(int i=0; i<FEATURES_NUM; i++)
        dis += (a[i] -b[i]) * (a[i] -b[i]);
    return sqrt(dis);
}

void argsort(float *distances, int len, int *sortDistances){ //Returns the indexs which would sort an array
    for(int i=0; i<len; i++)
        sortDistances[i] = i;
    for(int i=0; i<len-1; i++){
        for(int j=i+1; j<len; j++){
            if(distances[i] > distances[j]){
                float t = distances[i];
                distances[i] = distances[j];
                distances[j] = t;
                int m = sortDistances[i];
                sortDistances[i] = sortDistances[j];
                sortDistances[j] = m;
            }
        }
    }
}

int indexOfMax(int *a, int len){ // return the max index in the array
    int m=0;
    for(int i=1; i<len; i++){
        if(a[i] > a[m]){
            m = i;
        }
    }
    return m;
}

void print(int *a, int len){ // print the array
    for(int i=0; i<len; i++)
        printf("%d ", a[i]);
    printf("\n");
}

void print(float *a, int len){ // print the array
    for(int i=0; i<len; i++)
        printf("%.2f ", a[i]);
    printf("\n");
}

float accuracy(int *a, int *b, int len){ // calculate accuracy
    int num = 0;
    for(int i=0; i<len; i++){
        if(a[i] == b[i])
            num++;
    }
    return (float)(1.0*num/len);
}

float **Allocate2DArray(int nRows, int nCols){ //allocate 2d array
    float **array = new float*[nRows];
    for(int i = 0; i < nRows; ++i)
    {
        array[i] = new float[nCols];
    }
    return array;
}

int main(int argc, char* argv[]){

    int test_len = N * TEST_SIZE;
    int train_len = N - test_len;
    float **X_train = Allocate2DArray(train_len, FEATURES_NUM);
    float **X_test = Allocate2DArray(test_len, FEATURES_NUM);
    int *y_train = new int[train_len];
    int *y_test = new int[test_len];
    // the X_train is one-dimensional array, it is flattened.
    file_read();
    train_test_split(X_train, y_train, X_test, y_test, features, labels, TEST_SIZE, RANDOM_STATE); //split data to train and test

    //Free2DArray(X_train, train_len, FEATURES_NUM);
    //Free2DArray(X_test, test_len, FEATURES_NUM);
    int rank, num_process;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_process);

    int len_of_predict = test_len/num_process;
    int predict[len_of_predict];
    int *predicts;
    if(rank==0)
        predicts = new int[test_len];

    MPI_Barrier(MPI_COMM_WORLD);
    double begin = MPI_Wtime();

    int low = rank * test_len/num_process;
    int up = low + test_len/num_process;

    for(int i=low, predict_index=0; i<up; i++){
        
        float distances[train_len];
        for(int j=0; j<train_len; j++)
            distances[j] = distance(X_test[i], X_train[j]);
        int sortDistances[train_len];
        argsort(distances, train_len, sortDistances);
        int classCount[4]={0};
        for(int j=0; j<K; j++){
            int label = y_train[sortDistances[j]];
            classCount[label]++;
        }
        predict[predict_index] = indexOfMax(classCount, 4);
        predict_index++;
    }
    MPI_Gather(predict, len_of_predict, MPI_INT, predicts, len_of_predict, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    if(rank==0)
    {
        printf("MPI time:%fs\n",(end-begin));
        printf("accuracy = %.2f%%\n", 100*accuracy(predicts, y_test, test_len));
    }
    

    MPI_Finalize();
}

