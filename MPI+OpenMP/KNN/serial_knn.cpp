#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<algorithm>
#include<time.h>
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

int indexOfMax(int *a, int len){ // return the maximum value in the array
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

void knn(float **X_train, int *y_train, float **X_test, int *y_test, int k){
	int len_train = (int)(N-N*TEST_SIZE);  // the length of train set
	int len_test = (int)(N*TEST_SIZE);     // the length of test set
    int predict[len_test];
    clock_t start_cpu = clock();

    for(int i=0, predict_index=0; i<len_test; i++){
        float distances[len_train];
        for(int j=0; j<len_train; j++)
            distances[j] = distance(X_test[i], X_train[j]);
        int sortDistances[len_train];
        argsort(distances, len_train, sortDistances);
        int classCount[N]={0};
        for(int j=0; j<k; j++){
            int label = y_train[sortDistances[j]];
            classCount[label]++;
        }
        predict[predict_index] = indexOfMax(classCount, N);
        predict_index++;
    }
    
    printf("Serial time:%fs\n",(double)(clock()-start_cpu)/CLOCKS_PER_SEC);
    //printf("y_pred: ");
    //print(predict, len_test);
    //printf("y_test: ");
    //print(y_test, len_test);
    printf("accuracy = %.2f%%\n", 100*accuracy(predict, y_test, len_test));
}

float **Allocate2DArray(int nRows, int nCols){ //allocate 2d array
    float **array = new float*[nRows];
    for(int i = 0; i < nRows; ++i)
    {
        array[i] = new float[nCols];
    }
    return array;
}

void Free2DArray(float** Array, int nRows, int nCols){ //free 2d array

	for(int i=0; i<nRows; i++) {
  		delete[] Array[i];
 	}
 	delete[] Array;
}

int main(){

	int test_len = N * TEST_SIZE;
    int train_len = N - test_len;

    float **X_train = Allocate2DArray(train_len, FEATURES_NUM);
    float **X_test = Allocate2DArray(test_len, FEATURES_NUM);
    int *y_train = new int[train_len];
    int *y_test = new int[test_len];

    // the X_train is one-dimensional array, it is flattened.

    file_read();

    train_test_split(X_train, y_train, X_test, y_test, features, labels, TEST_SIZE, RANDOM_STATE); //split data to train and test

    knn(X_train, y_train, X_test, y_test, K);

    //Free2DArray(X_train, train_len, FEATURES_NUM);
    //Free2DArray(X_test, test_len, FEATURES_NUM);

    return 0;
}

