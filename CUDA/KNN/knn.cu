#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<time.h>
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

void train_test_split(float *X_train, int *y_train, float *X_test, int *y_test, float X[N][FEATURES_NUM], int *y, float test_size, int random_state){//partition data set
    int index[N];
    genRandArray(index, random_state);
    int test_len = N * test_size;
    int train_len = N - test_len;
    for(int i=0,k=0; i<train_len; i++)
        for(int j=0; j<FEATURES_NUM; j++,k++)
            X_train[k] = X[index[i]][j];
    for(int i=train_len,k=0; i<N; i++)
        for(int j=0; j<FEATURES_NUM; j++,k++)
            X_test[k] = X[index[i]][j];
    for(int i=0; i<train_len; i++)
        y_train[i] = y[index[i]];
    for(int i=train_len,k=0; i<N; i++,k++)
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

__global__ void calDisByGPU(float *X_train_cuda, float *X_test_cuda, float *distance_cuda, int n, float test_size, int features_num, clock_t *time){
	clock_t start = clock();
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x<3600){
		int num_train = n - n * test_size;
		float distance = 0.0;
		for(int i=0; i<features_num; i++)
			distance += pow((X_test_cuda[x/num_train*features_num+i]-X_train_cuda[x%num_train*features_num+i]),2);
		distance_cuda[x] = sqrt(distance);
	}
	*time = clock() - start;
}

void knnByGPU(float *X_train, int *y_train, float *X_test, int *y_test, int k){
	
	int num_train = (int)(N-N*TEST_SIZE);
	int num_test = (int)(N*TEST_SIZE);
	int len_X_train = num_train * FEATURES_NUM;
	int len_X_test = num_test * FEATURES_NUM;
	int predict[num_test];
	float *distances;
	float *distance_cuda;
	float *X_train_cuda;
	float *X_test_cuda;
	clock_t* time;
	clock_t time_used;
	cudaMalloc((void**)&time, sizeof(clock_t));
	distances = (float*)malloc(sizeof(float)*num_train*num_test);
	cudaMalloc((void**)&distance_cuda, sizeof(float)*num_train*num_test);
	cudaMalloc((void**)&X_train_cuda, sizeof(float)*len_X_train);
	cudaMalloc((void**)&X_test_cuda, sizeof(float)*len_X_test);

	cudaMemcpy(X_train_cuda, X_train, sizeof(float)*len_X_train, cudaMemcpyHostToDevice);
    cudaMemcpy(X_test_cuda, X_test, sizeof(float)*len_X_test, cudaMemcpyHostToDevice);

    int blocks_num = (N * TEST_SIZE) * (N - N * TEST_SIZE) / THREAD_NUM + 1;
	printf("number of blocks is %d\n",blocks_num);
	printf("number of threads is %d\n",THREAD_NUM);       
	calDisByGPU<<<blocks_num, THREAD_NUM>>>(X_train_cuda, X_test_cuda, distance_cuda, N, TEST_SIZE, FEATURES_NUM, time);
	cudaMemcpy(distances, distance_cuda, sizeof(float)*num_train*num_test, cudaMemcpyDeviceToHost);
	cudaMemcpy(&time_used, time, sizeof(clock_t), cudaMemcpyDeviceToHost);
	        
	        

    for(int i=0; i<num_test; i++){  //(N*TEST_SIZE)*FEATURES_NUM=len(X_test)
    	float *distance = distances + i * num_train;
    	    	
        int sortDistances[num_train];
        argsort(distance, num_train, sortDistances);
        int classCount[150]={0};

        for(int j=0; j<k; j++){
            int label = y_train[sortDistances[j]];
            classCount[label]++;
        }
        predict[i] = indexOfMax(classCount, 150);
    }
    printf("gpu:\n");
    printf("GPU's time:%fs\n",(double)(time_used)/CLOCKS_PER_SEC);
    printf("y_pred: ");
    print(predict, num_test);
    printf("y_test: ");
    print(y_test, num_test);
    printf("accuracy = %.2f%%\n", 100*accuracy(predict, y_test, num_test));

}
void knn(float *X_train, int *y_train, float *X_test, int *y_test, int k){
	int num_train = (int)(N-N*TEST_SIZE);
	int num_test = (int)(N*TEST_SIZE);
	int len_X_train = num_train * FEATURES_NUM;

	printf("cpu:\n");
    int predict[(int)(N*TEST_SIZE)];
	clock_t start_cpu, end_cpu;
    start_cpu = clock();
    for(int t=0; t<1e2; t++)
	    for(int i=0, predict_index=0; i<len_X_train; i+=4){  //(N*TEST_SIZE)*FEATURES_NUM=len(X_test)
	        float distances[num_train];
	        float *xtest = X_test+i; //select a test data from test set
	        for(int j=0, m=0; j<len_X_train; j+=4,m++){ //(N-N*TEST_SIZE)*FEATURES_NUM=len(X_train)
	            float *xtrain = X_train+j;
	            distances[m] = distance(xtest, xtrain);
	        }
	        int sortDistances[num_train];
	        argsort(distances, num_train, sortDistances);
	        int classCount[150]={0};
	        for(int j=0; j<k; j++){
	            int label = y_train[sortDistances[j]];
	            classCount[label]++;
	        }
	        predict[predict_index] = indexOfMax(classCount, 150);
	        predict_index++;
	    }

    end_cpu = clock();
    printf("CPU's time:%fs\n",(double)(end_cpu-start_cpu)/CLOCKS_PER_SEC/1e2);

    printf("y_pred: ");
    print(predict, num_test);
    printf("y_test: ");
    print(y_test, num_test);
    printf("accuracy = %.2f%%\n", 100*accuracy(predict, y_test, num_test));
}
int main(){
    float *X_train, *X_test;
    int *y_train, *y_test;
    int test_len = N * TEST_SIZE;
    int train_len = N - test_len;

    X_train = (float*)malloc(sizeof(float)*train_len*FEATURES_NUM);
    y_train = (int*)malloc(sizeof(int)*train_len);
    X_test  = (float*)malloc(sizeof(float)*test_len*FEATURES_NUM);
    y_test  = (int*)malloc(sizeof(int)*test_len);
    file_read();
    train_test_split(X_train, y_train, X_test, y_test, features, labels, TEST_SIZE, RANDOM_STATE); //split data to train and test
    // printf("X_train:\n");
    // print(X_train, train_len*FEATURES_NUM);
    // printf("X_test:\n");
    // print(X_test, test_len*FEATURES_NUM);
    // printf("y_train:\n");
    // print(y_train, train_len);
    // printf("y_test:\n");
    // print(y_test, test_len);
    knnByGPU(X_train, y_train, X_test, y_test, K);

    

    knn(X_train, y_train, X_test, y_test, K);
    
    return 0;
}

