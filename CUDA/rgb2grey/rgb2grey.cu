#include<opencv2/opencv.hpp>
#include<math.h>
using namespace cv;
//r*0.299 + g*0.587 + b*0.114


void byCPU(Mat src){
	Mat grey = Mat(src.rows, src.cols, CV_8UC1);
	clock_t start, end;
	start = clock();
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			uchar b = src.at<Vec3b>(i, j)[0];//注意rgb的存放顺序
			uchar g = src.at<Vec3b>(i, j)[1];
			uchar r = src.at<Vec3b>(i, j)[2];
			grey.at<uchar>(i,j)= r * 0.299 + g * 0.587 + b * 0.114;//公式
		}
	}
	end = clock();
    printf("CPU's time:%fs\n",(double)(end-start)/CLOCKS_PER_SEC);

	imwrite("./3_grey_cpu.jpg",grey);
}

__global__ void rgb2grey(int *bgr_cuda, float *grey_cuda, clock_t *time){
	clock_t start = clock();
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	grey_cuda[x] = bgr_cuda[3*x] * 0.114 + bgr_cuda[3*x+1] * 0.587 + bgr_cuda[3*x+2] * 0.299;
	*time = clock() - start;
}
int main()
{
    //像素操作
    Mat src;
    src = imread("./3.jpg");
    if(src.empty())
    {
        printf("can not load image \n");
        return -1;
    }
    byCPU(src);

    int *bgr;//用一维数组存放图片每个像素的bgr值
    float *grey;

    bgr = (int*)malloc(sizeof(int)*3*src.rows*src.cols);
    grey = (float*)malloc(sizeof(float)*src.rows*src.cols);

    for(int row = 0, k = 0; row < src.rows; row++)
    {
        for(int col = 0; col < src.cols; col++)
        {
            bgr[k++] = src.at<Vec3b>(row, col)[0];
            bgr[k++] = src.at<Vec3b>(row, col)[1];
            bgr[k++] = src.at<Vec3b>(row, col)[2];
        }
    }
    printf("bgr[0]:%d\n",bgr[0]);
    int *bgr_cuda;
    float *grey_cuda;
    clock_t* time;
    clock_t time_used;
    cudaMalloc((void**)&bgr_cuda, sizeof(int)*3*src.rows*src.cols);
    cudaMalloc((void**)&grey_cuda, sizeof(float)*src.rows*src.cols);
    cudaMalloc((void**)&time, sizeof(clock_t));
    cudaMemcpy(bgr_cuda, bgr, sizeof(int)*3*src.rows*src.cols, cudaMemcpyHostToDevice);
	int threads_num = 1024;
    int blocks_num = src.rows * src.cols / threads_num + 1;
    printf("number of blocks is %d\n",blocks_num);
    printf("number of threads is %d\n",threads_num);
	rgb2grey<<<blocks_num,threads_num>>>(bgr_cuda, grey_cuda, time);

	cudaMemcpy(grey, grey_cuda, sizeof(float)*src.rows*src.cols,cudaMemcpyDeviceToHost);
	cudaMemcpy(&time_used, time, sizeof(clock_t), cudaMemcpyDeviceToHost);
    //打印GPU计算的时间
    printf("GPU's time:%fs\n",(double)(time_used)/CLOCKS_PER_SEC);

    Mat grey_Mat = Mat(src.rows, src.cols, CV_8UC1);

	for (int i = 0, k=0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			grey_Mat.at<uchar>(i,j)= grey[k++];//公式
		}
	}
    imwrite("./3_grey_gpu.jpg",grey_Mat);

}