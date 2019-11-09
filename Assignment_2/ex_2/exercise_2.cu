#include <stdio.h>
#include <time.h>
#define TPB 256
#define ARRAY_SIZE 1000000


__global__ void saxpy_gpu(int n, float a, float* d_x, float* d_y)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		d_y[i] = a * d_x[i] + d_y[i];
}

void saxpy_cpu(int n, float a, float* x, float* y)
{
	for (int i = 0; i < n; i++)
		y[i] = a * x[i] + y[i];

}

int main(void)
{
	const float a = 1.0f;
	float* d_x, * d_y, * x, * y;
	float* result_y;
	clock_t start, end;
	int flag;
	double cpu_time_used = 0.0;

	x = (float*)malloc(ARRAY_SIZE * sizeof(float));
	y = (float*)malloc(ARRAY_SIZE * sizeof(float));
	result_y = (float*)malloc(ARRAY_SIZE * sizeof(float));

	cudaMalloc(&d_x, ARRAY_SIZE * sizeof(float));
	cudaMalloc(&d_y, ARRAY_SIZE * sizeof(float));

	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	cudaMemcpy(d_x, x, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	saxpy_gpu<<<(ARRAY_SIZE + 255) / TPB, TPB >>> (ARRAY_SIZE, a, d_x, d_y);
	cudaDeviceSynchronize();

	printf("Calculation on GPU Done \n");

	cudaMemcpy(result_y, d_y, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	start = clock();
	saxpy_cpu(ARRAY_SIZE, a, x, y);
	end = clock();

	cpu_time_used += (double)(end - start)/ CLOCKS_PER_SEC ;

	printf("Calculation on CPU Done \n");
	printf("CPU saxpy: %f seconds.\n", cpu_time_used);

	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		if (result_y[i] != y[i])
			printf("Error in results \n");
	}

	printf("Comparing the output for each implementation Correct!");

	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);
	free(result_y);

	return 0;

}