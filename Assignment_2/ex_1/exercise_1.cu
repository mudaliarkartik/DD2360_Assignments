#include <stdio.h>
#define N 64
#define TPB 256


__global__ void print_kernel()
{
	printf("Hello world! My threadID is %d\n", threadIdx.x);
}

int main()
{

	print_kernel <<<1, TPB >>> ();

	cudaDeviceSynchronize();
	return 0;

}