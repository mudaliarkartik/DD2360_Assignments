#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#define TPB 256
#define PARTICLES 10000000
#define ITTERATIONS 10

struct Particle
{
	float3 position;
	float3 velocity;
};

__device__ float3 operator+(const float3& p1, const float3& p2)
{
	return make_float3(p1.x + p2.x, p1.y + p2.y, p1.z + p2.z);
}

__device__ float3 operator*(const float3& p1, const int& p2)
{
	return make_float3(p1.x * p2, p1.y * p2, p1.z * p2);
}

__host__ int operator!=(const float3& p1, const float3& p2)
{
	if (p1.x != p2.x)
		return 1;
	else if (p1.y != p2.y)
		return 1;
	else if (p1.z != p2.z)
		return 1;
	else
		return 0;
}


__global__ void update_gpu(Particle* particles, int dt)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float3 delta_velocity = make_float3(1, 2, 3); 
	particles[i].velocity = particles[i].velocity + delta_velocity;
	particles[i].position = particles[i].position + particles[i].velocity * dt;

}

void update_cpu(int n, Particle* particles, int dt)
{
	for (int i = 0; i < n; i++)
	{
		particles[i].velocity.x += 1;
		particles[i].velocity.y += 2;
		particles[i].velocity.z += 3;
		particles[i].position.x += particles[i].velocity.x * dt;
		particles[i].position.y += particles[i].velocity.y * dt;
		particles[i].position.z += particles[i].velocity.z * dt;
	}
}

int particle_compare(int n, Particle* cpu, Particle* gpu)
{
	for (int i = 0; i < n; i++)
	{
		if (cpu[i].position != gpu[i].position || cpu[i].velocity != gpu[i].velocity)
			return 0;
	}
	return 1;
}


int main()
{
	double time_diff = 0.0;
	clock_t start, end;

	//Particle *particles = (Particle*)malloc(PARTICLES * sizeof(Particle));
	Particle* particles;
	cudaHostAlloc(&particles, PARTICLES * sizeof(Particle), cudaHostAllocDefault);

	for (int i = 0; i < PARTICLES; i++)
	{
		particles[i].position = make_float3(rand() % 100, rand() % 100, rand() % 100);
		particles[i].velocity = make_float3(rand() % 100, rand() % 100, rand() % 100);
	}

	Particle* particles_gpu;
	//Particle* results_particles = (Particle*)malloc(PARTICLES * sizeof(Particle));
	Particle* results_particles;
	cudaHostAlloc(&results_particles, PARTICLES * sizeof(Particle), cudaHostAllocDefault);

	cudaMalloc(&particles_gpu, PARTICLES * sizeof(Particle));
	//cudaMallocHost(&particles_gpu, PARTICLES * sizeof(Particle),cudaHostAllocDefault);

	
	start = clock();
	
	for (int i = 0; i < ITTERATIONS; i++)
	{
		cudaMemcpy(particles_gpu, particles, PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);
		update_gpu <<< (PARTICLES + TPB - 1) / TPB, TPB >>> (particles_gpu, 1);
		cudaMemcpy(results_particles, particles_gpu, PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);
		
	}
	
	cudaDeviceSynchronize();
	end = clock();

	time_diff = (double)(end - start) / CLOCKS_PER_SEC;
	printf("GPU execution time: %f seconds\n", time_diff);


	start = clock();
	for (int i = 0; i < ITTERATIONS; i++)
	{
		update_cpu(PARTICLES, particles, 1);
	}
	end = clock();

	time_diff = (double)(end - start) / CLOCKS_PER_SEC;

	printf("CPU execution time: %f seconds\n", time_diff);


	/*if (particle_compare(PARTICLES, particles, results_particles))
		printf("Comparison Successful\n");
	else
		printf("Error\n");*/

	cudaFree(particles_gpu);
	//free(particles);
	//free(results_particles);
	cudaFreeHost(particles);
	cudaFreeHost(results_particles);


	return 0;
}