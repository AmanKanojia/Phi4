#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>


using namespace std;




__global__ void gbmKernel(float* g1, float start_val){

	int threadid = blockIdx.x*blockDim.x + threadIdx.x-1;
	curandState state;
	curand_init(1234, threadid,0,&state); // seed, sequence, offset
	g1[365*threadid] = start_val;
	for(int k=365*threadid; k < 365*(threadid+1); k++){
		g1[k]=g1[k-1]*(1+curand_normal(&state));	
	}

}



int main(){
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	int N = 500;
	int M=365; //Number of time steps
	int numThreadsPerBlock = 128;
	//int N = numThreadsPerBlock; //Number of GBM paths
	int NoOfBlocks = static_cast<int>(static_cast<float>(N)/numThreadsPerBlock) + 1;
	cout<<NoOfBlocks;
	int size = N*M*sizeof(float);
	float *P1 = new float[N*M];

	//P1[0]=1;
	float start_val=0.1;

	float *h1;
	cudaMalloc(&h1,size);

	cudaMemcpy(h1,P1,size,cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(numThreadsPerBlock);
	dim3 numBlocks(NoOfBlocks);

	gbmKernel<<<numBlocks, threadsPerBlock>>>(h1,start_val);
	cudaDeviceSynchronize();
	cudaMemcpy(P1,h1,size,cudaMemcpyDeviceToHost);


	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout<<"CUDA GBM time: "<<milliseconds/1000.0<<" seconds\n";

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(h1); 
	delete[] h1; 

	return 0;
}
