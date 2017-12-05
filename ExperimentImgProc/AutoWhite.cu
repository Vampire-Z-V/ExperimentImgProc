#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_math.h>
#include <helper_functions.h>
#include <helper_cuda.h>


__global__ void collection(
	int srcWidth, int srcHeight,
	bool isRGB,
	int * srcR, int * srcG, int * srcB,
	double * totalR, double * totalG, double * totalB
)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < srcWidth && y < srcHeight)
	{
		int index = x + y * srcWidth;
		if (isRGB)
		{
			*totalR += srcR[index];
			*totalG += srcG[index];
		}
		*totalB += srcB[index];
	}
}

extern "C"
void collectRGB(
	int srcWidth, int srcHeight,
	bool isRGB,
	int * srcR, int * srcG, int * srcB,
	double * totalR, double * totalG, double * totalB
)
{
	int * srcR_in;
	int * srcG_in;
	int * srcB_in;
	double *totalR_out;
	double *totalG_out;
	double *totalB_out;

	int srcSize = srcWidth * srcHeight;

	dim3 dimBlock(32, 32);
	dim3 dimGrid(
		(srcWidth + dimBlock.x - 1) / dimBlock.x,
		(srcHeight + dimBlock.y - 1) / dimBlock.y
	);

	if (isRGB)
	{
		checkCudaErrors(cudaMalloc((void**) &srcR_in, sizeof(int) * srcSize));
		checkCudaErrors(cudaMalloc((void**) &srcG_in, sizeof(int) * srcSize));
		checkCudaErrors(cudaMalloc((void**) &totalR_out, sizeof(double)));
		checkCudaErrors(cudaMalloc((void**) &totalG_out, sizeof(double)));

		checkCudaErrors(cudaMemcpy(srcR_in, srcR, sizeof(int) * srcSize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(srcG_in, srcG, sizeof(int) * srcSize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(totalR_out, totalR, sizeof(double), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(totalG_out, totalG, sizeof(double), cudaMemcpyHostToDevice));
	}
	checkCudaErrors(cudaMalloc((void**) &srcB_in, sizeof(int) * srcSize));
	checkCudaErrors(cudaMalloc((void**) &totalB_out, sizeof(double)));

	checkCudaErrors(cudaMemcpy(srcB_in, srcB, sizeof(int) * srcSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(totalB_out, totalB, sizeof(double), cudaMemcpyHostToDevice));

	if (isRGB)
		collection <<< dimGrid, dimBlock >>> (
			srcWidth, srcHeight,
			true,
			srcR_in, srcG_in, srcB_in,
			totalR_out, totalG_out, totalB_out
		);
	else
		collection <<< dimGrid, dimBlock >>> (
			srcWidth, srcHeight,
			false,
			NULL, NULL, srcB_in,
			NULL, NULL, totalB_out
		);

	if (isRGB)
	{
		checkCudaErrors(cudaMemcpy(totalR, totalR_out, sizeof(double), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(totalG, totalG_out, sizeof(double), cudaMemcpyDeviceToHost));
		cudaFree(srcR_in);
		cudaFree(srcG_in);
		cudaFree(totalR_out);
		cudaFree(totalG_out);
	}
	checkCudaErrors(cudaMemcpy(totalB, totalB_out, sizeof(double), cudaMemcpyDeviceToHost));
	cudaFree(srcB_in);
	cudaFree(totalB_out);
}