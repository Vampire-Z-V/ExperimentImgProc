#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_math.h>
#include <helper_functions.h>
#include <helper_cuda.h>

inline __device__ double getWeight(double x)
{
	double f = (x / 2.0) * 1.5;
	if (f > -1.5 && f < -0.5)
	{
		return(0.5 * (f + 1.5)*(f + 1.5));
	}
	else if (f > -0.5 && f < 0.5)
	{
		return 3.0 / 4.0 - (f * f);
	}
	else if ((f > 0.5 && f < 1.5))
	{
		return(0.5 *(f - 1.5)*(f - 1.5));
	}
	return 0.0;
}

inline __device__ void backwardRotateCoordinate(double * x, double * y, int centerX, int centerY, double radian)
{
	*x = *x - centerX;
	*y = *y - centerY;

	double tempX = *x;
	double tempY = *y;

	*x = cos(radian) * tempX + sin(radian) * tempY;
	*y = -sin(radian) * tempX + cos(radian) * tempY;

	*x += centerX;
	*y += centerY;
}

inline __device__ int getPixel(int x, int y, int srcWidth, int srcHeight, int * colorSrc)
{
	x = x < 0 ? 0 : (x > srcWidth - 1 ? srcWidth - 1 : x);
	y = y < 0 ? 0 : (y > srcHeight - 1 ? srcHeight - 1 : y);

	return colorSrc[x + y * srcWidth];
}

__global__ void thirdOrderInterpolation(
	int srcWidth, int srcHeight,
	int targetWidth, int targetHeight,
	bool isRotate,
	double radian,
	bool isRGB,
	int * srcR, int * srcG, int * srcB,
	int * resultR, int * resultG, int * resultB,
	float * weight_out
)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < targetWidth && y < targetHeight)
	{
		int index = x + y * targetWidth;
		double backwardX;
		double backwardY;

		// find the new pixel backward to the src img
		if (isRotate)
		{
			backwardX = x + (srcWidth - targetWidth) / 2;
			backwardY = y + (srcHeight - targetHeight) / 2;

			backwardRotateCoordinate(&backwardX, &backwardY, srcWidth / 2, srcHeight / 2, radian);

			if (backwardX < 0 || backwardX > srcWidth - 1 ||
				backwardY < 0 || backwardY > srcHeight - 1)
			{
				if (isRGB)
				{
					resultR[index] = resultG[index] = -1;
				}
				resultB[index] = -1;
				return;
			}
		}
		else
		{
			backwardX = x / radian;
			backwardY = y / radian;
		}

		int backwardXInt = (int) floor(backwardX);
		int backwardYInt = (int) floor(backwardY);

		float backwardXfloat = backwardX - backwardXInt;
		float backwardYfloat = backwardY - backwardYInt;

		for (int m = -1; m < 3; m++)
		{
			for (int n = -1; n < 3; n++)
			{
				double w = getWeight(m - backwardXfloat) * getWeight(backwardYfloat - n);
				weight_out[index] += w;
				
				if (isRGB)
				{
					int RTemp = getPixel(backwardXInt + n, backwardYInt + m, srcWidth, srcHeight, srcR);
					int GTemp = getPixel(backwardXInt + n, backwardYInt + m, srcWidth, srcHeight, srcG);
					
					resultR[index] += RTemp * w;
					resultG[index] += GTemp * w;
				}
				int BTemp = getPixel(backwardXInt + n, backwardYInt + m, srcWidth, srcHeight, srcB);
				resultB[index] += BTemp * w;
			}
		}
	}
}

extern "C"
void Rotate_Scale(
	int srcWidth, int srcHeight,
	int targetWidth, int targetHeight,
	bool isRotate,
	double radian,
	bool isRGB,
	int * srcR, int * srcG, int * srcB,
	int * resultR, int * resultG, int * resultB,
	float * weight
)
{
	int * srcR_in;
	int * srcG_in;
	int * srcB_in;
	int *resultR_out;
	int *resultG_out;
	int *resultB_out;
	float *weight_out;

	int srcSize = srcWidth * srcHeight;
	int targetSize = targetWidth * targetHeight;

	dim3 dimBlock(32, 32);
	dim3 dimGrid(
		(targetWidth + dimBlock.x - 1) / dimBlock.x,
		(targetHeight + dimBlock.y - 1) / dimBlock.y
	);
	
	if (isRGB)
	{
		checkCudaErrors(cudaMalloc((void**) &srcR_in, sizeof(int) * srcSize));
		checkCudaErrors(cudaMalloc((void**) &srcG_in, sizeof(int) * srcSize));
		checkCudaErrors(cudaMalloc((void**) &resultR_out, sizeof(int) * targetSize));
		checkCudaErrors(cudaMalloc((void**) &resultG_out, sizeof(int) * targetSize));

		checkCudaErrors(cudaMemcpy(srcR_in, srcR, sizeof(int) * srcSize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(srcG_in, srcG, sizeof(int) * srcSize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(resultR_out, resultR, sizeof(int) * targetSize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(resultG_out, resultG, sizeof(int) * targetSize, cudaMemcpyHostToDevice));
	}
	checkCudaErrors(cudaMalloc((void**) &srcB_in, sizeof(int) * srcSize));
	checkCudaErrors(cudaMalloc((void**) &resultB_out, sizeof(int) * targetSize));

	checkCudaErrors(cudaMemcpy(srcB_in, srcB, sizeof(int) * srcSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(resultB_out, resultB, sizeof(int) * targetSize, cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMalloc((void**) &weight_out, sizeof(float) * targetSize));
	checkCudaErrors(cudaMemcpy(weight_out, weight, sizeof(float) * targetSize, cudaMemcpyHostToDevice));

	if (isRGB)
		thirdOrderInterpolation <<< dimGrid, dimBlock >>> (
			srcWidth, srcHeight,
			targetWidth, targetHeight,
			isRotate,
			radian,
			true,
			srcR_in, srcG_in, srcB_in,
			resultR_out, resultG_out, resultB_out,
			weight_out
		);
	else
		thirdOrderInterpolation <<< dimGrid, dimBlock >>> (
			srcWidth, srcHeight,
			targetWidth, targetHeight,
			isRotate,
			radian,
			false,
			NULL, NULL, srcB_in,
			NULL, NULL, resultB_out,
			weight_out
		);

	if (isRGB)
	{
		checkCudaErrors(cudaMemcpy(resultR, resultR_out, sizeof(int) * targetSize, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(resultG, resultG_out, sizeof(int) * targetSize, cudaMemcpyDeviceToHost));
		cudaFree(srcR_in);
		cudaFree(srcG_in);
		cudaFree(resultR_out);
		cudaFree(resultG_out);
	}
	checkCudaErrors(cudaMemcpy(resultB, resultB_out, sizeof(int) * targetSize, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(weight, weight_out, sizeof(float) * targetSize, cudaMemcpyDeviceToHost));
	cudaFree(srcB_in);
	cudaFree(resultB_out);
	cudaFree(weight_out);
}