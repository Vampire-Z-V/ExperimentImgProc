//#pragma OPENCL EXTENSION cl_khr_fp64: enable

#define CHECKRANG(x, small, large) \
		(x < small ? small : (x > large ? large : x))

#define INDEX(x, y, w) \
		(x + y * w)

inline double getWeight(double x)
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

inline void backwardRotateCoordinate(double * x, double * y, int centerX, int centerY, double radian)
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

__kernel void Rotate_Scale(
	int srcWidth, int srcHeight,
	int targetWidth, int targetHeight,
	int isRotate,
	double radian,
	int isRGB,
	__global int * srcR, __global int * srcG, __global int * srcB,
	__global int * resultR, __global int * resultG, __global int * resultB,
	__global float * weight_out
)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	if (x < targetWidth && y < targetHeight)
	{
		int index = INDEX(x, y, targetWidth);
		
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

				int p_x = CHECKRANG(backwardXInt + n, 0, srcWidth - 1);
				int p_y = CHECKRANG(backwardYInt + m, 0, srcHeight - 1);
				int p_index = INDEX(p_x, p_y, srcWidth);

				if (isRGB)
				{
					int RTemp = srcR[p_index];
					int GTemp = srcG[p_index];

					resultR[index] += RTemp * w;
					resultG[index] += GTemp * w;
				}
				int BTemp = srcB[p_index];
				resultB[index] += BTemp * w;
			}
		}
	}
}