__kernel void collection(
	int srcWidth, int srcHeight,
	int isRGB,
	__global int * srcR, __global int * srcG, __global int * srcB,
	__global double * totalR, __global double * totalG, __global double * totalB
)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

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