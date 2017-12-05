#pragma once
#define NOISE          0.2
#define MAX_SPAN       15
#define PI             3.1415926535897932384626433832795
#define GRAYLEVEL      256

#define DIAMETER       5
#define SIGMA_COLOR    12
#define SIGMA_SPACE    16

struct ThreadParam
{
	CImage * imgSrc_1;
	CImage * imgSrc_2;
	CImage * imgTemp;
	int startIndex;
	int endIndex;
	double radian;
	float scale_ratio;
	float alpha;
	int thread_id;
	HWND handle;
};

struct BasicThreadParam
{
	CImage * src;
	int startIndex;
	int endIndex;
	int thread_id;
	HWND handle;
};

struct TempThreadParam
	: BasicThreadParam
{
	CImage * target;
};

struct RotateThreadParam
	: TempThreadParam
{
	double radian;
};

struct ScaleThreadParam
	: TempThreadParam
{
	float scale_ratio;
};

struct MargeThreadParam
{
	CImage * src_1;
	CImage * src_2;
	CImage * target;
	int startIndex;
	int endIndex;
	float alpha;
	HWND handle;
};

namespace AutoLevel
{
	extern double freqency_B[GRAYLEVEL];
	extern double freqency_G[GRAYLEVEL];
	extern double freqency_R[GRAYLEVEL];

	void init();
}

namespace AutoWhite
{
	extern double B;
	extern double G;
	extern double R;

	void init();
}

class ImageProcess
{
public:
	static UINT addNoise(LPVOID param);
	static UINT medianFilter(LPVOID param);
	static UINT rotate(LPVOID param);
	static UINT scale(LPVOID param);
	static UINT marge(LPVOID param);
	static UINT autoLevels(LPVOID param);
	static UINT autoWhite(LPVOID param);
	static UINT bilateralFilter(LPVOID param);

	static void rotate_scale_CUDA(LPVOID param, bool isRotate);
	static void rotateCoordinate(double * x, double * y, int centerX, int centerY, double radian);
	static void getSourceColors(CImage & src, int ** srcR, int ** srcG, int ** srcB);
private:
	static bool GetValue(int p[], int size, int &value);
	static double bellInterpolation(double x);
	static int* getPixel(int x, int y, CImage * src);
	static void thirdOrderInterpolation(LPVOID param, bool isRotate=true);
	static void bilateralFilter(LPVOID param, int RGBOffset);

	static float distance(int x, int y, int i, int j);
	static double gaussian(float x, double sigma);
};

