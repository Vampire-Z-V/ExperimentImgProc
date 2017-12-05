#include "stdafx.h"
#include "ImageProcess.h"
#include "MultiThreadController.h"
#include <algorithm>

namespace AutoLevel
{
	double freqency_B[GRAYLEVEL];
	double freqency_G[GRAYLEVEL];
	double freqency_R[GRAYLEVEL];


	void init()
	{
		for (int i = 0; i < GRAYLEVEL; i++)
		{
			freqency_B[i] = 0.0;
			freqency_G[i] = 0.0;
			freqency_R[i] = 0.0;
		}
	}
}

namespace AutoWhite
{
	double B = 0.0;
	double G = 0.0;
	double R = 0.0;

	void init()
	{
		B = G = R = 0.0;
	}
}

static int checkRange(int src, int min, int max)
{

	if (src < min)
	{
		return min;
	}
	if (src > max)
	{
		return max;
	}
	return src;

}

UINT ImageProcess::addNoise(LPVOID p)
{
	srand((unsigned) time(NULL));
	ThreadParam* param = (ThreadParam*) p;
	int maxWidth = param->imgSrc_1->GetWidth();
	int maxHeight = param->imgSrc_1->GetHeight();

	int startIndex = param->startIndex;
	int endIndex = param->endIndex;
	byte* pRealData = (byte*) param->imgSrc_1->GetBits();
	int bitCount = param->imgSrc_1->GetBPP() / 8;
	int pit = param->imgSrc_1->GetPitch();

	for (int i = startIndex; i <= endIndex; ++i)
	{
		int x = i % maxWidth;
		int y = i / maxWidth;
		if ((rand() % 1000) * 0.001 < NOISE)
		{
			int value = 0;
			if (rand() % 1000 < 500)
			{
				value = 0;
			}
			else
			{
				value = 255;
			}
			if (bitCount == 1)
			{
				*(pRealData + pit * y + x * bitCount) = value;
			}
			else
			{
				*(pRealData + pit * y + x * bitCount) = value;
				*(pRealData + pit * y + x * bitCount + 1) = value;
				*(pRealData + pit * y + x * bitCount + 2) = value;
			}
		}
	}
	::PostMessage(param->handle, WM_NOISE, 1, WM_NOISE);
	return 0;
}

UINT ImageProcess::medianFilter(LPVOID p)
{
	ThreadParam* param = (ThreadParam*) p;

	int maxWidth = param->imgSrc_1->GetWidth();
	int maxHeight = param->imgSrc_1->GetHeight();
	int startIndex = param->startIndex;
	int endIndex = param->endIndex;
	int maxSpan = MAX_SPAN;
	int maxLength = (maxSpan * 2 + 1) * (maxSpan * 2 + 1);

	byte* pRealData = (byte*) param->imgSrc_1->GetBits();
	int pit = param->imgSrc_1->GetPitch();
	int bitCount = param->imgSrc_1->GetBPP() / 8;

	int *pixel = new int[maxLength];
	int *pixelR = new int[maxLength];
	int *pixelB = new int[maxLength];
	int *pixelG = new int[maxLength];
	int index = 0;
	for (int i = startIndex; i <= endIndex; ++i)
	{
		int Sxy = 1;
		int med = 0;
		bool state = false;
		int x = i % maxWidth;
		int y = i / maxWidth;
		while (Sxy <= maxSpan)
		{
			index = 0;
			for (int tmpY = y - Sxy; tmpY <= y + Sxy && tmpY < maxHeight; tmpY++)
			{
				if (tmpY < 0) continue;
				for (int tmpX = x - Sxy; tmpX <= x + Sxy && tmpX < maxWidth; tmpX++)
				{
					if (tmpX < 0) continue;
					if (bitCount == 1)
					{
						pixel[index] = *(pRealData + pit * tmpY + tmpX * bitCount);
						pixelR[index++] = pixel[index];
					}
					else
					{												 
						pixelR[index] = *(pRealData + pit * tmpY + tmpX * bitCount + 2);
						pixelG[index] = *(pRealData + pit * tmpY + tmpX * bitCount + 1);
						pixelB[index] = *(pRealData + pit * tmpY + tmpX * bitCount + 0);
						pixel[index++] = int(pixelB[index] * 0.299 + 0.587*pixelG[index] + pixelR[index] * 0.144);
					}
				}

			}
			if (index <= 0)
				break;
			if ((state = GetValue(pixel, index, med)))
				break;

			Sxy++;
		};

		if (state)
		{
			if (bitCount == 1)
			{
				*(pRealData + pit * y + x * bitCount) = pixelR[med];
			}
			else
			{
				*(pRealData + pit * y + x * bitCount + 2) = pixelR[med];
				*(pRealData + pit * y + x * bitCount + 1) = pixelG[med];
				*(pRealData + pit * y + x * bitCount + 0) = pixelB[med];
			}
		}

	}

	delete[] pixel;
	delete[] pixelR;
	delete[] pixelG;
	delete[] pixelB;

	::PostMessage(param->handle, WM_MEDIAN_FILTER, 1, WM_MEDIAN_FILTER);
	return 0;
}

UINT ImageProcess::rotate(LPVOID p)
{
	thirdOrderInterpolation(p);

	::PostMessage(((ThreadParam*)p)->handle, WM_ROTATE, 1, WM_ROTATE);
	return 0;
}

UINT ImageProcess::scale(LPVOID p)
{
	thirdOrderInterpolation(p, false);

	::PostMessage(((ThreadParam*) p)->handle, WM_SCALE, 1, WM_SCALE);
	return 0;
}

UINT ImageProcess::marge(LPVOID p)
{
	MargeThreadParam * param = (MargeThreadParam*) p;

	CImage * src_1 = param->src_1;
	CImage * src_2 = param->src_2;
	CImage * target = param->target;

	int maxWidth = target->GetWidth();
	int maxHeight = target->GetHeight();

	int startIndex = param->startIndex;
	int endIndex = param->endIndex;
	float alpha = param->alpha;

	byte* pRealData1 = (byte*) src_1->GetBits();
	byte* pRealData2 = (byte*) src_2->GetBits();
	byte* pRealDataTarget = (byte*) target->GetBits();

	int pit = target->GetPitch();
	int bitCount = target->GetBPP() / 8;

	for (int i = startIndex; i <= endIndex; i++)
	{
		int x = i % maxWidth;
		int y = i / maxWidth;

		int B1, G1, R1, B2, G2, R2;

		if (bitCount == 1)
		{
			B1 = *(pRealData1 + pit*y + x*bitCount + 0);
			B2 = *(pRealData2 + pit*y + x*bitCount + 0);
			*(pRealDataTarget + pit*y + x*bitCount + 0) = round(B1 * alpha + B2 * (1 - alpha));
		}
		else
		{
			B1 = *(pRealData1 + pit*y + x*bitCount + 0);
			G1 = *(pRealData1 + pit*y + x*bitCount + 1);
			R1 = *(pRealData1 + pit*y + x*bitCount + 2);
			B2 = *(pRealData2 + pit*y + x*bitCount + 0);
			G2 = *(pRealData2 + pit*y + x*bitCount + 1);
			R2 = *(pRealData2 + pit*y + x*bitCount + 2);

			*(pRealDataTarget + pit*y + x*bitCount + 0) = round(B1 * alpha + B2 * (1 - alpha));
			*(pRealDataTarget + pit*y + x*bitCount + 1) = round(G1 * alpha + G2 * (1 - alpha));
			*(pRealDataTarget + pit*y + x*bitCount + 2) = round(R1 * alpha + R2 * (1 - alpha));
		}
		
	}

	::PostMessage(param->handle, WM_MARGE, 1, WM_MARGE);
	return 0;
}

UINT ImageProcess::autoLevels(LPVOID p)
{
	ThreadParam* param = (ThreadParam*) p;
	MultiThreadController* mtc = MultiThreadController::getInstance();

	int maxWidth = param->imgSrc_1->GetWidth();
	int maxHeight = param->imgSrc_1->GetHeight();
	int startIndex = param->startIndex;
	int endIndex = param->endIndex;

	byte* pRealData = (byte*) param->imgSrc_1->GetBits();
	int pit = param->imgSrc_1->GetPitch();
	int bitCount = param->imgSrc_1->GetBPP() / 8;
	bool isRGB = bitCount == 3;

	int pixelSum = maxWidth * maxHeight;

	for (int i = startIndex; i <= endIndex; i++)
	{
		int x = i % maxWidth;
		int y = i / maxWidth;

		AutoLevel::freqency_B[*(pRealData + pit * y + x * bitCount + 0)]++;
		if (isRGB)
		{
			AutoLevel::freqency_G[*(pRealData + pit * y + x * bitCount + 1)]++;
			AutoLevel::freqency_R[*(pRealData + pit * y + x * bitCount + 2)]++;
		}
	}

	mtc->finish();
	mtc->waitAllFinished();

	// only thread 0 do this task
	if (param->thread_id == 0)
	{
		AutoLevel::freqency_B[0] /= pixelSum;
		if (isRGB)
		{
			AutoLevel::freqency_G[0] /= pixelSum;
			AutoLevel::freqency_R[0] /= pixelSum;
		}
		for (int i = 1; i < GRAYLEVEL; i++)
		{
			AutoLevel::freqency_B[i] = AutoLevel::freqency_B[i] / pixelSum + AutoLevel::freqency_B[i - 1];
			if (isRGB)
			{
				AutoLevel::freqency_G[i] = AutoLevel::freqency_G[i] / pixelSum + AutoLevel::freqency_G[i - 1];
				AutoLevel::freqency_R[i] = AutoLevel::freqency_R[i] / pixelSum + AutoLevel::freqency_R[i - 1];
			}
		}

		for (int i = 0; i < GRAYLEVEL; i++)
		{
			AutoLevel::freqency_B[i] *= (GRAYLEVEL - 1);
			if (isRGB)
			{
				AutoLevel::freqency_G[i] *= (GRAYLEVEL - 1);
				AutoLevel::freqency_R[i] *= (GRAYLEVEL - 1);
			}
		}
		mtc->finish(param->thread_id);
	}

	// all thread must wait for thread 0 finishing the task above
	mtc->waitFor(0);

	for (int i = startIndex; i <= endIndex; i++)
	{
		int x = i % maxWidth;
		int y = i / maxWidth;

		int srcGray_B = *(pRealData + pit * y + x * bitCount + 0);
		int newGray_B = round(AutoLevel::freqency_B[srcGray_B]);
		newGray_B = newGray_B < 0 ? 0 : (newGray_B >= GRAYLEVEL ? GRAYLEVEL - 1 : newGray_B);
		*(pRealData + pit * y + x * bitCount + 0) = newGray_B;

		if (isRGB)
		{
			int srcGray_G = *(pRealData + pit * y + x * bitCount + 1);
			int srcGray_R = *(pRealData + pit * y + x * bitCount + 2);

			int newGray_G = round(AutoLevel::freqency_G[srcGray_G]);
			int newGray_R = round(AutoLevel::freqency_R[srcGray_R]);

			newGray_G = newGray_G < 0 ? 0 : (newGray_G >= GRAYLEVEL ? GRAYLEVEL - 1 : newGray_G);
			newGray_R = newGray_R < 0 ? 0 : (newGray_R >= GRAYLEVEL ? GRAYLEVEL - 1 : newGray_R);

			*(pRealData + pit * y + x * bitCount + 1) = newGray_G;
			*(pRealData + pit * y + x * bitCount + 2) = newGray_R;
		}

	}

	::PostMessage(param->handle, WM_AUTO_LEVELS, 1, WM_AUTO_LEVELS);
	return 0;
}

UINT ImageProcess::autoWhite(LPVOID p)
{
	ThreadParam* param = (ThreadParam*) p;
	MultiThreadController* mtc = MultiThreadController::getInstance();

	int maxWidth = param->imgSrc_1->GetWidth();
	int maxHeight = param->imgSrc_1->GetHeight();
	int startIndex = param->startIndex;
	int endIndex = param->endIndex;

	byte* pRealData = (byte*) param->imgSrc_1->GetBits();
	int pit = param->imgSrc_1->GetPitch();
	int bitCount = param->imgSrc_1->GetBPP() / 8;
	bool isRGB = bitCount != 1;

	int pixelSum = maxWidth * maxHeight;
	
	for (int i = startIndex; i <= endIndex; i++)
	{
		int x = i % maxWidth;
		int y = i / maxWidth;

		AutoWhite::B += *(pRealData + pit * y + x * bitCount + 0);
		if (isRGB)
		{
			AutoWhite::G += *(pRealData + pit * y + x * bitCount + 1);
			AutoWhite::R += *(pRealData + pit * y + x * bitCount + 2);
		}
	}

	mtc->finish();

	// must start after all threads done the count task above
	mtc->waitAllFinished();
	double K;

	if (isRGB)
		K = (AutoWhite::R + AutoWhite::G + AutoWhite::B) / pixelSum / 3;
	else
		K = AutoWhite::B / pixelSum;

	double KB;
	double KG;
	double KR;

	KB = K / AutoWhite::B * pixelSum;
	if (isRGB)
	{
		KG = K / AutoWhite::G * pixelSum;
		KR = K / AutoWhite::R * pixelSum;
	}

	for (int i = startIndex; i <= endIndex; i++)
	{
		int x = i % maxWidth;
		int y = i / maxWidth;

		int srcGray_B = *(pRealData + pit * y + x * bitCount + 0) * KB;
		srcGray_B = srcGray_B >= GRAYLEVEL ? GRAYLEVEL - 1 : srcGray_B;
		*(pRealData + pit * y + x * bitCount + 0) = srcGray_B;

		if (isRGB)
		{
			int srcGray_G = *(pRealData + pit * y + x * bitCount + 1) * KG;
			int srcGray_R = *(pRealData + pit * y + x * bitCount + 2) * KR;

			srcGray_G = srcGray_G >= GRAYLEVEL ? GRAYLEVEL - 1 : srcGray_G;
			srcGray_R = srcGray_R >= GRAYLEVEL ? GRAYLEVEL - 1 : srcGray_R;

			*(pRealData + pit * y + x * bitCount + 1) = srcGray_G;
			*(pRealData + pit * y + x * bitCount + 2) = srcGray_R;
		}
		
	}

	::PostMessage(param->handle, WM_AUTO_WHITE, 1, WM_AUTO_WHITE);
	return 0;
}

UINT ImageProcess::bilateralFilter(LPVOID p)
{
	ThreadParam* param = (ThreadParam*) p;
	if (param->imgSrc_1->GetBPP() == 8)
	{
		bilateralFilter(p, 0);
	}
	else
	{
		bilateralFilter(p, 0);
		bilateralFilter(p, 1);
		bilateralFilter(p, 2);
	}

	::PostMessage(param->handle, WM_BILATERAL_FILTER, 1, WM_BILATERAL_FILTER);
	return 0;
}


bool ImageProcess::GetValue(int p[], int size, int & value)
{
	int zxy = p[(size - 1) / 2];

	int *a = new int[size];
	int index = 0;
	for (int i = 0; i < size; ++i)
		a[index++] = i;

	// Bubble Sort
	for (int i = 0; i < size - 1; i++)
		for (int j = i + 1; j < size; j++)
			if (p[i] > p[j])
			{
				int tempA = a[i];
				a[i] = a[j];
				a[j] = tempA;
				int temp = p[i];
				p[i] = p[j];
				p[j] = temp;

			}
	int zmax = p[size - 1];
	int zmin = p[0];
	int zmed = p[(size - 1) / 2];

	if (zmax > zmed && zmin < zmed)
	{
		if (zxy > zmin && zxy < zmax)
			value = (size - 1) / 2;
		else
			value = a[(size - 1) / 2];
		delete[] a;
		return true;
	}
	else
	{
		delete[] a;
		return false;
	}
}

double ImageProcess::bellInterpolation(double x)
{
	double f = (x / 2.0) * 1.5;
	if (f > -1.5 && f < -0.5)
	{
		return(0.5 * pow(f + 1.5, 2.0));
	}
	else if (f > -0.5 && f < 0.5)
	{
		return 3.0 / 4.0 - (f * f);
	}
	else if ((f > 0.5 && f < 1.5))
	{
		return(0.5 * pow(f - 1.5, 2.0));
	}
	return 0.0;
}

int * ImageProcess::getPixel(int x, int y, CImage * src)
{
	int bitCount = src->GetBPP() / 8;
	byte* pSrcData = (byte*) src->GetBits();
	int pit = src->GetPitch();
	int maxWidth = src->GetWidth();
	int maxHeight = src->GetHeight();

	x = x < 0 ? 0 : (x > maxWidth - 1 ? maxWidth - 1 : x);
	y = y < 0 ? 0 : (y > maxHeight - 1 ? maxHeight - 1 : y);

	int * pixel = NULL;

	if (bitCount == 1)
	{
		pixel = new int[1];
		pixel[0] = *(pSrcData + pit * y + x * bitCount);
	}
	else
	{
		pixel = new int[3];
		pixel[0] = *(pSrcData + pit * y + x * bitCount);
		pixel[1] = *(pSrcData + pit * y + x * bitCount + 1);
		pixel[2] = *(pSrcData + pit * y + x * bitCount + 2);
	}
	return pixel;
}

void ImageProcess::rotateCoordinate(double * x, double * y, int centerX, int centerY, double radian)
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

void ImageProcess::getSourceColors(CImage & src, int ** srcR, int ** srcG, int ** srcB)
{
	byte* pSrcData = (byte*) src.GetBits();
	int srcPit = src.GetPitch();
	int bitCount = src.GetBPP() / 8;
	bool isRGB = bitCount != 1;

	int srcWidth = src.GetWidth();
	int srcHeight = src.GetHeight();
	int srcSize = srcWidth * srcHeight;

	*srcR = isRGB ? new int[srcSize] : NULL;
	*srcG = isRGB ? new int[srcSize] : NULL;
	*srcB = new int[srcSize];
	
	int index = 0;
	for (int y = 0; y < srcHeight; y++)
	{
		for (int x = 0; x < srcWidth; x++)
		{
			if (isRGB)
			{
				(*srcR)[index] = *(pSrcData + srcPit * y + x * bitCount + 2);
				(*srcG)[index] = *(pSrcData + srcPit * y + x * bitCount + 1);
			}
			(*srcB)[index++] = *(pSrcData + srcPit * y + x * bitCount + 0);
		}
	}
}

void ImageProcess::thirdOrderInterpolation(LPVOID p, bool isRotate)
{
	ThreadParam* param = (ThreadParam*) p;

	int targetWidth = param->imgTemp->GetWidth();
	int targetHeight = param->imgTemp->GetHeight();
	int srcWidth = param->imgSrc_1->GetWidth();
	int srcHeight = param->imgSrc_1->GetHeight();

	int startIndex = param->startIndex;
	int endIndex = param->endIndex;

	byte* pSrcData = (byte*) param->imgSrc_1->GetBits();
	byte* pTargetData = (byte*) param->imgTemp->GetBits();

	int srcPit = param->imgSrc_1->GetPitch();
	int targetPit = param->imgTemp->GetPitch();
	int bitCount = param->imgTemp->GetBPP() / 8;

	COLORREF backColor = GetSysColor(COLOR_3DFACE);


	for (int i = startIndex; i <= endIndex; i++)
	{
		int targetX = i % targetWidth;
		int targetY = i / targetWidth;

		double srcX;
		double srcY;

		if (isRotate)
		{
			double radian = param->radian;

			srcX = targetX + (srcWidth / 2 - targetWidth / 2);
			srcY = targetY + (srcHeight / 2 - targetHeight / 2);

			rotateCoordinate(&srcX, &srcY, srcWidth / 2, srcHeight / 2, radian);

			if (srcX < 0 || srcX > srcWidth - 1 ||
				srcY < 0 || srcY > srcHeight - 1)
			{
				COLORREF backColor = GetSysColor(COLOR_3DFACE);
				if (bitCount == 1)
				{
					*(pTargetData + targetPit * targetY + targetX * bitCount) = GetBValue(backColor);
				}
				else
				{
					*(pTargetData + targetPit * targetY + targetX * bitCount + 0) = GetBValue(backColor);
					*(pTargetData + targetPit * targetY + targetX * bitCount + 1) = GetGValue(backColor);
					*(pTargetData + targetPit * targetY + targetX * bitCount + 2) = GetRValue(backColor);
				}
				continue;
			}
		}
		else
		{
			double ratio = param->scale_ratio;
			srcX = (double) targetX / ratio;
			srcY = (double) targetY / ratio;
		}

		int srcXInt = (int) floor(srcX);
		int srcYInt = (int) floor(srcY);

		double srcXfloat = srcX - srcXInt;
		double srcYfloat = srcY - srcYInt;

		int rgbData[3] = { 0 };
		double weight = 0;

		for (int m = -1; m < 3; m++)
		{
			for (int n = -1; n < 3; n++)
			{
				int * rgbTemp = getPixel(srcXInt + n, srcYInt + m, param->imgSrc_1);

				double fun1, fun2 = 0;
				fun1 = bellInterpolation(m - srcXfloat);
				fun2 = bellInterpolation(srcYfloat - n);

				weight += fun1*fun2;

				if (bitCount == 1)
				{
					rgbData[0] += rgbTemp[0] * fun1 * fun2;
				}
				else
				{
					rgbData[0] += rgbTemp[0] * fun1 * fun2;
					rgbData[1] += rgbTemp[1] * fun1 * fun2;
					rgbData[2] += rgbTemp[2] * fun1 * fun2;
				}

				delete rgbTemp;
			}
		}

		if (bitCount == 1)
		{
			*(pTargetData + targetPit * targetY + targetX * bitCount) = rgbData[0] / weight;
		}
		else
		{
			*(pTargetData + targetPit * targetY + targetX * bitCount + 0) = rgbData[0] / weight;
			*(pTargetData + targetPit * targetY + targetX * bitCount + 1) = rgbData[1] / weight;
			*(pTargetData + targetPit * targetY + targetX * bitCount + 2) = rgbData[2] / weight;
		}
	}
}

void ImageProcess::bilateralFilter(LPVOID p, int RGBOffset)
{
	ThreadParam* param = (ThreadParam*) p;
	int maxWidth = param->imgSrc_1->GetWidth();
	int maxHeight = param->imgSrc_1->GetHeight();
	int startIndex = param->startIndex;
	int endIndex = param->endIndex;

	byte* pRealData = (byte*) param->imgSrc_1->GetBits();
	int pit = param->imgSrc_1->GetPitch();
	int bitCount = param->imgSrc_1->GetBPP() / 8;

	for (int i = startIndex; i <= endIndex; i++)
	{
		int x = i % maxWidth;
		int y = i / maxWidth;

		int centralPixel = *(pRealData + pit * y + x * bitCount + RGBOffset);
		double newPixel = 0.0;
		double w = 0.0;

		for (int neighbor_y = y - DIAMETER; neighbor_y <= y + DIAMETER && neighbor_y < maxHeight; neighbor_y++)
		{
			if (neighbor_y < 0) continue;
			for (int neighbor_x = x - DIAMETER; neighbor_x <= x + DIAMETER && neighbor_x < maxWidth; neighbor_x++)
			{
				if (neighbor_x < 0) continue;

				int neighborPixel = *(pRealData + pit * neighbor_y + neighbor_x * bitCount + RGBOffset);

				double wSpace = gaussian(distance(x, y, neighbor_x, neighbor_y), SIGMA_SPACE);
				double wColor = gaussian(neighborPixel - centralPixel, SIGMA_COLOR);

				newPixel += wSpace * wColor * neighborPixel;
				w += wSpace * wColor;
			}
		}
		*(pRealData + pit * y + x * bitCount + RGBOffset) = newPixel / w;
	}

}

float ImageProcess::distance(int x, int y, int i, int j)
{
	return float(sqrt(pow(x - i, 2) + pow(y - j, 2)));
}

double ImageProcess::gaussian(float x, double sigma)
{
	return exp(-(pow(x, 2)) / (2 * pow(sigma, 2)));
}

