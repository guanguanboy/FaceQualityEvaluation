#include "stdafx.h"
#include "illuminationDetect.h"

float illlut[101] = { 0.30,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.40,
0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.50,
0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.60,
0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.70,
0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.80,
0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.90,
0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1.00,
1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09,1.10,
1.11,1.12,1.13,1.14,1.15,1.16,1.17,1.18,1.19,1.20,
1.21,1.22,1.23,1.24,1.25,1.26,1.27,1.28,1.29,1.30};

int illNum = (maxGamma - minGamma)*1.0 / 0.01;

int illuminationDetect(cv::Mat srcImage, cv::Mat image_cut, float &illumination)
{
	assert(srcImage.channels == 1);
	if (srcImage.empty())
	{
		return -1;
	}

	cv::Mat mat_mean, mat_stddev;
	cv::meanStdDev(image_cut, mat_mean, mat_stddev); //计算矩阵的均值和标准差
	double m = mat_mean.at<double>(0, 0);
	illumination = m;

	illumination = illumination > maxILLMEAN ? maxILLMEAN : illumination;
	illumination = illumination < minILLMEAN ? minILLMEAN : illumination;

	return 0;
}

int gammaCorrection(cv::Mat srcImage, cv::Mat &image_gamma, float gamma)
{
	unsigned char lut[256];
	for (int i = 0; i < 256; i++)
	{
		lut[i] = cv::saturate_cast<uchar>(pow((float)(i / 255.0), gamma) * 255.0f);
	}

	image_gamma = srcImage.clone();
	cv::MatIterator_<uchar> it, end;
	for (it = image_gamma.begin<uchar>(), end = image_gamma.end<uchar>(); it != end; it++)
		*it = lut[(*it)];

	return 0;
}