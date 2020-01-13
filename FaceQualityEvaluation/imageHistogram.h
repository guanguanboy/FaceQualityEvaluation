#pragma once
#include "util.h"

//计算图像直方图
class Histogram1D
{
private:
	int histSize[1];
	float hranges[2];
	const float* ranges[1];
	int channels[1];
public:
	Histogram1D()
	{
		histSize[0] = 256;
		hranges[0] = 0.0;
		hranges[1] = 255.0;//最大像素值
		ranges[0] = hranges;
		channels[0] = 0;
	}
	cv::Mat getHistogram(const cv::Mat& image)
	{
		cv::Mat hist;
		cv::calcHist(&image,
			1,//仅为一个图像的直方图
			channels,//使用的通道
			cv::Mat(),//不使用掩码
			hist,//作为结果的直方图
			1,//这时一维的直方图
			histSize,//箱子数量
			ranges//像素值的范围
			);
		return hist;
	}
};