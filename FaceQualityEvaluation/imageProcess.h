#pragma once

#include "util.h"

#define MDIFFVALUE 100
#define DIFFVALUE 30

#define RATE 0.95

#define HSCALE 0.8//高度保留比例
#define WSCALE 0.8//宽度保留比例

#define MAXDEPTH 3000//深度
#define MINDEPTH 20

#define MINPV 100

#define THRED 0.97

int imageProcess0(cv::Mat srcImage, std::vector<cv::Point2f> featruePoint,
	cv::Mat &dstImage, cv::Mat &srcImage_cut, cv::Mat &dstImage_cut);

int imageProcess(cv::Mat srcImage, std::vector<cv::Point2f> featruePoint, 
	cv::Mat &dstImage, cv::Mat &srcImage_cut, cv::Mat &dstImage_cut);

int histRemoveHighlt(cv::Mat srcImage, float rate, cv::Mat &dstImage);

int getCamDistance(cv::Mat imagedepth, std::vector<cv::Point2f> featruePoint, float &distance);

int overExposureCounts(cv::Mat &srcImage, float thred, float* isOverExposure);