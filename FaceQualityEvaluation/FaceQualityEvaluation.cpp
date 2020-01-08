// FaceQualityEvaluation.cpp : 定义 DLL 应用程序的导出函数。
//

#include "stdafx.h"
#include "FaceQualityEvaluation.h"
#include <opencv2\opencv.hpp>

int panny(int i, int(*call_back)(int a, int b))
{
	int aa;
	aa = i*i;
	call_back(i, aa);
	return 0;
}

Anchor detectMaxFace(cv::Mat irFrame)
{
	Anchor maxFace;

	return maxFace;
}