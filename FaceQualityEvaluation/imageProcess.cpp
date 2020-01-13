#include "stdafx.h"
#include "imageProcess.h"
#include "imageHistogram.h"
#include<algorithm>

int imageProcess0(cv::Mat srcImage, std::vector<cv::Point2f> featruePoint, cv::Mat &dstImage, cv::Mat &srcImage_cut, cv::Mat &dstImage_cut)
{
	assert(srcImage.channels == 1);
	if (srcImage.empty())
	{
		return -1;
	}
	dstImage = srcImage.clone();
	
	unsigned char  radius = 2;
	short tempx = 0;
	short tempy = 0;
	short width = srcImage.cols;
	short height = srcImage.rows;
	for (int i = 0; i < height; i++)
	{
	    for (int j = 0; j < width; j++)
	    {
	        std::vector<unsigned char> pxvalue;
	        for (int m = -radius; m <= radius; m++)
	        {
	            for (int n = -radius; n <= radius; n++)
	            {
	                tempx = i + m;
	                tempy = j + n;
	                tempx = tempx > 0 ? tempx : 0;
	                tempy = tempy > 0 ? tempy : 0;
	                tempx = tempx < height ? tempx : (height - 1);
	                tempy = tempy < width ? tempy : (width - 1);
	                pxvalue.push_back(dstImage.at<unsigned char>(tempx, tempy));
	            }
	        }
	        
	        sort(pxvalue.begin(), pxvalue.end());
	        unsigned char biggest = pxvalue[pxvalue.size()-1];
	        unsigned char smallest = pxvalue[0];
	        unsigned char median = pxvalue[pxvalue.size()/2];
	      
	        if (abs(biggest - smallest) > MDIFFVALUE)
	        {
	            for (int m = -radius; m <= radius; m++)
	            {
	                for (int n = -radius; n <= radius; n++)
	                {
	                    tempx = i + m;
	                    tempy = j + n;
	                    tempx = tempx > 0 ? tempx : 0;
	                    tempy = tempy > 0 ? tempy : 0;
	                    tempx = tempx < height ? tempx : (height - 1);
	                    tempy = tempy < width ? tempy : (width - 1);
	                    unsigned char res = abs(biggest - dstImage.at<unsigned char>(tempx, tempy));
	                    if (res < DIFFVALUE)
	                    {
	                        dstImage.at<unsigned char>(tempx, tempy) = median;
	                    }
	                }
	            }
	        }
	    }
	}

	int lx = (1.0 - WSCALE) / 2 * width + 1;
	int ly = (1.0 - HSCALE) / 2 * height + 1;
	int rectw = WSCALE * width;
	int recth = HSCALE * height;
	cv::Rect roiRect(lx,ly, rectw, recth);
	dstImage_cut = dstImage(roiRect);

	return 0;
}

int imageProcess(cv::Mat srcImage, std::vector<cv::Point2f> featruePoint,
	cv::Mat &dstImage, cv::Mat &srcImage_cut, cv::Mat &dstImage_cut)
{
	assert(srcImage.channels == 1);
	if (srcImage.empty())
	{
		return -1;
	}
	dstImage = srcImage.clone();
	histRemoveHighlt(srcImage, RATE, dstImage);

	cv::Rect roiRect = cv::boundingRect(featruePoint); //计算出这些点的最小包围矩形
	//用一个最小的矩形，把找到的形状包起来

	int w = roiRect.width * 3.0 / 2;
	int h = roiRect.height * 3.0 / 2;
	roiRect.x = (roiRect.x + roiRect.width / 2 - w / 2) > 0 ? (roiRect.x + roiRect.width / 2 - w / 2) : 0;
	roiRect.y = (roiRect.y + roiRect.height / 2 - h / 2) >0 ? (roiRect.y + roiRect.height / 2 - h / 2): 0;
	
	roiRect.width = (roiRect.x + w) < srcImage.cols ? w : (srcImage.cols - roiRect.x);
	roiRect.height = (roiRect.y + h) < srcImage.rows ? h : (srcImage.rows - roiRect.y);

	srcImage_cut = srcImage(roiRect);
	dstImage_cut = dstImage(roiRect);

	return 0;
}

int overExposureCounts(cv::Mat &srcImage, float thred, float* isOverExposure)
{
	assert(srcImage.channels == 1);
	if (srcImage.empty())
	{
		return -1;
	}
	cv::Mat srcImgBinary;
	cv::threshold(srcImage, srcImgBinary, 245, 255, cv::THRESH_TOZERO_INV);
	float iVal255 = cv::countNonZero(srcImgBinary);
	float FullPix = srcImage.cols * srcImage.rows;
	float ExpourseRate = (iVal255 / FullPix);

	*isOverExposure = ExpourseRate > thred ? 0 : 1;
	return 0;

}
int histRemoveHighlt(cv::Mat srcImage, float rate, cv::Mat &dstImage)
{
	assert(srcImage.channels == 1);
	if (srcImage.empty())
	{
		return -1;
	}

	Histogram1D histogram1D;
	cv::Mat hist = histogram1D.getHistogram(srcImage);
	int histSize = hist.rows;
	int allPixelNum = srcImage.rows * srcImage.cols;
	int normalNum = 0;
	int pixelVal;
	for (int h = 0; h < histSize; h++)
	{
		int binVal = hist.at<float>(h);
		if (binVal < 1)
			continue;
		normalNum = normalNum + binVal;
		float r = normalNum * 1.0 / allPixelNum;
		if (r > rate)
		{
			pixelVal = h;
			break;
		}
	}

	dstImage = srcImage.clone();

	if (pixelVal < MINPV)
	{
		return 0;
	}

	for (int i = 0; i < srcImage.rows; i++)
	{
		for (int j = 0; j < srcImage.cols; j++)
		{
			uchar pval = srcImage.at<unsigned char>(i, j);
			if (pval > pixelVal)
			{
				dstImage.at<unsigned char>(i, j) = pixelVal;
			}
		}
	}

	return 0;
}


int getCamDistance(cv::Mat imagedepth, std::vector<cv::Point2f> featruePoint, float &distance)
{
	distance = 0;
	assert(imagedepth.channels == 1);
	if (imagedepth.empty())
	{
		return -1;
	}
	
	cv::Rect roiRect = cv::boundingRect(featruePoint);
	//cv::Mat dstImage = imagedepth.clone();
	//cv::rectangle(dstImage, roiRect, cv::Scalar(255), 2, cv::LINE_8, 0);

	cv::Mat depthRoi = imagedepth(roiRect);
	int dnum = 0;
	float dsum = 0;
	for (int i = 0; i < depthRoi.rows; i++)
	{
		for (int j = 0; j < depthRoi.cols; j++)
		{
			ushort dval = depthRoi.at<ushort>(i, j);
			if (dval>MINDEPTH && dval<MAXDEPTH)
			{
				dsum = dsum + dval;
				dnum++;
			}
		}
	}

	if (dnum == 0)
	{
		distance = 0;
		return -1;
	}
	distance = dsum / dnum;

	return 0;
}