#include "stdafx.h"
#include "facepreprocessengine.h"
#include <queue>
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;

typedef struct ConnectedComponentDescription
{
	int ccType; //连通域的类型 关于连通区域类别的说明：第一个连通区域的类别为1，第二个连通区域的类别为2.依次类推
	int ccPixelCount; //连通域内像素个数
	double ccPixelMeanValue; //连通域内像素的平均值
}ConnectedComponentDes;

FacePreProcessEngine::FacePreProcessEngine()
	:backgroundRemovalThreshold(3000), depthSimilarityThreshold(0.3), validFaceAreaRatio(0.8)
{

}

FacePreProcessEngine::FacePreProcessEngine(unsigned int bgThreshold /* = 3000 */, double depthSimThreshold /* = 0.3 */, double ratio /* = 0.8 */)
	: backgroundRemovalThreshold(3000), depthSimilarityThreshold(0.3), validFaceAreaRatio(0.8)
{
	init(bgThreshold, depthSimThreshold, ratio);
}

FacePreProcessEngine::~FacePreProcessEngine()
{

}

void FacePreProcessEngine::init(unsigned int bgThreshold /* = 3000 */, double depthSimThreshold /* = 0.3 */, double ratio /* = 0.8 */)
{
	backgroundRemovalThreshold = bgThreshold;
	depthSimilarityThreshold = depthSimThreshold;
	validFaceAreaRatio = ratio;


	return;
}

void FacePreProcessEngine::proceed(Mat& originDepthImg, vector<Rect>& faceBoxes, cv::Mat& outputDepthImg)
{
	Rect biggestFaceBox;
	Mat  croppedBiggestFace;
	Mat  clonedDepthFace;
	vector<ConnectedComponentDes> ccList;
	vector<vector<Point> > allCcPointsLists;

	findBiggiestFaceBox(faceBoxes, biggestFaceBox);

	croppedBiggestFace = originDepthImg(biggestFaceBox); //将最大人脸区域的深度信息裁剪出来

	imshow("cropped depth image", croppedBiggestFace);

	clonedDepthFace = croppedBiggestFace.clone();

	normalizeCroppedFace(clonedDepthFace);

	//ccList存储的是找到的所有连通域的描述信息，allCcPointsLists存储的是联通域中所有点的坐标
	findAllConnectedComponment(clonedDepthFace, ccList, allCcPointsLists);

	//提取最大连通域，其余像素值赋值为0

	extractFaceArea(ccList, allCcPointsLists, croppedBiggestFace, outputDepthImg);

	renormalizeDepthFace(outputDepthImg);

	return;
}

unsigned int FacePreProcessEngine::getBgRemovalThreshold()
{
	return backgroundRemovalThreshold;
}

double FacePreProcessEngine::getDepthSimilarityThreshold()
{
	return depthSimilarityThreshold;
}

double FacePreProcessEngine::getValidFaceAreaRatio()
{
	return validFaceAreaRatio;
}

void FacePreProcessEngine::setBgRemovalThreshold(unsigned int threshold)
{
	backgroundRemovalThreshold = threshold;

	return;
}

void FacePreProcessEngine::setDepthSimilarityThreshold(double threshold)
{
	depthSimilarityThreshold = threshold;

	return;
}

void FacePreProcessEngine::setValidFaceAreaRatio(double ratio)
{
	validFaceAreaRatio = ratio;

	return;
}

void FacePreProcessEngine::findBiggiestFaceBox(vector<Rect>& faceBoxes, Rect& biggestFaceBox)
{
	unsigned int areaOfBiggestFaceBox = 0;
	unsigned int areaOfcurrentFaceBox;
	vector<Rect>::iterator iter;

	for (iter = faceBoxes.begin(); iter != faceBoxes.end(); iter++)
	{
		areaOfcurrentFaceBox = (*iter).area();

		if (areaOfcurrentFaceBox > areaOfBiggestFaceBox)
		{
			areaOfBiggestFaceBox = areaOfcurrentFaceBox;
			biggestFaceBox = *iter;
		}
	}

	return;
}

void FacePreProcessEngine::setRemovedPixelValueToZero(Mat& img)
{
	int rowsNum = img.rows;
	int colsNum = img.cols * img.channels();

	for (int i = 0; i < rowsNum; i++)
	{
		float* rowdata = img.ptr<float>(i);

		for (int j = 0; j < colsNum; j++)
		{
			if (rowdata[j] > backgroundRemovalThreshold)
			{
				rowdata[j] = 0;
			}
		}
	}

	return;
}

void FacePreProcessEngine::findMaxAndMinPixelValue(Mat& img, int *p_maxPixelValue, int *p_minPixelValue)
{

	int rowsNum = img.rows;
	int colsNum = img.cols * img.channels();
	int initalized = false;

	for (int i = 0; i < rowsNum; i++)
	{
		float* rowdata = img.ptr<float>(i);

		for (int j = 0; j < colsNum; j++)
		{
			/* 由于图像经过setPixelValueBiggerThanThreadholdToZero处理后，部分元素
			的像素点值为0；所以需将这部分元素忽略，否则最小值肯定是0 */
			if (0 == rowdata[j])
			{
				continue;
			}

			if (false == initalized)
			{
				*p_maxPixelValue = rowdata[j];
				*p_minPixelValue = rowdata[j];
				initalized = true;
				continue;
			}

			if (rowdata[j] > *p_maxPixelValue)
			{
				*p_maxPixelValue = rowdata[j];
			}

			if (rowdata[j] < *p_minPixelValue)
			{
				*p_minPixelValue = rowdata[j];
			}
		}
	}

	return;
}

void FacePreProcessEngine::normalizeFaceImg(Mat& img, int maxPixelValue, int minPixelValue, int valSetToPixelValEquZero)
{
	int rowsNum = img.rows;
	int colsNum = img.cols * img.channels();
	int range = maxPixelValue - minPixelValue;

	for (int i = 0; i < rowsNum; i++)
	{

		float* rowdata = img.ptr<float>(i);

		for (int j = 0; j < colsNum; j++)
		{
			//这里也不处理值为0的像素点
			if (0 == rowdata[j])
			{
				//完成matlab源码中的norm_face(sub2ind(size(norm_face), m, n))=100;逻辑
				rowdata[j] = valSetToPixelValEquZero;
				continue;
			}

			rowdata[j] = (float)(rowdata[j] - minPixelValue) / range;
		}
	}

	return;
}

void FacePreProcessEngine::normalizeCroppedFace(Mat& croppedFace)
{
	int maxPixelValue;
	int minPixelValue;

	//将Mat 转换为浮点型Mat
	croppedFace.convertTo(croppedFace, CV_32FC1);

	setRemovedPixelValueToZero(croppedFace);

	findMaxAndMinPixelValue(croppedFace, &maxPixelValue, &minPixelValue);

	normalizeFaceImg(croppedFace, maxPixelValue, minPixelValue, 100);
	//完成matlab源码中的norm_face(sub2ind(size(norm_face), m, n))=100;逻辑

	/*% Set the background value to 1
	[m, n] = find(depth_crop == Inf);
	norm_face(sub2ind(size(norm_face), m, n)) = 100;*/
	/*
	Matlab源码中这里要完成的事情其实是：找到depth_crop数组中值为0的点的坐标，然后将norm_face中
	相同坐标上的值设置为100。
	*/
	return;
}

bool FacePreProcessEngine::isTwoPointsConnected(double checkedPonitPixelValue, double currentCcTypePixelMeanValue)
{
	double differenceValue = checkedPonitPixelValue - currentCcTypePixelMeanValue;
	double pointDistance;

	pointDistance = sqrt(differenceValue * differenceValue);

	if (pointDistance <= depthSimilarityThreshold)
	{
		return true;
	}

	return false;
}

double FacePreProcessEngine::updateCurrentPixelMeanValue(Mat& inputNormFace, vector<Point> & curentCcPointList)
{
	double totalPixelValue = 0;
	int curentCcSize = curentCcPointList.size();

	for (Point currentPoint : curentCcPointList)
	{
		totalPixelValue += inputNormFace.at<float>(currentPoint);
	}

	return (totalPixelValue / curentCcSize);
}

void FacePreProcessEngine::updateInputNormFace(vector<Point> &curentCc, double currentCcTypePixelMeanValue, Mat& inputNormFace)
{
	for (Point currentPoint : curentCc)
	{
		inputNormFace.at<float>(currentPoint) = currentCcTypePixelMeanValue;
	}

	return;
}


void FacePreProcessEngine::findAllConnectedComponment(Mat& inputNormFace, vector<ConnectedComponentDes>& ccList, vector<vector<Point> >& allCcPointsLists)
{
	/*
	关于连通区域类别的说明：
	第一个连通区域的类别为1，第二个连通区域的类别为2.依次类推

	关于连通区域的算法说明：
	1，遍历从第二行第二列的像素点开始
	*/

	int currentCcType = 1;	//将当前连通域的类别初始化为1

	double currentCcTypePixelMeanValue; // = inputNormFace.at<float>(1, 1); //由于opencv中mat结构的数组下标是从0开始的，所以如果要获取第二行第二列的元素，at的行列索引为1，1

	const vector<Point> directions = { Point(-1,0), Point(0, 1), Point(1, 0), Point(0, -1) }; //分别表示上、右、下、左四个方向
	Mat classMatrix = Mat::zeros(inputNormFace.size(), CV_8UC1);
	//classMatrix.at<uchar>(1, 1) = firstCcClassPixelValue;

	queue<Point> currentPoints;
	Point currentPoint;

	Mat visited = Mat::zeros(inputNormFace.size(), CV_8UC1);
	int faceCols = inputNormFace.cols;
	int faceRows = inputNormFace.rows;

	int faceColsMaxIndex = faceCols - 1;
	int faceRowsMaxIndex = faceRows - 1;

	double currentCcTypeTotalPixelValue;
	unsigned currentCcTypePixelCount;

	//创建一个vector用来保存当前连通域中的元素，以便更新均值
	vector<Point> curentCcPointsList;
	//set<Point> uniqueCcPointsList;
	ConnectedComponentDes currentCcDes;

	for (int i = 1; i < faceRowsMaxIndex; i++)
	{
		for (int j = 1; j < faceColsMaxIndex; j++)
		{
			if (true == visited.at<uchar>(Point(i, j)))
			{
				continue;
			}

			currentPoints.push(Point(i, j));
			curentCcPointsList.push_back(Point(i, j));
			//uniqueCcPointsList.insert(Point(i, j));

			//初始化当前均值
			currentCcTypePixelMeanValue = inputNormFace.at<float>(Point(i, j));
			currentCcTypeTotalPixelValue = currentCcTypePixelMeanValue;
			currentCcTypePixelCount = 1;

			while (true != currentPoints.empty())
			{
				currentPoint = currentPoints.front();
				currentPoints.pop();

				if (true == visited.at<uchar>(currentPoint))
				{
					continue;
				}

				visited.at<uchar>(currentPoint) = true;

				classMatrix.at<uchar>(currentPoint) = currentCcType;

				for (Point dir : directions)
				{
					int x = currentPoint.x + dir.x;//{ Point(-1,0), Point(0, 1), Point(1, 0), Point(0, -1) }; //分别表示上、右、下、左四个方向
					int y = currentPoint.y + dir.y;

					///判断x 和 y坐标是否在边界上(或者边界之外)，如果在则不处理
					if (x < 1 || x > faceRowsMaxIndex - 1 || y < 1 || y > faceColsMaxIndex - 1)
					{
						continue;
					}

					if (true == visited.at<uchar>(Point(x, y)))
					{
						continue;
					}

					//判断一下当前点与（x，y）是否是连通的，如果不是连通的，则不处理该点
					double checkedPonitPixelValue = inputNormFace.at<float>(Point(x, y));

					if (true == isTwoPointsConnected(checkedPonitPixelValue, currentCcTypePixelMeanValue))
					{
						currentPoints.push(Point(x, y));

						curentCcPointsList.push_back(Point(x, y));
						//uniqueCcPointsList.insert(Point(x, y));

						currentCcTypeTotalPixelValue += checkedPonitPixelValue;
						currentCcTypePixelCount++;
					}
				}

				currentCcTypePixelMeanValue = currentCcTypeTotalPixelValue / currentCcTypePixelCount;

			}//end of while

			updateInputNormFace(curentCcPointsList, currentCcTypePixelMeanValue, inputNormFace);

			currentCcDes.ccType = currentCcType;
			currentCcDes.ccPixelMeanValue = currentCcTypePixelMeanValue;
			currentCcDes.ccPixelCount = curentCcPointsList.size();
			//currentCcDes.ccPixelCount = uniqueCcPointsList.size();

			ccList.push_back(currentCcDes);

			allCcPointsLists.push_back(curentCcPointsList);

			if (true != curentCcPointsList.empty())
			{
				curentCcPointsList.clear();
			}

			//while循环结束之后，说明已经找到了一个连通域，这个时候就得找下一个连通域了。所以需要将连通域的类型标识加1.
			currentCcType++;
		}
	}

	return;
}

bool FacePreProcessEngine::comparePixelValue(const ConnectedComponentDes& first, const ConnectedComponentDes& second)
{
	if (first.ccPixelMeanValue < second.ccPixelMeanValue)
	{
		return true;
	}
	else
	{
		return false;
	}
}

void FacePreProcessEngine::extractFaceArea(vector<ConnectedComponentDes>& ccList, vector<vector<Point> >& allCcPointsLists, Mat& OriginDepthFace, Mat& outputDepthFace)
{
	/*
	1, 需要知道最大连通域内所有像素的坐标。
	2，需要计算每个连通域内的像素的个数。
	3，需要知道每个连通域内的的像素值（连通域内的像素值是相等的）
	4，需要对连通域内的像素值进行排序
	5，需要去掉像素值比较大的那些连通域，并在剩余连通域中找出最大连通域
	*/

	/* 将ccList 按照连通域的平均深度值从小到大排序 */
	sort(ccList.begin(), ccList.end(), comparePixelValue);

	int ccCount = ccList.size();

	int retainedCcCount = floor(ccCount * validFaceAreaRatio);

	//删除后面ccList中后面的几个值
	ccList.erase(ccList.begin() + retainedCcCount, ccList.end());

	//找出最大像素个数的连通域
	vector<ConnectedComponentDes>::iterator iter;
	double maxPixelConut = 0;
	ConnectedComponentDes maxPixelValueCc;

	for (iter = ccList.begin(); iter != ccList.end(); iter++)
	{
		if (maxPixelConut < iter->ccPixelCount)
		{
			maxPixelConut = iter->ccPixelCount;
			maxPixelValueCc = *iter;
		}
	}

	vector<Point>& maxPixelValueCcPointList = allCcPointsLists[maxPixelValueCc.ccType - 1];
	vector<Point>::iterator pointsIter;

	//遍历全图，将原深度图中在最大连通域中的像素赋值到outputDepthFace
	for (pointsIter = maxPixelValueCcPointList.begin(); pointsIter != maxPixelValueCcPointList.end(); pointsIter++)
	{
		outputDepthFace.at<ushort>(*pointsIter) = OriginDepthFace.at<ushort>(*pointsIter);
	}

	return;
}

void FacePreProcessEngine::renormalizeDepthFace(Mat& depthFace)
{
	int maxPixelValue;
	int minPixelValue;

	//将Mat 转换为浮点型Mat
	depthFace.convertTo(depthFace, CV_32FC1);

	findMaxAndMinPixelValue(depthFace, &maxPixelValue, &minPixelValue);

	normalizeFaceImg(depthFace, maxPixelValue, minPixelValue, 1);

	return;
}

unsigned int FacePreProcessEngine::getBiggestCCPixelCountFromdepthFace(Mat& depthface)
{
	//计算深度图人脸区域中所有连通域
	vector<ConnectedComponentDes> ccList;
	vector<vector<Point> > allCcPointsLists;

	Mat  clonedDepthFace;
	unsigned int BiggestCCPixelCount = 0;
	clonedDepthFace = depthface.clone();
	Mat outputImg;
	outputImg = Mat::zeros(Size(depthface.cols, depthface.rows), depthface.type());
	//找到最大的连通域
	normalizeCroppedFace(clonedDepthFace);

	findAllConnectedComponment(clonedDepthFace, ccList, allCcPointsLists);

	extractFaceArea(ccList, allCcPointsLists, depthface, outputImg);

	for (int i = 0; i < outputImg.rows; i++)
	{
		for (int j = 0; j < outputImg.cols; j++)
		{
			if (outputImg.at<uchar>(i, j) != 0)
			{
				BiggestCCPixelCount++;
			}
		}
	}
	//BiggestCCPixelCount = getBiggestCCPixelCountFromCC(ccList, allCcPointsLists);

	return BiggestCCPixelCount;
}

unsigned int FacePreProcessEngine::getBiggestCCPixelCountFromCC(vector<ConnectedComponentDes>& ccList, vector<vector<Point> >& allCcPointsLists)
{
	/* 将ccList 按照连通域的平均深度值从小到大排序 */
	sort(ccList.begin(), ccList.end(), comparePixelValue);

	int ccCount = ccList.size();

	int retainedCcCount = floor(ccCount * validFaceAreaRatio);

	//删除后面ccList中后面的几个值
	ccList.erase(ccList.begin() + retainedCcCount, ccList.end());

	//找出最大像素个数的连通域
	vector<ConnectedComponentDes>::iterator iter;
	double maxPixelConut = 0;
	ConnectedComponentDes maxPixelValueCc;

	for (iter = ccList.begin(); iter != ccList.end(); iter++)
	{
		if (maxPixelConut < iter->ccPixelCount)
		{
			maxPixelConut = iter->ccPixelCount;
			maxPixelValueCc = *iter;
		}
	}

	return maxPixelConut;
}