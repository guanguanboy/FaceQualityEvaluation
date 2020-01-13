// FaceQualityEvaluation.cpp : 定义 DLL 应用程序的导出函数。
//

#include "stdafx.h"
#include "FaceQualityEvaluation.h"
#include <opencv2\opencv.hpp>
#include <opencv2\core\fast_math.hpp>
#include <vector>
#include "tools.h"
#include "face_pose_three_degree.h"
#include "imageProcess.h"
#include "illuminationDetect.h"
#include "blurDetect.h"
#include "facepreprocessengine.h"

using namespace cv;
using namespace std;

//宏定义区
#define MAX_DEPTH_VALUE 10000

//全局数据定义区
ncnn::Net g_facedectec_net;
FacePoseEstimate g_face_pose;

/*
内部函数定义区
*/
extern float getDeepFaceIntegrity(cv::Mat depthFrame, Anchor face);
extern float getDeepFaceMaxCCRatio(cv::Mat depthFrame, Anchor face);
extern void getDeepFaceLayerCountAndPrecision(cv::Mat depthFrame, Anchor face, int *pLayerCount, float *pPrecision);

int find_max_face_retinaFace(vector<Anchor> &result, Anchor &maxFace) {
	int max_index = 0;
	float max_area = 0.0;
	for (int i = 0; i < result.size(); ++i) {
		result[i].finalbox.width = result[i].finalbox.width - result[i].finalbox.x;
		result[i].finalbox.height = result[i].finalbox.height - result[i].finalbox.y;
		if (max_area < result[i].finalbox.area()) {
			max_area = result[i].finalbox.area();
			max_index = i;
		}
	}
	maxFace = result[max_index];
	return 0;
}


int load_modal(ncnn::Net &net, std::string model_path = "./models/retina") {
	std::string param_path = model_path + ".param";
	std::string bin_path = model_path + ".bin";
	net.load_param(param_path.data());
	net.load_model(bin_path.data());
	return 0;
}

int model_forward(ncnn::Mat &input, ncnn::Net &_net, std::vector<Anchor> &proposals) {
	input.substract_mean_normalize(pixel_mean, pixel_std);
	ncnn::Extractor _extractor = _net.create_extractor();
	_extractor.input("data", input);

	proposals.clear();

	vector<AnchorGenerator> ac(_feat_stride_fpn.size());
	for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
		int stride = _feat_stride_fpn[i];
		ac[i].Init(stride, anchor_cfg[stride], false);
	}

	for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
		ncnn::Mat cls;
		ncnn::Mat reg;
		ncnn::Mat pts;

		// get blob output
		char clsname[100]; sprintf_s(clsname, 100, "face_rpn_cls_prob_reshape_stride%d", _feat_stride_fpn[i]);
		char regname[100]; sprintf_s(regname, 100, "face_rpn_bbox_pred_stride%d", _feat_stride_fpn[i]);
		char ptsname[100]; sprintf_s(ptsname, 100, "face_rpn_landmark_pred_stride%d", _feat_stride_fpn[i]);
		_extractor.extract(clsname, cls);
		_extractor.extract(regname, reg);
		_extractor.extract(ptsname, pts);

		ac[i].FilterAnchor(cls, reg, pts, proposals);
	}
	return 0;
}

/*
如下为导出函数定义区
*/

/*
在初始化模型中加载nn模型
*/
void initFaceEvaluation(std::string face_dectect_model_path)
{
	//加载人脸检测模块
	//std::string model_path = "../models/face_detection_models/retina";
	load_modal(g_facedectec_net, face_dectect_model_path);

	// 加载人脸遮挡判断模块
	//PCANet pcaNet;
	//PCA_Train_Result* result = new PCA_Train_Result;
	//std::string svmpath = "../../../getFaceQuality/face_occlusion_model/svm.xml";
	//cv::Ptr<cv::ml::SVM> SVM;
	//SVM = cv::ml::StatModel::load<cv::ml::SVM>(svmpath);
	//getPCANetPara(result, pcaNet);
}

Anchor detectMaxFace(cv::Mat irFrame)
{
	cv::Rect2f maxFaceRect;
	Anchor maxFace;


	// 红外图像单通道转三通道
	Mat irMatc3[3], irDivideMat3;
	irMatc3[0] = irFrame.clone();
	irMatc3[1] = irFrame.clone();
	irMatc3[2] = irFrame.clone();
	cv::merge(irMatc3, 3, irDivideMat3);

	// 多尺度人脸检测
	ncnn::Mat input = ncnn::Mat::from_pixels_resize(irDivideMat3.data, ncnn::Mat::PIXEL_BGR2RGB, irDivideMat3.cols, irDivideMat3.rows, 640, 400);
	std::vector<Anchor> proposals;
	model_forward(input, g_facedectec_net, proposals);

	// NMS非极大值抑制
	std::vector<Anchor> NmsFacePara;
	nms_cpu(proposals, nms_threshold, NmsFacePara);

	if (NmsFacePara.size() != 0) {

		find_max_face_retinaFace(NmsFacePara, maxFace);
	}

	//maxFaceRect = maxFace.finalbox;

	return maxFace;
}


void getFacePose(cv::Mat irFrame, cv::Mat depthFrame, Anchor face, FacePosition_s *p_facepose)
{
	// 红外图像单通道转三通道
	Mat irMatc3[3], irDivideMat3;
	irMatc3[0] = irFrame.clone();
	irMatc3[1] = irFrame.clone();
	irMatc3[2] = irFrame.clone();
	cv::merge(irMatc3, 3, irDivideMat3);

	// 人脸姿态计算
	g_face_pose.updata(face, depthFrame, irDivideMat3);
	g_face_pose.calFacePose();
	float *pose = g_face_pose.getAngle();

	float Pitch = pose[0];
	float Yaw = pose[1];
	float Roll = pose[2];

	p_facepose->Pitch = Pitch;
	p_facepose->Yaw = Yaw;
	p_facepose->Roll = Roll;

	//定义一个结构体同时返回这三个值
	return;
}

//获取IR图像的亮度

//获取IR图像的模糊度

//在原始的实现中，将亮度和模糊度这些指标的计算都糅合到了如下一个函数中
void getFaceQuality(cv::Mat irFrame, cv::Mat imagedepth, Anchor face, FaceQuality *pfaceQuality)
{
	float blur, illumination;
	cv::Mat imgresize, imageprs, image_cut, imageprs_cut;

	cv::Mat image = irFrame(face.finalbox);

	double scale = 0.25;
	cv::Size dsize = cv::Size(image.cols*scale, image.rows*scale);
	cv::resize(image, imgresize, dsize, 0, 0, 1);

	//将检测出来的五个关键点的坐标转换到人脸区域的坐标中
	for (int i = 0; i < 5; i++)
	{
		face.pts[i].x -= face.finalbox.x;
		face.pts[i].y -= face.finalbox.y;
	}

	imageProcess(image, face.pts, imageprs, image_cut, imageprs_cut);
	illuminationDetect(image, imageprs_cut, illumination);
	float illumScale = (illumination - minILLMEAN) / (maxILLMEAN - minILLMEAN);
	pfaceQuality->illumQuality = illumScale;
	if (illumScale < 0)
	{
		pfaceQuality->illumQuality = 1;
		return;
	}

	float distance = 0;
	getCamDistance(imagedepth, face.pts, distance);
	blurDetect(distance, illumination, blur);
	pfaceQuality->blurQuality = blur;

	//获取人脸深度完整性
	pfaceQuality->depthFaceIntegrity = getDeepFaceIntegrity(imagedepth, face);

	//获取人脸最大连通域占比
	pfaceQuality->depthFaceMaxCcRatio = getDeepFaceMaxCCRatio(imagedepth, face);

	//获取人脸深度图层数及量化精度值
	int layerCount = 0;
	float precision = 0;

	getDeepFaceLayerCountAndPrecision(imagedepth, face, &layerCount, &precision);

	pfaceQuality->depthFaceLayerCount = layerCount;
	pfaceQuality->depthFacePrecision = precision;

	return;
}

/*
人脸深度完整性的计算
计算公式：
人脸区域非0像素点的个数/人脸区域总的像素点的个数
返回值取值范围：0--1
*/
float getDeepFaceIntegrity(cv::Mat depthFrame, Anchor face)
{
	unsigned int deepFacePixelCount = 0;
	unsigned int deepFaceNoneZeroPixelCount = 0;

	cv::Mat deepFace = depthFrame(face.finalbox);

	deepFacePixelCount = deepFace.cols * deepFace.rows;

	for (int i = 0; i < deepFace.rows; i++)
	{
		for (int j = 0; j < deepFace.cols; j++)
		{
			if (deepFace.at<ushort>(i, j) != 0)
			{
				deepFaceNoneZeroPixelCount++;
			}
		}
	}

	return (deepFaceNoneZeroPixelCount * 1.0) / deepFacePixelCount;
}

/*
以下函数用来求取连通域
计算公式：
最大连通域中的像素点个数/人脸区域总的像素点个数
返回值取值范围：0--1
*/
float getDeepFaceMaxCCRatio(cv::Mat depthFrame, Anchor face)
{
	unsigned int maxccpixelcount = 0;
	unsigned int deepFacePixelCount = 0;

	float ratio = 0;
	//将最大人脸区域的深度信息裁剪出来
	Mat  deepFace = depthFrame(face.finalbox);

	//从最大连通域的描述信息中，获取最大连通域中像素点的个数
	FacePreProcessEngine *pfacePreProcessEngine = new FacePreProcessEngine();
	maxccpixelcount = pfacePreProcessEngine->getBiggestCCPixelCountFromdepthFace(deepFace);

	//根据公式计算人脸最大连通域占比
	deepFacePixelCount = deepFace.cols * deepFace.rows;

	ratio = maxccpixelcount*1.0 / deepFacePixelCount;

	delete pfacePreProcessEngine;

	return ratio;

}


int getLayerCountAndMaxMinPixelValue(Mat& deepFace, int *p_maxPixelValue, int *p_minPixelValue)
{
	int initalized = false;

	//统计不同深度值的个数
	bool *phistogramArray = new bool[MAX_DEPTH_VALUE];
	int layerCount = 0;

	//初始化深度值是否出现的标记
	for (int i = 0; i < MAX_DEPTH_VALUE; i++)
	{
		phistogramArray[i] = false;
	}


	for (int i = 0; i < deepFace.rows; i++)
	{
		for (int j = 0; j < deepFace.cols; j++)
		{
			int pixelValue = deepFace.at<ushort>(i, j);

			if (pixelValue  == 0)
			{
				continue;
			}

			phistogramArray[pixelValue] = true;

			if (false == initalized)
			{
				*p_maxPixelValue = pixelValue;
				*p_minPixelValue = pixelValue;
				initalized = true;
				continue;
			}

			if (pixelValue > *p_maxPixelValue)
			{
				*p_maxPixelValue = pixelValue;
			}

			if (pixelValue < *p_minPixelValue)
			{
				*p_minPixelValue = pixelValue;
			}
		}
	}

	//计算深度的层数
	for (int i = 0; i < MAX_DEPTH_VALUE; i++)
	{
		if (true == phistogramArray[i])
		{
			layerCount++;
		}
	}

	delete phistogramArray;

	return layerCount;
}

void getDeepFaceLayerCountAndPrecision(cv::Mat depthFrame, Anchor face, int *pLayerCount, float *pPrecision)
{
	int layerCount = 0;
	int maxPixelValue;
	int minPixelValue;
	Mat  deepFace = depthFrame(face.finalbox);

	layerCount = getLayerCountAndMaxMinPixelValue(deepFace, &maxPixelValue, &minPixelValue);

	*pLayerCount = layerCount;

	*pPrecision = (maxPixelValue - minPixelValue) * 1.0 / (layerCount - 1);

	return;
}