// FaceQualityEvaluation.cpp : 定义 DLL 应用程序的导出函数。
//

#include "stdafx.h"
#include "FaceQualityEvaluation.h"
#include <opencv2\opencv.hpp>
#include <opencv2\core\fast_math.hpp>
#include <vector>
#include "tools.h"
#include "face_pose_three_degree.h"

using namespace cv;
using namespace std;

//全局数据定义区
ncnn::Net g_facedectec_net;
FacePoseEstimate g_face_pose;

int panny(int i, int(*call_back)(int a, int b))
{
	int aa;
	aa = i*i;
	call_back(i, aa);
	return 0;
}

/*
内部函数定义区
*/
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
	cv::Mat irDivideMat;

	// 红外图像10位转8位
	irFrame.convertTo(irDivideMat, CV_8UC1, 0.25);

	// 红外图像单通道转三通道
	Mat irMatc3[3], irDivideMat3;
	irMatc3[0] = irDivideMat.clone();
	irMatc3[1] = irDivideMat.clone();
	irMatc3[2] = irDivideMat.clone();
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


void calcFacePose(Anchor face, cv::Mat depthFrame, cv::Mat irFrame)
{

	cv::Mat irDivideMat;

	// 红外图像10位转8位
	irFrame.convertTo(irDivideMat, CV_8UC1, 0.25);

	// 红外图像单通道转三通道
	Mat irMatc3[3], irDivideMat3;
	irMatc3[0] = irDivideMat.clone();
	irMatc3[1] = irDivideMat.clone();
	irMatc3[2] = irDivideMat.clone();
	cv::merge(irMatc3, 3, irDivideMat3);

	// 人脸姿态计算
	g_face_pose.updata(face, depthFrame, irDivideMat3);
	g_face_pose.calFacePose();
	float *pose = g_face_pose.getAngle();

	float Pitch = pose[0];
	float Yaw = pose[1];
	float Roll = pose[2];

	std::cout << "Pitch = " << Pitch << std::endl;
	std::cout << "Yaw = " << Yaw << std::endl;
	std::cout << "Roll = " << Roll << std::endl;
}

//获取IR图像的亮度

//获取IR图像的模糊度
