#include <stdlib.h>
#include <opencv2\opencv.hpp>
#include "FaceQualityEvaluation.h"

using namespace std;
using namespace cv;

/*
如下程序主要完成对FaceQualityEvaluation.dll动态库的测试
*/
int main()
{
	//读取IR图片
	Mat irFrame = cv::imread("./testImages/ir_3.png", CV_LOAD_IMAGE_UNCHANGED);

	cv::Mat ir8BitMat;

	// 红外图像10位转8位
	irFrame.convertTo(ir8BitMat, CV_8UC1, 0.25);

	//注意后面算法需要8位的IR图进行处理

	//初始化算法模块
	std::string facedetect_model_path = std::string("./models/face_detection_models/retina");
	initFaceEvaluation(facedetect_model_path);

	//检测人脸
	cv::Rect2f maxface;

	Anchor anchorFace;
	anchorFace = detectMaxFace(ir8BitMat);

	maxface = anchorFace.finalbox;

	//输出结果
	std::cout << "max face x = " << maxface.x << std::endl;
	std::cout << "max face y = " << maxface.y << std::endl;
	std::cout << "max face width = " << maxface.width << std::endl;
	std::cout << "max face height = " << maxface.height << std::endl;

	//读取深度图
	Mat depthFrame = cv::imread("./testImages/depth_3.png", CV_LOAD_IMAGE_UNCHANGED);

	//获取人脸姿态
	FacePosition_s facepose;

	getFacePose(ir8BitMat, depthFrame, anchorFace, &facepose);

	std::cout << "Pitch = " << facepose.Pitch << std::endl;
	std::cout << "Yaw = " << facepose.Yaw << std::endl;
	std::cout << "Roll = " << facepose.Roll << std::endl;

	//获取人脸质量，亮度评价和模糊度评价
	FaceQuality faceQuality;
	memset(&faceQuality, 0, sizeof(FaceQuality));
	getFaceQuality(ir8BitMat, depthFrame, anchorFace, &faceQuality);

	cout << "Face illumination Quality = " << faceQuality.illumQuality << endl;
	cout << "Face Blur Quality = " << faceQuality.blurQuality << endl;
	cout << "Face Integrity = " << faceQuality.depthFaceIntegrity << endl;
	cout << "Face faceMacCcRatio = " << faceQuality.depthFaceMaxCcRatio << endl;
	cout << "Face layerCount = " << faceQuality.depthFaceLayerCount << endl;
	cout << "Face precision = " << faceQuality.depthFacePrecision << endl;

	system("pause");
	return 0;
}
