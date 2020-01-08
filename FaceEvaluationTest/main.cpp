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

	//初始化算法模块
	std::string facedetect_model_path = std::string("./models/face_detection_models/retina");
	initFaceEvaluation(facedetect_model_path);

	//检测人脸
	cv::Rect2f maxface;

	Anchor anchorFace;
	anchorFace = detectMaxFace(irFrame);

	maxface = anchorFace.finalbox;

	//输出结果
	std::cout << "max face x = " << maxface.x << std::endl;
	std::cout << "max face y = " << maxface.y << std::endl;
	std::cout << "max face width = " << maxface.width << std::endl;
	std::cout << "max face height = " << maxface.height << std::endl;

	system("pause");
	return 0;
}
