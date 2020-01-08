//
// Created by hubenchuan on 2019/11/19.
//

#ifndef FACE_OFFLINE_SDK_ANDROID_FACE_POSE_H
#define FACE_OFFLINE_SDK_ANDROID_FACE_POSE_H
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "anchor_generator.h"

#define INTRINSIC_LENGTH (3)
#define NOSEINDEX 2       //鼻尖索引
#define NEIGHBOR  7       //稳定深度值计算领域的大小
#define FARDISTANCE  850  // 人脸有效范围最远距离
#define NEARDISTANCE 400  // 人脸有效范围最近距离
#define PITCHANGLE 17     // pitch 俯仰角阈值
#define YAWANGLE   45     // yaw   偏航角阈值
#define ROLLANGLE  45     // roll  横滚角阈值
#define DELTAANGLE 3      // 默认人脸正向pitch为3度

enum FaceState {
	NORMALFACE = 1,    //正常的可以识别的脸
	SMALLFACE = 2,     //脸部很小
	SIDEFACE = 3,      //侧脸，偏转过大
	ERRORDEPTH = 4,    //深度值发生异常错误,姿态计算异常值(深度图错误)
	TOONEAR = 5,       //人脸距离太近
	TOOFAR = 6,        //人脸距离太远
	FACERIGTH = 7,     //人脸太右
	FACETOP = 8,       //人脸太上
	FACEBOTTOM = 9,    //人脸太下
	FACELEFT = 10,     //人脸太左
};

struct keyPoint
{
	float x;
	float y;
	float z;
};

class FacePoseEstimate {

public:
	FacePoseEstimate();
	~FacePoseEstimate();
	FaceState face_state = NORMALFACE;
	int flag = 0;                               // 0:输入只有RGB，没有face_state判断；1:输入包含RGB和depth，包含face_state判断

	int updata(Anchor &maxFace, cv::Mat &image_depth, cv::Mat &image_rgb); //更新每一帧的检测结果
	int updata(Anchor &maxFace, cv::Mat &image_rgb); //更新每一帧的检测结果
	int calFacePose();

	Anchor getAlignFaceInfo();
	cv::Mat getAlignFace();
	cv::Mat getAlignFaceSquare();
	float *getAngle();
	int getFaceDistance();
	Anchor  getFaceInfo();

private:
	int face_distance;                           // 人脸距离
	std::vector<cv::Point3f> robust_points;      // 稳定的关键点
	double cameraIntrinsic[3][3] = { 0.0 };      // 相机内参
	cv::Mat face_region_square;                  // 旋转对齐后的人脸扩充为方形，用于人脸识别
	cv::Mat face_region;                         // 旋转对齐后的人脸，紧致的人脸区域，用于活体检测
	cv::Mat depth_image;                         // 深度图像
	cv::Mat rgb_image;                           // 彩色图像
	Anchor face_info;                            // 检测出的最大的人脸信息
	float angle[3] = { -1.0, -1.0, -1.0 };                    // 计算出人脸的偏转角度 pitch(俯仰角) yaw(偏航角) roll(翻滚角)

	int effeciteRange();                         // 判断人脸是否在规定的有效范围内
	int completeFace();                          // 判断检测的人脸框是否完全在图像规定范围内
	int standardKeypoints();                     // 根据五个landmarks的坐标和深度值初步判断是否满足要求
	int depth2point3d(cv::Point3f &point);       // 深度值根据内参转换为3D点
	int robustKeypoints();                       // 获取稳定的至多4个landmarks
	int panelFitting();                          // 平面拟合估计人脸pose 
	int panelFittingRGB();                       // 只有RGB图像,只计算图像的roll 
	int OLS_Plane(std::vector<cv::Point3f> point, double *param_plane);     //拟合平面方程																		 
	int rotateArbitrarilyAngle(cv::Mat &src);    // 图像旋转	
	int pointsRotate(float *points, float edge);                            // 特征点旋转
	int alignFaceFun();                                                     // 人脸信息旋转
};



#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#endif //ORBBEC_FACE_OFFLINE_SDK_ANDROID_FACE_POSE_H