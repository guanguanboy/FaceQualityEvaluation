#pragma once

#ifdef FACEQUALITYEVALUATION_EXPORTS
#define FACE_EVAL_API __declspec(dllexport)  
#else
#define FACE_EVAL_API __declspec(dllimport)  
#endif

#include <opencv2\opencv.hpp>
#include "anchor.h"

//结构体定义
typedef struct {
	float Pitch;
	float Yaw;
	float Roll;
}FacePosition_s;

struct FaceQuality {
	float illumQuality; //亮度评估，取值范围0-1，值越小越
	float blurQuality;  //模糊度评估，取值范围0-1，值越小越清晰
	float depthFaceIntegrity; //深度人脸完整性
	float depthFaceMaxCcRatio; //深度人脸最大连通域占比
	int depthFaceLayerCount; //深度人脸层数
	float depthFacePrecision; //深度人脸量化精度值
};

//对外接口定义
FACE_EVAL_API void initFaceEvaluation(std::string face_dectect_model_path);

FACE_EVAL_API Anchor detectMaxFace(cv::Mat irFrame);

FACE_EVAL_API void getFacePose(cv::Mat irFrame, cv::Mat depthFrame, Anchor face, FacePosition_s *p_facepose);

FACE_EVAL_API void getFaceQuality(cv::Mat irFrame, cv::Mat depthFrame, Anchor face, FaceQuality *p_facequality);

//FACE_EVAL_API float getDeepFaceIntegrity(cv::Mat depthFrame, Anchor face);
//
//FACE_EVAL_API float getDeepFaceMaxCCRatio(cv::Mat depthFrame, Anchor face);
//
//FACE_EVAL_API void getDeepFaceLayerCountAndPrecision(cv::Mat depthFrame, Anchor face, int *pLayerCount, float *pPrecision);