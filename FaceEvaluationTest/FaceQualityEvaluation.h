#pragma once

#ifdef FACEQUALITYEVALUATION_EXPORTS
#define FACE_EVAL_API __declspec(dllexport)  
#else
#define FACE_EVAL_API __declspec(dllimport)  
#endif

#include <opencv2\opencv.hpp>
#include "anchor.h"

FACE_EVAL_API int panny(int i, int(*call_back)(int a, int b));

FACE_EVAL_API Anchor detectMaxFace(cv::Mat irFrame);

FACE_EVAL_API void initFaceEvaluation(std::string face_dectect_model_path);
