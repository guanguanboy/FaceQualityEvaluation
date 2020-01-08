#pragma once

#ifdef FACEQUALITYEVALUATION_EXPORTS
#define FACE_EVAL_API __declspec(dllexport)  
#else
#define FACE_EVAL_API __declspec(dllimport)  
#endif

#include "anchor_generator.h"
#include <opencv2\opencv.hpp>

FACE_EVAL_API int panny(int i, int(*call_back)(int a, int b));

FACE_EVAL_API Anchor detectMaxFace(cv::Mat irFrame);