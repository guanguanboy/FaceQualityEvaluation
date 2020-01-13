#pragma once

#include "util.h"
#include "opencv.hpp"  
#include "opencv2/core/core.hpp"  
#include "opencv2/highgui/highgui.hpp"   
#include <opencv2/imgproc/imgproc.hpp>    
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui_c.h>



#define maxILLMEAN 180//200
#define minILLMEAN 20//10

#define maxGamma 1.3//1.0
#define minGamma 0.3


int illuminationDetect(cv::Mat srcImage, cv::Mat image_cut, float &illumination);

int gammaCorrection(cv::Mat srcImage, cv::Mat &image_gamma, float gamma);