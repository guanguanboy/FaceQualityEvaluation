#pragma once

#include "util.h"
#include<cmath>


#define maxBlurScale 1.0
#define minBlurScale 0.0

#define maxDistance 1700//1400
#define minDistance 700//400

#define maxILLMean 180//200
#define minILLMean 20//10


int blurDetect(float distance, float illumination, float &blurPer);