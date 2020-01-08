#include "stdafx.h"
#include "face_pose_three_degree.h"
#include "opencv2/opencv.hpp"

FacePoseEstimate::FacePoseEstimate() {
	cameraIntrinsic[0][0] = 517.679;   //fx
	cameraIntrinsic[1][1] = 517.679;   //fy
	cameraIntrinsic[0][2] = 320;       //cx
	cameraIntrinsic[1][2] = 240.5;     //cy
}

FacePoseEstimate::~FacePoseEstimate() {}

int FacePoseEstimate::updata(Anchor &maxFace, cv::Mat &image_depth, cv::Mat &image_rgb) {
	face_info = maxFace;
	depth_image = image_depth;
	rgb_image = image_rgb;

	robust_points.clear();
	face_distance = 0;

	angle[0] = -1.0;
	angle[1] = -1.0;
	angle[2] = -1.0;

	face_state = NORMALFACE;

	return 0;
}

int FacePoseEstimate::updata(Anchor &maxFace, cv::Mat &image_rgb) {
	face_region.release();
	face_region_square.release();

	face_info = maxFace;
	rgb_image = image_rgb;
	
	angle[0] = 0.0;
	angle[1] = 0.0;
	angle[2] = 0.0;

	return 0;
}

Anchor FacePoseEstimate::getAlignFaceInfo() {
	return face_info;
}

float* FacePoseEstimate::getAngle() {
	return angle;
}

cv::Mat FacePoseEstimate::getAlignFaceSquare() {
	return face_region_square;
}

cv::Mat FacePoseEstimate::getAlignFace() {
	return face_region;
}

int FacePoseEstimate::getFaceDistance(){
	return face_distance;
}

Anchor  FacePoseEstimate::getFaceInfo() {
	return face_info;
}

int FacePoseEstimate::effeciteRange() {
	int distance_statistical[250] = { 0 };

	cv::Mat face_depth = depth_image(face_info.finalbox);
	for (int i = 0; i<face_depth.rows; i++)
	{
		const char16_t* inData = face_depth.ptr<char16_t>(i);
		for (int i = 0; i < face_depth.cols; i++) {
			int index = inData[i];
			if (2499 < index) index = 2499;
			index = index / 10;
			distance_statistical[index]++;
		}
	}
	for (int i = 0; i < 250; i++) {
		if (distance_statistical[face_distance] < distance_statistical[i])
			face_distance = i;
	}
	face_distance = face_distance * 10 + 5;

	if (face_distance < NEARDISTANCE) face_state = TOONEAR;
	if (face_distance > FARDISTANCE)  face_state = TOOFAR;
	return 0;
}

int FacePoseEstimate::depth2point3d(cv::Point3f &point) {

	double fdx = cameraIntrinsic[0][0];
	double fdy = cameraIntrinsic[1][1];
	double u0 = cameraIntrinsic[0][2];
	double v0 = cameraIntrinsic[1][2];

	point.x = (point.z * (point.x - u0)) / fdx;
	point.y = (point.z * (point.y - v0)) / fdy;

	return 0;
}

int FacePoseEstimate::completeFace() {
	if (face_info.finalbox.x < 0)
		face_state = FACERIGTH;
	if (face_info.finalbox.y < 0)
		face_state = FACETOP;
	if ((face_info.finalbox.x + face_info.finalbox.width) > rgb_image.cols)
		face_state = FACELEFT;
	if ((face_info.finalbox.y + face_info.finalbox.height) > rgb_image.rows)
		face_state = FACEBOTTOM;
	return 0;
}

int FacePoseEstimate::standardKeypoints() {
	for (int i = 0; i < 5; i++) {
		if (face_info.pts[i].x < 0) {
			face_state = FACERIGTH;
			return 0;}
		if (face_info.pts[i].y < 0) {
			face_state = FACETOP;
			return 0;}
		if (face_info.pts[i].x > 639) {
			face_state = FACELEFT;
			return 0;}
		if (face_info.pts[i].y > 399) {
			face_state = FACEBOTTOM;
			return 0;}
	}
	cv::Point2f middle_eyes = (face_info.pts[0] + face_info.pts[1]) / 2;
	cv::Point2f middle_mouth = (face_info.pts[3] + face_info.pts[4]) / 2;
	cv::Point2f eyes_to_mouth = middle_eyes - middle_mouth;

	// 两眼之间的距离大于18个像素
	cv::Point2f eye_l_to_r = face_info.pts[0] - face_info.pts[1];
	float distance_eye_l_to_r = sqrtf(pow(eye_l_to_r.x, 2) + pow(eye_l_to_r.y, 2));
	if (18 > distance_eye_l_to_r) { face_state = SMALLFACE; }

	// 嘴角之间的距离大于15个像素
	cv::Point2f mouth_l_to_r = face_info.pts[3] - face_info.pts[4];
	float distance_mouth_l_to_r = sqrtf(pow(mouth_l_to_r.x, 2) + pow(mouth_l_to_r.y, 2));
	if (15 > distance_mouth_l_to_r) { face_state = SMALLFACE; }

	// 两眼中心到嘴角中心的距离大于20个像素
	float distance_eyes_to_mouth = sqrtf(pow(eyes_to_mouth.x, 2) + pow(eyes_to_mouth.y, 2));
	if (20 > distance_eyes_to_mouth) { face_state = SMALLFACE; }

	// 两眼之间的距离与眼嘴之间的距离满足一定的比例条件
	if ((3 < (distance_eye_l_to_r / distance_eyes_to_mouth)) || (1 / 3 > (distance_eye_l_to_r / distance_eyes_to_mouth)))
	{
		face_state = SIDEFACE;
		return 0;
	}

	// 嘴角之间的距离与眼嘴之间的距离满足一定的比例条件
	if ((3 < (distance_mouth_l_to_r / distance_eyes_to_mouth)) || (1 / 3 > (distance_mouth_l_to_r / distance_eyes_to_mouth)))
	{
		face_state = SIDEFACE;
		return 0;
	}

	return 0;
}

int FacePoseEstimate::robustKeypoints() {
	float top    = 9999;
	float bottom = 0;
	float left   = 9999;
	float right  = 0;
	float depth_min = 9999;
	float depth_max = 0;

	robust_points.clear();
	for (int i = 0; i < face_info.pts.size(); ++i) {
		if (NOSEINDEX == i)	continue;
		if (face_info.pts[i].y < top)    top = face_info.pts[i].y;
		if (face_info.pts[i].y > bottom) bottom = face_info.pts[i].y;
		if (face_info.pts[i].x < left)   left = face_info.pts[i].x;
		if (face_info.pts[i].x > right)  right  = face_info.pts[i].x;

		// 计算稳定的深度值
		int edge = (NEIGHBOR - 1) / 2;
		float depth_value = 0.0;
		int count_depth = 0;
		for (int j = 0; j < NEIGHBOR; ++j) {
			for (int k = 0; k < NEIGHBOR; ++k) {
				float depth = depth_image.at<char16_t>(cv::Point((int)(face_info.pts[i].x - edge + j), (int)(face_info.pts[i].y - edge + k)));
				if (100 < abs(depth - face_distance)) continue;
				depth_value += depth;
				count_depth++;
			}
		}
		if (0 == count_depth)
			continue;	

		cv::Point3f robust_point = cv::Point3f(face_info.pts[i].x, face_info.pts[i].y, (depth_value / count_depth));
		depth2point3d(robust_point);
		robust_points.push_back(robust_point);

		if (robust_point.z > depth_max) depth_max = robust_point.z;
		if (robust_point.z < depth_min) depth_min = robust_point.z;
	}
	// 保证鼻尖区域在人脸的中心，否则判断为人脸偏转角度过大
	if ((face_info.pts[NOSEINDEX].y >= bottom) || (face_info.pts[NOSEINDEX].y <= top) ||
		(face_info.pts[NOSEINDEX].x <= left) || (face_info.pts[NOSEINDEX].x >= right)) {
		face_state = SIDEFACE;
		return 0;
	}
	return 0;
}

int FacePoseEstimate::OLS_Plane(std::vector<cv::Point3f> point, double *param_plane){
	/*输入一组坐标值，根据最小二乘法计算平面方程
	分别返回 a ,b, c 的值
	aX + bY - Z + c = 0 */
	//Ax = 0的形式，将A, b 写成矩阵的形式
	cv::Mat A((int)point.size(), 3, CV_32F);
	cv::Mat b((int)point.size(), 1, CV_32F);

	//  cout <<"原始点为:"<< point << endl;

	//初始化矩阵A
	for (size_t i = 0; i < point.size(); i++) {
		A.at<float>((int)i, 0) = (point[i].x);
		A.at<float>((int)i, 1) = (point[i].y);
		A.at<float>((int)i, 2) = 1;
	}
	
	//初始化矩阵b 
	for (size_t i = 0; i< point.size(); i++)
		b.at<float>((int)i, 0) = -(point[i].z);

	//根据线性代数知识，A'* A * x = A' * b 求得的矩阵 x 即为最优解
	//解 x = (A' * A)^-1 * A' * b
	cv::Mat par = -((A.t()*A).inv()*A.t()*b);

	param_plane[0] = par.at<float>(0, 0);
	param_plane[1] = par.at<float>(0, 1);
	param_plane[3] = par.at<float>(0, 2);
	param_plane[2] = -1;

	return 0;
}

int FacePoseEstimate::panelFitting() {
	double param_plane[4] = { 0.0 };
	if (3 == robust_points.size()) {
		// 三个点估计一个平面方程
		cv::Point3f ptemp1 = robust_points[1] - robust_points[0];
		cv::Point3f ptemp2 = robust_points[2] - robust_points[0];

		param_plane[0] = ptemp1.y*ptemp2.z - ptemp2.y*ptemp1.z;
		param_plane[1] = ptemp1.z*ptemp2.x - ptemp2.z*ptemp1.x;
		param_plane[2] = ptemp1.x*ptemp2.y - ptemp2.x*ptemp1.y;
		param_plane[3] = -(param_plane[0] * robust_points[0].x + param_plane[1] * robust_points[0].y + param_plane[2] * robust_points[0].z);
	};

	if (4 == robust_points.size()) {
		//最小二乘法拟合平面方程 aX + bY - Z + c = 0
		OLS_Plane(robust_points, param_plane);
	}

	// pitch(俯仰角) yaw(偏航角) roll(翻滚角)
	angle[0] = asin(param_plane[1] / sqrt(pow(param_plane[1], 2) + pow(param_plane[2], 2)));
	angle[1] = asin(param_plane[0] / sqrt(pow(param_plane[0], 2) + pow(param_plane[2], 2)));
	angle[0] = -(angle[0] * 180) / CV_PI;
	angle[1] = -(angle[1] * 180) / CV_PI;
	if (isnan(angle[0])) face_state = ERRORDEPTH;
	if (isnan(angle[1])) face_state = ERRORDEPTH;

	//roll(翻滚角)
	cv::Point2f middle_eyes = (face_info.pts[0] + face_info.pts[1]) / 2;
	cv::Point2f middle_mouth = (face_info.pts[3] + face_info.pts[4]) / 2;
	cv::Point2f eyes_to_mouth = middle_eyes - middle_mouth;
	angle[2] = asin(eyes_to_mouth.x / sqrt(pow(eyes_to_mouth.x, 2) + pow(eyes_to_mouth.y, 2)));
	angle[2] = (angle[2] * 180) / CV_PI;

	angle[0] = angle[0] - DELTAANGLE;
	if (abs(angle[0])> PITCHANGLE) {
		face_state = SIDEFACE;
		return 0;
	}
	if (abs(angle[1])> YAWANGLE) {
		face_state = SIDEFACE;
		return 0;
	}
	if (abs(angle[2])> ROLLANGLE) {
		face_state = SIDEFACE;
		return 0;
	}
	return 0;
}

int FacePoseEstimate::panelFittingRGB() {
	//roll(翻滚角)
	cv::Point2f middle_eyes = (face_info.pts[0] + face_info.pts[1]) / 2;
	cv::Point2f middle_mouth = (face_info.pts[3] + face_info.pts[4]) / 2;
	cv::Point2f eyes_to_mouth = middle_eyes - middle_mouth;
	angle[2] = asin(eyes_to_mouth.x / sqrt(pow(eyes_to_mouth.x, 2) + pow(eyes_to_mouth.y, 2)));
	angle[2] = (angle[2] * 180) / CV_PI;
	return 0;
}

int FacePoseEstimate::rotateArbitrarilyAngle(cv::Mat &src)
{
	cv::Mat face_region_temp;

	float radian = (float)(angle[2] * CV_PI / 180.0);

	//填充图像
	int maxBorder = (int)(max(src.cols, src.rows)* 1.414); //即为sqrt(2)*max
	int dx = (maxBorder - src.cols) / 2;
	int dy = (maxBorder - src.rows) / 2;
	cv::copyMakeBorder(src, face_region_temp, dy, dy, dx, dx, cv::BORDER_CONSTANT);

	//旋转
	cv::Point2f center((float)(face_region_temp.cols / 2), (float)(face_region_temp.rows / 2));
	cv::Mat affine_matrix = cv::getRotationMatrix2D(center, angle[2], 1.0);//求得旋转矩阵
	warpAffine(face_region_temp, face_region_temp, affine_matrix, face_region_temp.size());

	//计算图像旋转之后包含图像的最大的矩形
	float sinVal = abs(sin(radian));
	float cosVal = abs(cos(radian));
	
	// 紧致人脸区域
	cv::Size targetSize(src.size());
	//剪掉多余边框
	int x = (face_region_temp.cols - targetSize.width) / 2;
	int y = (face_region_temp.rows - targetSize.height) / 2;
	targetSize.width = min(targetSize.width, face_region_temp.cols - x);
	targetSize.height = min(targetSize.height, face_region_temp.rows - y);
	cv::Rect rect(x, y, targetSize.width, targetSize.height);
	face_region = cv::Mat(face_region_temp, rect);

	// 方形人脸区域
	int maxEdge = max(src.cols, src.rows);
	cv::Size targetSize_square(cv::Size(maxEdge, maxEdge));
	int x_square = (face_region_temp.cols - targetSize_square.width) / 2;
	int y_square = (face_region_temp.rows - targetSize_square.height) / 2;
	targetSize_square.width = min(targetSize_square.width, face_region_temp.cols - x_square);
	targetSize_square.height = min(targetSize_square.height, face_region_temp.rows - y_square);
	cv::Rect rect_square(x_square, y_square, targetSize_square.width, targetSize_square.height);
	face_region_square = cv::Mat(face_region_temp, rect_square);

	return 0;
}

int FacePoseEstimate::pointsRotate(float *points, float edge)
{
	float theta = angle[2] * CV_PI / 180;
	float pointTemp[10];
	float matRotate[2][2]{
		{ std::cos(theta), -std::sin(theta) },
		{ std::sin(theta), std::cos(theta) }
	};
	for (int i = 0; i < 5; ++i) {
		points[i] = points[i] - edge;
		points[i + 5] = points[i + 5] - edge;

		pointTemp[i] = points[i] * matRotate[0][0] + points[i + 5] * matRotate[1][0] + edge;
		pointTemp[i + 5] = points[i] * matRotate[0][1] + points[i + 5] * matRotate[1][1] + edge;
	}
	for (int i = 0; i < 5; ++i) {
		points[i] = pointTemp[i];
		points[i + 5] = pointTemp[i + 5];
	}
	return 0;
}

int FacePoseEstimate::alignFaceFun() {
	double roll = angle[2];
	float pointResult[10];
	float edge_ratio = 1.00;

	cv::Mat image_face = rgb_image(face_info.finalbox);

	rotateArbitrarilyAngle(image_face);

	int box_width = face_info.finalbox.width;
	int box_height = face_info.finalbox.height;

	float maxEdge = float(max(box_width, box_height)) * edge_ratio;
	int edgeHalf = maxEdge / 2;

	for (int i = 0; i < 5; ++i) {
		pointResult[i] = (face_info.pts[i].x - face_info.finalbox.x + (maxEdge - box_width) / 2) * edge_ratio;
		pointResult[i + 5] = (face_info.pts[i].y - face_info.finalbox.y + (maxEdge - box_height) / 2) * edge_ratio;
	}

	pointsRotate(pointResult, edgeHalf);

	float ratio = 112.0 / face_region_square.cols;
	for (int i = 0; i < 5; ++i) {
		face_info.pts[i].x = round(float(pointResult[i]) * float(ratio));
		face_info.pts[i].y = round(float(pointResult[i + 5]) * float(ratio));
	}
	cv::resize(face_region_square, face_region_square, cv::Size(112, 112));

	return 0;
}

int FacePoseEstimate::calFacePose() {
	if (depth_image.data) {
		completeFace();
		if (NORMALFACE != face_state) {
			std::cout << "0.人脸不满足识别条件：人脸不完整！" << std::endl;
			return 1;
		}

		effeciteRange();
		if (NORMALFACE != face_state) {
			std::cout << "1.人脸不满足识别条件：不在有效深度距离内！" << std::endl;
			return 1;
		}

		standardKeypoints();
		if (NORMALFACE != face_state) {
			std::cout << "2.人脸不满足识别条件：关键点不在有效范围！" << std::endl;
			return 1;
		}

		robustKeypoints();
		if (NORMALFACE != face_state) {
			std::cout << "3.人脸不满足识别条件：关键点发生异常！" << std::endl;
			return 1;
		}

		panelFitting();
		if (NORMALFACE != face_state) {
			std::cout << "4.人脸不满足识别条件：人脸角度太大！" << std::endl;
			return 1;
		}

		alignFaceFun();
	}
	else {
		panelFittingRGB();

		alignFaceFun();
	}

	return 0;
}