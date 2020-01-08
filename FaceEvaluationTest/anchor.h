#pragma once

#include <iostream>

#include "opencv2/opencv.hpp"

class Anchor {
public:
	Anchor() {
	}

	~Anchor() {
	}

	bool operator<(const Anchor &t) const {
		return score < t.score;
	}

	bool operator>(const Anchor &t) const {
		return score > t.score;
	}

	float& operator[](int i) {
		assert(0 <= i && i <= 4);

		if (i == 0)
			return finalbox.x;
		if (i == 1)
			return finalbox.y;
		if (i == 2)
			return finalbox.width;
		if (i == 3)
			return finalbox.height;
	}

	float operator[](int i) const {
		assert(0 <= i && i <= 4);

		if (i == 0)
			return finalbox.x;
		if (i == 1)
			return finalbox.y;
		if (i == 2)
			return finalbox.width;
		if (i == 3)
			return finalbox.height;
	}

	cv::Rect_< float > anchor; // x1,y1,x2,y2
	float reg[4]; // offset reg
	cv::Point center; // anchor feat center
	float score; // cls score
	std::vector<cv::Point2f> pts; // pred pts

	cv::Rect_< float > finalbox; // final box res

	void print() {
		printf("finalbox %f %f %f %f, score %f\n", finalbox.x, finalbox.y, finalbox.width, finalbox.height, score);
		printf("landmarks ");
		for (int i = 0; i < pts.size(); ++i) {
			printf("%f %f, ", pts[i].x, pts[i].y);
		}
		printf("\n");
	}
};