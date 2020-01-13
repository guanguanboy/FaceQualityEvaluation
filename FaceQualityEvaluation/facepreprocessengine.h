#ifndef _FACE_PRE_PROCESS_ENGINE
#define _FACE_PRE_PROCESS_ENGINE

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

typedef struct ConnectedComponentDescription ConnectedComponentDes;

class FacePreProcessEngine
{
public:
	FacePreProcessEngine();
	FacePreProcessEngine(unsigned int bgThreshold, double depthSimThreshold = 0.3, double ratio = 0.8);
	~FacePreProcessEngine();

	void init(unsigned int bgThreshold = 3000, double depthSimThreshold = 0.3, double ratio = 0.8);

	void proceed(Mat& originDepthImg, vector<Rect>& faceBoxes, Mat& outputDepthImg);

	void findBiggiestFaceBox(vector<Rect>& faceBoxes, Rect& biggestFaceBox);

	unsigned int getBgRemovalThreshold();
	double       getDepthSimilarityThreshold();
	double       getValidFaceAreaRatio();
	void         setBgRemovalThreshold(unsigned int threshold);
	void         setDepthSimilarityThreshold(double threshold);
	void         setValidFaceAreaRatio(double ratio);

	unsigned int getBiggestCCPixelCountFromdepthFace(Mat& depthface);
	unsigned int getBiggestCCPixelCountFromCC(vector<ConnectedComponentDes>& ccList, vector<vector<Point> >& allCcPointsLists);

private:
	
	void setRemovedPixelValueToZero(Mat& img);
	void findMaxAndMinPixelValue(Mat& img, int *p_maxPixelValue, int *p_minPixelValue);
	void normalizeFaceImg(Mat& img, int maxPixelValue, int minPixelValue, int valSetToPixelValEquZero);
	void normalizeCroppedFace(Mat& croppedFace);
	bool isTwoPointsConnected(double checkedPonitPixelValue, double currentCcTypePixelMeanValue);
	double updateCurrentPixelMeanValue(Mat& inputNormFace, vector<Point> & curentCcPointList);
	void updateInputNormFace(vector<Point> &curentCc, double currentCcTypePixelMeanValue, Mat& inputNormFace);
	void findAllConnectedComponment(Mat& inputNormFace, vector<ConnectedComponentDes>& ccList, vector<vector<Point> >& allCcPointsLists);
	static bool comparePixelValue(const ConnectedComponentDes& first, const ConnectedComponentDes& second);
	void extractFaceArea(vector<ConnectedComponentDes>& ccList, vector<vector<Point> >& allCcPointsLists, Mat& OriginDepthFace, Mat& outputDepthFace);
	void renormalizeDepthFace(Mat& depthFace);
private:
	unsigned int backgroundRemovalThreshold;
	double       depthSimilarityThreshold;
	double       validFaceAreaRatio;
};

#endif // !_FACE_PRE_PROCESS_ENGINE


