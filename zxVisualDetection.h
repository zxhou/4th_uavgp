
#ifndef ZXVISUALDETECTION_H
#define ZXVISUALDETECTION_H	

#include "visualDetection.h"
#include <opencv2/opencv.hpp>
#include <cstdint>

struct rectPoints
{
	cv::Point leftTop;
	cv::Point leftBottom;
	cv::Point rightTop;
	cv::Point rightBottom;
};

struct boundPoints
{
	cv::Point top;
	cv::Point bottom;
	cv::Point left;
	cv::Point right;
};

class visualDetection
{
private:
	int pixelTopThreshold;
	int pixelDiffThreshold;
	int thresholdCC;
	// normalize size
	int heightNormal;
	int widthNormal;

	cv::KNearest knn;
	cv::KNearest knnLED;
	void dataTransInit(Digit_INFO *output);
	void connectedComponent(cv::Mat& binary, cv::Mat& label_mat, std::vector<int>& numPixelsPerCC);
	void findTotalCC(cv::Mat& roadMask, cv::Mat& roadMask_, std::vector<cv::Mat>& totalCC, int thresholdCC);
	void normalize(cv::Mat& sourceImg, cv::Mat& digitExtNormal, int normHeight, int normWidth, rectPoints& rectVertex, boundPoints& boundVertex);
	void recognitionKnn(cv::Mat& src, float& p);
	void dataCreateProcess(cv::Mat& src, cv::Mat& tmp2, cv::Mat& response_array);
	bool liSpotLocalization(cv::Mat& imLiSpot, cv::Point& pLiSpotCenter);
	void imPreCut(cv::Mat& imSrc, cv::Mat imDst);
	void screenExtract(cv::Mat& imgLed, std::vector<cv::Mat>& digitArea);
	void recognitionKnnLED(cv::Mat& src, float& p);
public:
	bool fileInit();
	void dataTrain(char* pathImg, int numStart, int numEnd);
	void run(cv::Mat& srcImg, Uav2Img_INFO *input, Digit_INFO *output, cv::Mat& dstImg);
	void dataLEDTrain(char* pathImg, int numStart, int numEnd);
	
	visualDetection(int ptt, int pdt, int tcc, int hn, int wn);
};

#endif