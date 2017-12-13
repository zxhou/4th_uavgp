/************************************************************************
  	This file is part of 4th_uavgp(visual detection part) 
    Copyright (C) 2016  Zhixing Hou <zxhou2016@gmail.com>
    (Intelligent Mobile Robotics Lab (iMRL),
    Nanjing University of Science and Technology, China)

    4th_uavgp is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    4th_uavgp is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*******************************************************************************/
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