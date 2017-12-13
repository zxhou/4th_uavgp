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



#include "zxVisualDetection.h"
#include "visualDetection.h"

using namespace std;

int widthSrc = 0;
int heightSrc = 0;

// maximal pixel value of digits---80
// maximal pixel differences between channels---30
// threshold of cc---100
// image height normalization---50
// image width normalization---40
static visualDetection uavgp(80, 30, 100, 50, 40);

bool VisualDetection_Initialization(int width, int height)
{
	uavgp.fileInit();
	widthSrc = width;
	heightSrc = height;
	return true;
}

void VisualDetection_DoNext(unsigned char *pOrgImg, Uav2Img_INFO *input, Digit_INFO *output, unsigned char *pResImg)
{
	//.................................................................................
	//...............when training data, please uncomment it...........................
	//..............................zxhou..............................................
	//............................2016-10-15...........................................
	//.................................................................................
//		char pathImgR[256];
	//	char* pathImg = "F:\\dataset\\onSite\\1018\\all_rename\\frame_";
//		sprintf(pathImgR, "%s%d.jpg", pathImg, digitLabel);
		// training dataset path
		// number of the first image
		// number of the last image
	//		uavgp.dataTrain(pathImg, 1000, 1300);
	//		uavgp.dataLEDTrain(pathImg, 0, 2736);
	
	double time0 = static_cast<double>(cv::getTickCount());

	cv::Mat srcImg(heightSrc, widthSrc, CV_8UC3, (void*)pOrgImg);
	cv::Mat dstImg;
	uavgp.run(srcImg, input, output, dstImg);

	time0 = 1 / (((double)cv::getTickCount() - time0) / cv::getTickFrequency());

	char fps[16];
	sprintf(fps, "%s%f", "FPS=", time0);
	string words = fps;
	cv::Point pDraw;
	pDraw.x = 30;
	pDraw.y = dstImg.rows-1;
	cv::putText(dstImg, words, pDraw, CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0));

	memcpy(pResImg, dstImg.data, sizeof(unsigned char)*heightSrc*widthSrc * 3);
}


int main()
{
	Uav2Img_INFO *input = new Uav2Img_INFO;
	Digit_INFO *output = new Digit_INFO;

	//	VisualDetection_Initialization(1920, 1080);

	cv::Mat srcImg;
	for (int i = 555; i < 2086; i++)
	{
		cout << "frame" << i << endl;
//		int digitLabel = 73;
		char pathImgR[256];
		char* pathImg = "F:\\dataset\\onSite\\interrupt\\2585\\frame_";
		sprintf(pathImgR, "%s%04d.jpg", pathImg, i);

		srcImg = cv::imread(pathImgR);
		unsigned char *imgIn = srcImg.data;

		unsigned char *imgOut = new unsigned char[640 * 480 * 3];
		int w = srcImg.cols, h = srcImg.rows;

		VisualDetection_Initialization(w, h);

		VisualDetection_DoNext(imgIn, input, output, imgOut);
		cv::Mat dst(heightSrc, widthSrc, CV_8UC3, (void*)imgOut);
		
		delete imgOut;
	}
	

	
	delete input;
	delete output;

}
