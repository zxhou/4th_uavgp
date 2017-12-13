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
using namespace std;


visualDetection::visualDetection(int ptt, int pdt, int tcc, int hn, int wn)
{
	pixelTopThreshold = ptt;	// threshold of the black parts' pixel value
	pixelDiffThreshold = pdt;	// pixel value differences betweem two channels	on black parts
	thresholdCC = tcc;			// threshold of connected components
	heightNormal = hn;			// height of the normalized image
	widthNormal = wn;			// width of the normalized image
}

void visualDetection::imPreCut(cv::Mat& imSrc, cv::Mat imDst)
{
	imSrc.copyTo(imDst);

	cv::Mat pRoiLeft = imDst(cv::Rect(0, 0, 100, 480));
	pRoiLeft.setTo(cv::Scalar(255, 255, 255));

	cv::Mat pRoiRight = imDst(cv::Rect(540, 0, 100, 480));
	pRoiRight.setTo(cv::Scalar(255, 255, 255));

	cv::Mat pRoiTop = imDst(cv::Rect(0, 0, 640, 40));
	pRoiTop.setTo(cv::Scalar(255, 255, 255));

	cv::Mat pRoiBottom = imDst(cv::Rect(0, 400, 640, 80));
	pRoiBottom.setTo(cv::Scalar(255, 255, 255));

}

void visualDetection::dataTransInit(Digit_INFO *output)
{
	output->Head = 0x24;
	output->FrameID = 0;
	output->Key = 1;
	output->Spray_Statue = 0;
	output->SprayScale = 0;
	output->ShotPoint_Statue = 0;
	output->ShotPoint_x_mm = 0;
	output->ShotPoint_y_mm = 0;
	output->valid_Digit_N = 0;
	for (int i = 0; i <= 11; i++)
	{
		if (i == 1)
		{
			output->tagDigit_POINT[i].DigitName = 10;
		}
		else
		{
			output->tagDigit_POINT[i].DigitName = 0;
		}
		
		output->tagDigit_POINT[i].ImgStatue = 0;
		output->tagDigit_POINT[i].xPiont_mm = 0;
		output->tagDigit_POINT[i].yPiont_mm = 0;
	}

	output->tmp1 = 0;
	output->tmp2 = 0;
}

bool visualDetection::liSpotLocalization(cv::Mat& imLiSpot, cv::Point& pLiSpotCenter)
{
	unsigned int LiRedThreshold = 200;
	unsigned int LiGreenThreshold = 150;
	unsigned int LiBlueThreshold = 150;

	int height = imLiSpot.rows;
	int width = imLiSpot.cols;
	cv::Mat imLiROI = imLiSpot(cv::Rect(height / 4, width / 4, width / 2, height / 2));

	int heightROI = imLiROI.rows;
	int widthROI = imLiROI.cols;
	cv::Mat bwLiSpot(heightROI, widthROI, CV_8UC1, cv::Scalar(0));
	unsigned int numPointsPerSpot = 0;
	unsigned int pointCoordinateX = 0;
	unsigned int pointCoordinateY = 0;

	for (int i = 0; i < heightROI; i++)
	{
		for (int j = 0; j < widthROI; j++)
		{
			if (imLiROI.at<cv::Vec3b>(i, j)[0]<LiBlueThreshold && imLiROI.at<cv::Vec3b>(i, j)[1] < LiGreenThreshold && imLiROI.at<cv::Vec3b>(i, j)[2] > LiRedThreshold)
			{
				bwLiSpot.at<uchar>(i, j) = 255;
				numPointsPerSpot++;
				pointCoordinateX += j;
				pointCoordinateY += i;
			}
		}
	}
	if (numPointsPerSpot == 0)
	{
		return false;
	}
	else
	{
		pLiSpotCenter.x = pointCoordinateX / numPointsPerSpot + width / 4;
		pLiSpotCenter.y = pointCoordinateY / numPointsPerSpot + height / 4;
		return true;
	}


}

void visualDetection::findTotalCC(cv::Mat& roadMask, cv::Mat& roadMask_, std::vector<cv::Mat>& totalCC, int thresholdCC)
{
	cv::Mat label_mat;
	std::vector<int> numPixelsPerCC;
	connectedComponent(roadMask_, label_mat, numPixelsPerCC);
	if (numPixelsPerCC.size() != 0)
	{
		cv::Mat numPixelsPerCCMat = cv::Mat(numPixelsPerCC.size(), 1, CV_32SC1);
		for (int i = 0; i < numPixelsPerCC.size(); i++)
		{
			numPixelsPerCCMat.at<int>(i) = numPixelsPerCC.at(i);
		}
		cv::Mat numPixelsPerCCMat_index = numPixelsPerCCMat;
		cv::sortIdx(numPixelsPerCCMat, numPixelsPerCCMat_index, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);


		int numCC = 0;
		for (int counter = 0; counter < numPixelsPerCC.size(); counter++)
		{
			int largestCCvalue = numPixelsPerCCMat_index.at<int>(numCC) +1;

			cv::Mat matDummy;
			roadMask_.copyTo(matDummy);
			int tempNumCC = 0;
			for (int i = 0; i < label_mat.rows; i++)
			{
				int *row = (int*)label_mat.ptr(i);
				uchar* ptr_matDummy = (uchar*)matDummy.ptr(i);
				for (int j = 0; j < label_mat.cols; j++)
				{
					if (row[j] != largestCCvalue)
					{
						ptr_matDummy[j] = 0;
					}
					else
						tempNumCC++;
				}
			}
			if ((tempNumCC >= thresholdCC) && (tempNumCC == numPixelsPerCCMat.at<int>(numPixelsPerCCMat_index.at<int>(numCC))))
			{
				totalCC.push_back(matDummy);
				numCC++;
			}
			else
			{
				continue;
			}
		}

	}
}

void visualDetection::connectedComponent(cv::Mat& binary, cv::Mat& label_mat, std::vector<int>& numPixelsPerCC)
{
	int label_count;
	//int label_number = 0;
	cv::Mat int_image;

	cv::Mat binary2 = binary.clone();

	for (int i = 0; i < binary2.rows; i++)
	{
		for (int j = 0; j < binary2.cols; j++)
		{
			if (binary2.at<uchar>(i, j)>0)
				binary2.at<uchar>(i, j) = 1;
		}
	}


	label_mat = cv::Mat::zeros(binary2.rows, binary2.cols, CV_32SC1);
	binary2.convertTo(int_image, CV_32SC1);

	label_count = 2;
	for (int y = 0; y < int_image.rows; y++) {
		int *row = (int*)int_image.ptr(y);
		for (int x = 0; x < int_image.cols; x++) {
			if (row[x] != 1) {
				continue;
			}

			cv::Rect rect;
			cv::floodFill(int_image, cv::Point(x, y), label_count, &rect, 0, 0, 8);

			//std::vector <cv::Point2i> blob;

			long int numPixelCC = 0;
			for (int i = rect.y; i < (rect.y + rect.height); i++) {
				int *row2 = (int*)int_image.ptr(i);
				for (int j = rect.x; j < (rect.x + rect.width); j++)
				{
					if (row2[j] != label_count) {
						continue;
					}
					label_mat.at<int>(i, j) = label_count - 1;
					numPixelCC++;
					//blob.push_back(cv::Point2i(j, i));
				}
			}
			numPixelsPerCC.push_back(numPixelCC);
			label_count++;
		}
	}

}

void visualDetection::normalize(cv::Mat& sourceImg, cv::Mat& digitExtNormal, int normHeight, int normWidth, rectPoints& rectVertex,boundPoints& boundVertex)
{
	// define four vertexs of the bounding box
	//	cv::Point leftTop, leftBottom, rightTop, rightBottom;

//	cv::Point left, right, top, bottom;		// interVariance to calculate the vertexs.
	bool zeroLeft = true, zeroTop = true;
	unsigned int height = sourceImg.rows;
	unsigned int width = sourceImg.cols;

	// calculate the bound of the extracted digit:top,bottom,left&right
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (sourceImg.at<uchar>(i, j) == 255 && zeroTop == true)
			{
				boundVertex.top.y = i;
				boundVertex.top.x = j;
				zeroTop = false;
			}
			if (sourceImg.at<uchar>(i, j) == 255 && zeroTop == false)
			{
				boundVertex.bottom.y = i;
				boundVertex.bottom.x = j;
			}
		}
	}

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			if (sourceImg.at<uchar>(j, i) == 255 && zeroLeft == true)
			{
				boundVertex.left.y = j;
				boundVertex.left.x = i;
				zeroLeft = false;
			}
			if (sourceImg.at<uchar>(j, i) == 255 && zeroLeft == false)
			{
				boundVertex.right.y = j;
				boundVertex.right.x = i;
			}
		}
	}

	// convert bound points to bounding box vertex.
	rectVertex.leftTop.x = boundVertex.left.x;
	rectVertex.leftTop.y = boundVertex.top.y;
	rectVertex.rightTop.x = boundVertex.right.x;
	rectVertex.rightTop.y = boundVertex.top.y;
	rectVertex.leftBottom.x = boundVertex.left.x;
	rectVertex.leftBottom.y = boundVertex.bottom.y;
	rectVertex.rightBottom.x = boundVertex.right.x;
	rectVertex.rightBottom.y = boundVertex.bottom.y;

	unsigned int boxWidth = boundVertex.right.x - boundVertex.left.x + 1;
	unsigned int boxHeight = boundVertex.bottom.y - boundVertex.top.y + 1;

	cv::Mat boundingBox(boxHeight, boxWidth, CV_8UC1, cv::Scalar(0));

	// extract the bounding box from the image digitExtracted.
	unsigned int orderRow = boundVertex.top.y;
	unsigned int orderColume = boundVertex.left.x;
	boundingBox = sourceImg(cv::Rect(orderColume, orderRow, boxWidth, boxHeight));
	// normalize the bounding box to digitExtNormal.
	cv::Size normalSize(normWidth, normHeight);
	resize(boundingBox, digitExtNormal, normalSize);

}

void visualDetection::recognitionKnn(cv::Mat& src, float& p)
{
	// Read stored sample and label for training
	cv::Mat tmp;
	cv::Mat tmp1, tmp2;
//	resize(src, tmp1, cv::Size(10, 10), 0, 0, cv::INTER_LINEAR);
	src.copyTo(tmp1);
	tmp1 = tmp1.reshape(1, 1);
	tmp1.convertTo(tmp2, CV_32FC1);
	p = knn.find_nearest(tmp2, 1);

}

//
void visualDetection::recognitionKnnLED(cv::Mat& src, float& p)
{
	// Read stored sample and label for training
	cv::Mat tmp;
	cv::Mat tmp1, tmp2;
	//	resize(src, tmp1, cv::Size(10, 10), 0, 0, cv::INTER_LINEAR);
	src.copyTo(tmp1);
	tmp1 = tmp1.reshape(1, 1);
	tmp1.convertTo(tmp2, CV_32FC1);
	p = knnLED.find_nearest(tmp2, 1);

}

void visualDetection::dataCreateProcess(cv::Mat& src, cv::Mat& tmp2, cv::Mat& response_array)
{
	cv::Mat tmp1;
	// Create sample and label data	

//	resize(src, tmp1, cv::Size(10, 10), 0, 0, cv::INTER_LINEAR); //resize to 10X10

	src.copyTo(tmp1);
	tmp1.convertTo(tmp2, CV_32FC1); //convert to float
	tmp2 = tmp2.reshape(1, 1);
	imshow("src", src);
	int c = cv::waitKey(0); // Read corresponding label for contour from keyoard
	c -= 0x30;     // Convert ascii to intiger value
	response_array.push_back(c); // Store label to a mat
}

bool visualDetection::fileInit()
{
	cv::Mat sample, response;
	cv::FileStorage Data("TrainingData.yml", cv::FileStorage::READ); // Read traing data to a Mat
	Data["data"] >> sample;
	Data.release();

	cv::FileStorage Label("LabelData.yml", cv::FileStorage::READ); // Read label data to a Mat
	Label["label"] >> response;
	Label.release();

	knn.train(sample, response); // Train with sample and responses

	cv::Mat sampleLED, responseLED;
	cv::FileStorage DataLED("TrainingLED.yml", cv::FileStorage::READ); // Read traing data to a Mat
	DataLED["data"] >> sampleLED;
	DataLED.release();

	cv::FileStorage LabelLED("LabelLED.yml", cv::FileStorage::READ); // Read label data to a Mat
	LabelLED["label"] >> responseLED;
	LabelLED.release();

	knnLED.train(sampleLED, responseLED); // Train with sample and responses
	//cout << "Training compleated.....!!" << endl;
	return true;
}

void visualDetection::screenExtract(cv::Mat& imgLed, vector<cv::Mat>& digitArea)
{
	cv::Mat imgLedBlur;
	GaussianBlur(imgLed, imgLedBlur, cv::Size(15, 15), 0);
	int width = imgLed.cols;
	int height = imgLed.rows;

	cv::Mat imLedHsv;
	cv::Mat digLed(height, width, CV_8UC1, cv::Scalar(0));
	cv::cvtColor(imgLedBlur, imLedHsv, CV_BGR2HSV);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if ((imLedHsv.at<cv::Vec3b>(i, j)[0] < 15 || imLedHsv.at<cv::Vec3b>(i, j)[0] > 150) && imLedHsv.at<cv::Vec3b>(i, j)[1]>130 && imLedHsv.at<cv::Vec3b>(i, j)[2]>130)
			{
				digLed.at<uchar>(i, j) = 255;
			}
		}
	}
	/*
	vector<cv::Mat> channels;
	cv::Mat imLedH;
	cv::split(imLedHsv, channels);
	imLedH = channels.at(0);

	cv::Mat imLedHOpen, imLedHClose;
	cv::Mat eleLedHO = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	morphologyEx(imLedH, imLedHOpen, cv::MORPH_OPEN, eleLedHO);

	cv::Mat eleLedHC = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
	morphologyEx(imLedHOpen, imLedHClose, cv::MORPH_CLOSE, eleLedHC);
	*/


//	cv::threshold(imLedHClose, digLed,50,255,CV_THRESH_BINARY_INV);
	
	cv::Mat digLedOpen, digLedClose;
	/*
	cv::Mat eledigLedO = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
	morphologyEx(digLed, digLedOpen, cv::MORPH_OPEN, eledigLedO);
	*/
	cv::Mat eledigLedC = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
	morphologyEx(digLed, digLedClose, cv::MORPH_CLOSE, eledigLedC);

	vector<cv::Mat> forwardTotalCC;
	findTotalCC(digLedClose, digLedClose, forwardTotalCC, 100);

	vector<cv::Mat> ccNormal;
	vector<rectPoints> totalRectVertex;
	vector<boundPoints> totalBoundVertex;

	unsigned int maxCCnum = 15;
	if (forwardTotalCC.size() < maxCCnum)
	{
		maxCCnum = forwardTotalCC.size();
	}

//	for (int i = 0; i < totalCC.size(); i++)
for (int i = 0; i < maxCCnum; i++)
	{
		rectPoints ScreenRectVertex;
		boundPoints ScreenBoundVertex;
		cv::Mat ccNormalSingle;
		normalize(forwardTotalCC[i], ccNormalSingle, heightNormal, widthNormal, ScreenRectVertex, ScreenBoundVertex);
		ccNormal.push_back(ccNormalSingle);
		totalRectVertex.push_back(ScreenRectVertex);
		totalBoundVertex.push_back(ScreenBoundVertex);
	}

	vector<cv::Point> digitCenter;
	for (int i = 0; i < maxCCnum; i++)
	{
		// the center is sum up of 4 points, so if you want to get coordinate of one point, please divide 4;
		cv::Point centerDigit4times = totalRectVertex[i].leftBottom + totalRectVertex[i].leftTop + totalRectVertex[i].rightBottom + totalRectVertex[i].rightTop;
		unsigned int rectWidthLed = 0, rectHeightLed = 0;
		rectWidthLed = abs(totalRectVertex[i].leftTop.x - totalRectVertex[i].rightTop.x);
		rectHeightLed = abs(totalRectVertex[i].leftBottom.y - totalRectVertex[i].leftTop.y);
		if ((centerDigit4times.x)/4>220&&(centerDigit4times.x)/4<420&&centerDigit4times.y/4>190&&centerDigit4times.y/4<290&&rectHeightLed*rectWidthLed>180*50)
		{
				digitArea.push_back(ccNormal[i]);
				digitCenter.push_back(centerDigit4times);
		}
	}

}



void visualDetection::run(cv::Mat& srcImg, Uav2Img_INFO *input,Digit_INFO *output,cv::Mat& dstImg)
{

	srcImg.copyTo(dstImg);
	cv::Mat imgLED;
	srcImg.copyTo(imgLED);
	// LED display screen process 
	vector<cv::Mat> LEDArea;
	screenExtract(imgLED, LEDArea);
	if (LEDArea.size() == 1)
	{
		// recognize digit and cout it.
		char nameLED[16];
		float pLED = 0.0;
		recognitionKnnLED(LEDArea[0], pLED);
		unsigned int pIntLED = (unsigned int)pLED;

		if (pIntLED >= 0 && pIntLED <= 9)
		{

			cv::Point pDrawLED;
			pDrawLED.x = 0;
			pDrawLED.y = dstImg.rows - 1;
			sprintf(nameLED, "%d", pIntLED);
			string wordsLED = nameLED;
			cv::putText(dstImg, wordsLED, pDrawLED, CV_FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 3);
			output->tagDigit_POINT[0].DigitName = pIntLED;
			output->tagDigit_POINT[0].ImgStatue = 1;
		}
		else if (pIntLED == 71)
		{
			sprintf(nameLED, "%d", pIntLED);
			string words = "w_LED";
			cv::Point pDraw;
			pDraw.x = 0;
			pDraw.y = dstImg.rows - 1;
			cv::putText(dstImg, words, pDraw, CV_FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 3);
		}
		else if (pIntLED == 67)
		{
			sprintf(nameLED, "%d", pIntLED);
			string words = "s_LED";
			cv::Point pDraw;
			pDraw.x = 0;
			pDraw.y = dstImg.rows - 1;
			cv::putText(dstImg, words, pDraw, CV_FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 3);
		}

	}
		
	// spray board process
	imPreCut(srcImg, srcImg);

	dataTransInit(output);

	output->FrameID++;

	cv::Mat response_array, imResize;

	unsigned int reHeight = 0, reWidth = 0;
	unsigned int width = srcImg.cols;
	unsigned int height = srcImg.rows;
	reHeight = height / 1;
	reWidth = width / 1;
	cv::resize(srcImg, imResize, cv::Size(reWidth, reHeight));

	cv::Mat bwDigitImg(reHeight, reWidth, CV_8UC1, cv::Scalar(0));

	for (int i = 0; i < reHeight; i++)
	{
		for (int j = 0; j < reWidth; j++)
		{
			if (imResize.at<cv::Vec3b>(i, j)[0] < pixelTopThreshold && imResize.at<cv::Vec3b>(i, j)[1] < pixelTopThreshold && imResize.at<cv::Vec3b>(i, j)[2] < pixelTopThreshold
				&& abs(imResize.at<cv::Vec3b>(i, j)[0] - imResize.at<cv::Vec3b>(i, j)[1]) < pixelDiffThreshold && abs(imResize.at<cv::Vec3b>(i, j)[1] - imResize.at<cv::Vec3b>(i, j)[2]) < pixelDiffThreshold && abs(imResize.at<cv::Vec3b>(i, j)[0] - imResize.at<cv::Vec3b>(i, j)[2]) < pixelDiffThreshold)
			{
				bwDigitImg.at<uchar>(i, j) = 255;
			}
			else
				bwDigitImg.at<uchar>(i, j) = 0;
		}
	}

	vector<cv::Mat> totalConnectComponent;
	vector<cv::Mat> ccNormal;
	vector<rectPoints> totalRectVertex;  // rectangle box 
	vector<boundPoints> totalBoundVertex;	//  top,bottom,left&right points of cc
	findTotalCC(bwDigitImg, bwDigitImg, totalConnectComponent, thresholdCC);

	unsigned int maxTotalCCnum = 10;

	if (totalConnectComponent.size() < maxTotalCCnum)
	{
		maxTotalCCnum = totalConnectComponent.size();
	}

	for (int i = 0; i < maxTotalCCnum; i++)
	{
		rectPoints rectVertex;
		boundPoints boundVertex;
		cv::Mat ccNormalSingle;
		normalize(totalConnectComponent[i], ccNormalSingle, heightNormal, widthNormal, rectVertex,boundVertex);
		ccNormal.push_back(ccNormalSingle);
		totalRectVertex.push_back(rectVertex);
		totalBoundVertex.push_back(boundVertex);
	}


	//..........................................................................
	//....................extract the lidar spot center.........................
	//...............exploit the lidar spot prior information...................
	//.............the function will be added in the next version...............
	//..............................2016-10-10..................................
	//.................................zxhou....................................
	
	if (input->DisImg_mm <= 1200 && input->DisImg_mm != 0)
	{
		cv::Point pLiSpotCenter;
		output->ShotPoint_Statue = liSpotLocalization(imResize, pLiSpotCenter);
		if (output->ShotPoint_Statue == true)
		{
			output->ShotPoint_x_mm = pLiSpotCenter.x * 1;
			output->ShotPoint_y_mm = pLiSpotCenter.y * 1;
		}
		else
		{
			output->ShotPoint_x_mm = 0;
			output->ShotPoint_y_mm = 0;
		}
		
	}
	


	// to calculate the spray area  (approximate calculation)
	for (int i = 0; i < maxTotalCCnum; i++)
	{
		int situation1_x = abs(abs(totalBoundVertex[i].left.x - totalBoundVertex[i].top.x) - abs(totalBoundVertex[i].bottom.x - totalBoundVertex[i].right.x));
		int situation1_y = abs(abs(totalBoundVertex[i].left.x - totalBoundVertex[i].bottom.x) - abs(totalBoundVertex[i].top.x - totalBoundVertex[i].right.x));
		int situation2_x = abs(abs(totalBoundVertex[i].top.y - totalBoundVertex[i].right.y) - abs(totalBoundVertex[i].left.y - totalBoundVertex[i].bottom.y));
		int situation2_y = abs(abs(totalBoundVertex[i].left.y - totalBoundVertex[i].top.y) - abs(totalBoundVertex[i].bottom.y - totalBoundVertex[i].right.y));
		
		if ((situation1_x<5 && situation1_y < 5 )||( situation2_x<5 && situation2_y < 5))
		{
			unsigned int boxHeight = totalConnectComponent[i].rows;
			unsigned int boxWidth = totalConnectComponent[i].cols;
			unsigned int areaBlankInit = 0;
			unsigned int areaBlankSumUp = 0;
			// sprayScale = (sprayBlank - initBlank)/(numPixelSumUp-initBlank)
			for (int h = 0; h < boxHeight; h++)
			{
				for (int w = 0; w < boxWidth; w++)
				{
					if (totalConnectComponent[i].at<uchar>(h, w) = 255)
					{
						areaBlankSumUp++;
					}
				}
			}
			if (areaBlankSumUp / (boxWidth*boxHeight) - 20>0)
			{
				output->SprayScale = unsigned int(areaBlankSumUp / (boxWidth*boxHeight) - 20);
			}
			else
			{
				output->Spray_Statue = 0;
			}
		}
	}

	///
	vector<cv::Mat> boxArea;
	vector<cv::Mat> digitArea;
	vector<cv::Point> digitCenter;

	for (int i = 0; i < maxTotalCCnum; i++)
	{
		for (int j = i + 1; j < maxTotalCCnum; j++)
		{
			// the center is sum up of 4 points, so if you want to get coordinate of one point, please divide 4;
			cv::Point centerFirst = totalRectVertex[i].leftBottom + totalRectVertex[i].leftTop + totalRectVertex[i].rightBottom + totalRectVertex[i].rightTop;
			cv::Point centerSecond = totalRectVertex[j].leftBottom + totalRectVertex[j].leftTop + totalRectVertex[j].rightBottom + totalRectVertex[j].rightTop;
			double disCenter = cv::norm(centerFirst - centerSecond);
			if (disCenter < 10*4)
			{
				if (totalRectVertex[i].leftTop.y > totalRectVertex[j].leftTop.y)
				{
					digitArea.push_back(ccNormal[i]);
					digitCenter.push_back(centerFirst);
					boxArea.push_back(ccNormal[j]);
				}
				else
				{
					digitArea.push_back(ccNormal[j]);
					digitCenter.push_back(centerSecond);
					boxArea.push_back(ccNormal[i]);
				}
			}
		}
	}

	// part of digit recognition
	unsigned int numValid = 0;
	for (int i = 0; i < digitArea.size(); i++)
	{
		// recognize digit and cout it.
		char name[16];
		float p = 0.0;
		recognitionKnn(digitArea[i], p);
		unsigned int pInt = (unsigned int)p;

		if (pInt >= 0 && pInt <= 9)
		{
			cv::Point pDraw;
			pDraw.x = digitCenter[i].x / 4;
			pDraw.y = digitCenter[i].y / 4;
			sprintf(name, "%d%s%d%d%s", pInt,"(",pDraw.x,pDraw.y,")");
			string words = name;
			cv::putText(dstImg, words, pDraw, CV_FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0),3);
			output->tagDigit_POINT[pInt + 1].DigitName = pInt;
			output->tagDigit_POINT[pInt + 1].ImgStatue = 1;
			// uncomment it when run formally
			output->tagDigit_POINT[pInt + 1].xPiont_mm = pDraw.x;
			output->tagDigit_POINT[pInt + 1].yPiont_mm = pDraw.y;
			numValid++;
		}
		else if (pInt == 71)
		{
			sprintf(name, "%d", pInt);
			string words = "w";
			cv::Point pDraw;
			pDraw.x = digitCenter[i].x / 4;
			pDraw.y = digitCenter[i].y / 4;
			cv::putText(dstImg, words, pDraw, CV_FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255),3);
		}
		else if (pInt == 67)
		{
			sprintf(name, "%d", pInt);
			string words = "s";
			cv::Point pDraw;
			pDraw.x = digitCenter[i].x / 4;
			pDraw.y = digitCenter[i].y / 4;
			cv::putText(dstImg, words, pDraw, CV_FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255),3);
		}
	}
	output->valid_Digit_N = numValid;
}

void visualDetection::dataLEDTrain(char* pathImg, int numStart, int numEnd)
{
	cv::Mat digitImg;
	cv::Mat sample;
	cv::Mat response_array, response;
	unsigned int numImg = numEnd - numStart + 1;
	char nameImg[256];
	for (int numImg = numStart; numImg < numEnd; numImg++)
	{
		sprintf(nameImg, "%s%04d.jpg", pathImg, numImg);
		digitImg = cv::imread(nameImg);
		vector<cv::Mat> digitArea;
		screenExtract(digitImg, digitArea);
		if (digitArea.size() == 1)
		{
			cv::Mat src = digitArea[0], dst;
			dataCreateProcess(src, dst, response_array);
			sample.push_back(dst);
		}
		cout << "frame " << numImg << endl;
	}

	cv::Mat tmp;
	tmp = response_array.reshape(1, 1);
	tmp.convertTo(response, CV_32FC1);  //convert to float

	cv::FileStorage Data("TrainingLED.yml", cv::FileStorage::WRITE); // Store the sample data in a file
	Data << "data" << sample;
	Data.release();

	cv::FileStorage Label("LabelLED.yml", cv::FileStorage::WRITE); // Store the label data in a file
	Label << "label" << response;
	Label.release();
	cout << "Training and Label data created successfully....!! " << endl;
}

void visualDetection::dataTrain(char* pathImg,int numStart,int numEnd)
{
	cv::Mat digitImg;
	cv::Mat sample;
	cv::Mat response_array, response;
	unsigned int numImg = numEnd - numStart + 1;
	char nameImg[256];
	for (int numImg = numStart; numImg <= numEnd; numImg++)
	{
		sprintf(nameImg, "%s%04d.jpg", pathImg, numImg);
		digitImg = cv::imread(nameImg);
	
		imPreCut(digitImg,digitImg);

		unsigned int reHeight = 0, reWidth = 0;
		unsigned int width = digitImg.cols;
		unsigned int height = digitImg.rows;
		reHeight = height / 1;
		reWidth = width / 1;
		cv::resize(digitImg, digitImg, cv::Size(reWidth, reHeight));

		cv::Mat bwDigitImg(reHeight, reWidth, CV_8UC1, cv::Scalar(0));

		for (int i = 0; i < reHeight; i++)
		{
			for (int j = 0; j < reWidth; j++)
			{
				if (digitImg.at<cv::Vec3b>(i, j)[0] < pixelTopThreshold && digitImg.at<cv::Vec3b>(i, j)[1] < pixelTopThreshold && digitImg.at<cv::Vec3b>(i, j)[2] < pixelTopThreshold
					&& abs(digitImg.at<cv::Vec3b>(i, j)[0] - digitImg.at<cv::Vec3b>(i, j)[1]) < pixelDiffThreshold && abs(digitImg.at<cv::Vec3b>(i, j)[1] - digitImg.at<cv::Vec3b>(i, j)[2]) < pixelDiffThreshold && abs(digitImg.at<cv::Vec3b>(i, j)[0] - digitImg.at<cv::Vec3b>(i, j)[2]) < pixelDiffThreshold)
				{
					bwDigitImg.at<uchar>(i, j) = 255;
				}
				else
					bwDigitImg.at<uchar>(i, j) = 0;
			}
		}

		vector<cv::Mat> totalConnectComponent;
		vector<cv::Mat> ccNormal;
		vector<rectPoints> totalRectVertex;
		vector<boundPoints> totalBoundVertex;
		findTotalCC(bwDigitImg, bwDigitImg, totalConnectComponent, thresholdCC);
		for (int i = 0; i < totalConnectComponent.size(); i++)
		{
			rectPoints rectVertex;
			boundPoints boundVertex;
			cv::Mat ccNormalSingle;
			normalize(totalConnectComponent[i], ccNormalSingle, heightNormal, widthNormal, rectVertex, boundVertex);
			ccNormal.push_back(ccNormalSingle);
			totalRectVertex.push_back(rectVertex);
			totalBoundVertex.push_back(boundVertex);
		}
		vector<cv::Mat> digitArea;
		for (int i = 0; i < totalConnectComponent.size(); i++)
		{
			for (int j = i + 1; j < totalConnectComponent.size(); j++)
			{
				cv::Point centerFirst = totalRectVertex[i].leftBottom + totalRectVertex[i].leftTop + totalRectVertex[i].rightBottom + totalRectVertex[i].rightTop;
				cv::Point centerSecond = totalRectVertex[j].leftBottom + totalRectVertex[j].leftTop + totalRectVertex[j].rightBottom + totalRectVertex[j].rightTop;
				double disCenter = cv::norm(centerFirst - centerSecond);
				if (disCenter < 10 * 4)
				{
					if (totalRectVertex[i].leftTop.y > totalRectVertex[j].leftTop.y)
					{
						digitArea.push_back(ccNormal[i]);
						cv::Mat src = ccNormal[i], dst;
						dataCreateProcess(src, dst, response_array);
						sample.push_back(dst);
					}
					else
					{
						digitArea.push_back(ccNormal[j]);
						cv::Mat src = ccNormal[j], dst;
						dataCreateProcess(src, dst, response_array);
						sample.push_back(dst);
					}
				}
			}
		}
	}

	cv::Mat tmp;
	tmp = response_array.reshape(1, 1);
	tmp.convertTo(response, CV_32FC1);  //convert to float

	cv::FileStorage Data("TrainingData.yml", cv::FileStorage::WRITE); // Store the sample data in a file
	Data << "data" << sample;
	Data.release();

	cv::FileStorage Label("LabelData.yml", cv::FileStorage::WRITE); // Store the label data in a file
	Label << "label" << response;
	Label.release();
	cout << "Training and Label data created successfully....!! " << endl;
}