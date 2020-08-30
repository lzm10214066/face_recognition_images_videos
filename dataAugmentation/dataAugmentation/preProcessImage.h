#pragma once
#include <opencv2/opencv.hpp>
#include <random>
#include <iostream>
using namespace cv;
using namespace std;

class ImagePreProcess
{
public:
	//ImagePreProcess();
	//~ImagePreProcess();
	void addGaussianNoise(Mat & img, const int& mu, const int & sigma);
	void colorOverlay(const Mat &src, const Mat &reference, Mat &out, double betaValue=1);
	void colorOverlay(const Mat &src, Mat &out, int h_value, int s_value, int v_value=50);
	void resizeBlur(const Mat &src, Mat &out, double s);

	void addShadow(const Mat &src, Mat &out,int direction);
	void grayNormalize(Mat &img, int sdmean, int sdvar, int nonlin);
private:
	

};
