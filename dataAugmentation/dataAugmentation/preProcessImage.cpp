#include "preProcessImage.h"

void ImagePreProcess::addGaussianNoise(Mat & img, const int& mu, const int & sigma)
{
	auto cols = img.cols*img.channels();
	auto rows = img.rows;
	// 产生Gauss分布随机数发生器
	std::random_device rd;
	std::mt19937 gen(rd());

	std::normal_distribution<> gaussR(mu, sigma);
	for (int i = 0; i < rows; i++)
	{
		auto p = img.ptr<uchar>(i);
		for (int j = 0; j < cols; j++)
		{
			auto tmp = p[j] + gaussR(gen);
			tmp = tmp > 255 ? 255 : tmp;
			tmp = tmp < 0 ? 0 : tmp;
			p[j] = tmp;
		}
	}
}

void ImagePreProcess::colorOverlay(const Mat &src, const Mat &reference, Mat &out,double betaValue)
{
	CV_Assert(reference.rows >= src.rows && reference.cols >= src.cols);
	Mat mask(reference, Rect(0, 0, src.cols, src.rows));

	Mat hsv_src;
	cvtColor(src, hsv_src, CV_BGR2HSV);
	vector<Mat> channels_hsv_src;
	split(hsv_src, channels_hsv_src);

	Mat h_src = channels_hsv_src[0];
	Mat s_src = channels_hsv_src[1];
	Mat v_src = channels_hsv_src[2];


	Mat hsv_mask;
	cvtColor(mask, hsv_mask, CV_BGR2HSV);
	vector<Mat> channels_hsv_mask;
	split(hsv_mask, channels_hsv_mask);

	Mat h_mask = channels_hsv_mask[0];
	Mat s_mask = channels_hsv_mask[1];
	Mat v_mask = channels_hsv_mask[2];

	addWeighted(h_src, 0, h_mask, betaValue, 0.0, h_src);
	addWeighted(s_src, 0, s_mask , betaValue, 0.0, s_src);
	merge(channels_hsv_src, out);
	cvtColor(out, out, CV_HSV2BGR);
}

void ImagePreProcess::colorOverlay(const Mat &src, Mat &out, int h_value, int s_value, int v_value)
{
	Mat hsv_src;
	cvtColor(src, hsv_src, CV_BGR2HSV);
	vector<Mat> channels_hsv_src;
	split(hsv_src, channels_hsv_src);

	Mat h_src = channels_hsv_src[0];
	Mat s_src = channels_hsv_src[1];
	Mat v_src = channels_hsv_src[2];

	h_src = h_value;
	s_src = s_value;
	v_src += v_value;

	merge(channels_hsv_src, out);
	cvtColor(out, out, CV_HSV2BGR);
}

void ImagePreProcess::resizeBlur(const Mat &src, Mat &out, double s)
{
	Mat t;
	resize(src, t, Size(), 1/s, 1/s, INTER_LINEAR);
	resize(t, out, Size(96,112), 0, 0, INTER_LINEAR);
}

void ImagePreProcess::addShadow(const Mat &src, Mat &out, int direction)
{
	Mat black_l(112, 96, CV_8UC1, Scalar::all(0));
	int min = 0, max = 128;
	Mat hsv_src;
	cvtColor(src, hsv_src, CV_BGR2HSV);
	vector<Mat> channels_hsv_src;
	split(hsv_src, channels_hsv_src);

	switch (direction)
	{
	case 0:
		for (int y = 0; y < 112; ++y)
		{
			//int r = rand()%(24) +48;
			//int r = -(i - 56)*(i - 56) / 80. + 60;
			double r = 72;
			for (int x = 0; x < r; ++x)
			{
				black_l.at<uchar>(y, x) = round((max - min) / r * (r -x));
			}
		}; break;
	case 1:
		for (int x = 0; x < 96; ++x)
		{
			//int r = rand()%(24) +48;
			//int r = -(i - 56)*(i - 56) / 80. + 60;
			double r = 84;
			for (int y = 0; y < r; ++y)
			{
				black_l.at<uchar>(y, x) = round((max - min) / r * (r - y));
			}
		}; break;
	case 2:
		for (int y = 0; y < 112; ++y)
		{
			//int r = rand()%(24) +48;
			//int r = -(i - 56)*(i - 56) / 80. + 60;
			double r = 72;
			for (int x = 96-r; x < 96; ++x)
			{
				black_l.at<uchar>(y, x) = round((max - min) / r * (r-96+x));
			}
		}; break;
	case 3:
		for (int x = 0; x < 96; ++x)
		{
			//int r = rand()%(24) +48;
			//int r = -(i - 56)*(i - 56) / 80. + 60;
			double r = 84;
			for (int y = 112-r; y < 112; ++y)
			{
				black_l.at<uchar>(y, x) = round((max - min) / r * (r -112+ y));
			}
		}; break;
	default:
		break;
	}
	
	channels_hsv_src[2] -= black_l;
	merge(channels_hsv_src, out);
	cvtColor(out, out, CV_HSV2BGR);
}

void ImagePreProcess::grayNormalize(Mat &img, int sdmean, int sdvar, int nonlin)
{
	float imgMean = 0, imgVar = 0;

	int i, j;
	int pnum = 0;
	for (i = 0; i < img.rows; ++i)
	{
		uchar *p = img.ptr<uchar>(i);
		for (j = 0; j < img.cols; ++j)
		{
			if (p[j])
			{
				++pnum;
				imgMean += p[j];
				imgVar += float(p[j])*float(p[j]);
			}
		}
	}
	imgMean /= pnum;
	imgVar /= pnum;
	imgVar -= imgMean*imgMean;
	if (imgVar < 1)
		imgVar = 1;
	imgVar = (float)sqrt(imgVar);

	float pa;
	float b1, b2;
	if (nonlin)
	{
		float a1, a2;
		b1 = imgMean + 2 * imgVar;
		b2 = (float)(sdmean + 2 * sdvar);
		a1 = imgMean / b1;
		a2 = (float)sdmean / b2;
		pa = log(a2) / log(a1);
	}
	else
		pa = sdvar / imgVar;

	float tg;
	for (i = 0; i < img.rows; ++i)
	{
		uchar *p = img.ptr<uchar>(i);
		for (j = 0; j < img.cols; ++j)
		{
			if (nonlin)
				tg = b2*pow(img.at<uchar>(i, j) / b1, pa);
			else
				tg = pa*(img.at<uchar>(i, j) - imgMean) + sdmean;

			if (tg < 0)
			{
				p[j] = 0;
			}
			else if (tg > 255)
			{
				p[j] = 255;
			}
			else
				p[j] = uchar(tg);
		}
	}

}