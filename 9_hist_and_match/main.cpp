#define _CRT_SECURE_NO_WARNINGS

#include <iostream>

#pragma warning(push)
#pragma warning(disable:4819)
#include <opencv2/opencv.hpp>
#pragma warning(pop)

#include <functional>
#include <vector>
#include <iomanip>
#include <random>

using namespace std;
using namespace cv;

// 9.2.3
namespace hue_saturation_histogram {

void test(const char* img)
{
	Mat src, hsv;
	src = imread(img);
	cvtColor(src, hsv, COLOR_BGR2HSV);

	int hue_bins = 30;
	int saturation_bins = 32;
	int hist_sz[] = { hue_bins, saturation_bins };
	float hue_ranges[] = { 0,180 };
	float saturation_ranges[] = { 0, 256 };
	const float* ranges[] = { hue_ranges, saturation_ranges };
	
	Mat dst;
	int channels[] = { 0,1 };

	calcHist(&hsv, 1, channels, Mat(), dst, 2, hist_sz, ranges, true, false);

	double max_val = 0;
	minMaxLoc(dst, 0, &max_val);

	int scale = 10;

	Mat hist = Mat::zeros(saturation_bins * scale, hue_bins * 10, CV_8UC3);

	for (int hue = 0; hue < hue_bins; hue++) {
		for (int saturation = 0; saturation < saturation_bins; saturation++) {
			float bin = dst.at<float>(hue, saturation);
			int intensity = cvRound(bin * 255 / max_val);
			rectangle(hist, Point(hue*scale, saturation*scale),
					  Point((hue + 1)*scale - 1, (saturation + 1)*scale - 1),
					  Scalar::all(intensity), FILLED);
		}
	}

	imshow("origin", src);
	imshow("H-S histogram", hist);

	waitKey();
}

}

// 9.2.4
namespace one_dimension_histogram {
void test(const char*img) {
	Mat src = imread(img);
	imshow("origin", src);

	Mat hist;
	int dims = 1;
	float hranges[] = { 0,255 };
	const float* ranges[] = { hranges };
	int size = 256;
	int channels = 0;

	calcHist(&src, 1, &channels, Mat(), hist, dims, &size, ranges);

	int scale = 1;
	Mat dst(size*scale, size, CV_8U, Scalar(0));

	double minval = 0, maxval = 0;
	minMaxLoc(hist, &minval, &maxval);

	int hpt = saturate_cast<int>(0.9*size);
	for (int i = 0; i < size; i++) {
		float bin = hist.at<float>(i);
		int realval = saturate_cast<int>(bin*hpt / maxval);
		rectangle(dst, Point(i*scale, size - 1), Point((i + 1)*scale - 1, size - realval), Scalar(255));
	}

	imshow("one dimension histogram", dst);
	waitKey();
}
}

int main()
{
	//hue_saturation_histogram::test("hs_histogram.jpg");

	one_dimension_histogram::test("one_dimension_histogram.jpg");
}
