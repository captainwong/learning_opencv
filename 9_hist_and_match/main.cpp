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


// 9.2.5
namespace rgb_histogram {
void test(const char* img)
{
	Mat src = imread(img);
	imshow("origin", src);

	int bins = 256;
	int hist_size[] = { bins };
	float range[] = { 0,256 };
	const float* ranges[] = { range };
	Mat red, green, blue;
	int channels_r[] = { 0 };

	calcHist(&src, 1, channels_r, Mat(), red, 1, hist_size, ranges, true, false);

	int channels_g[] = { 1 };
	calcHist(&src, 1, channels_g, Mat(), green, 1, hist_size, ranges, true, false);

	int channels_b[] = { 2 };
	calcHist(&src, 1, channels_b, Mat(), blue, 1, hist_size, ranges, true, false);

	double red_max, green_max, blue_max;
	minMaxLoc(red, 0, &red_max);
	minMaxLoc(green, 0, &green_max);
	minMaxLoc(blue, 0, &blue_max);

	int scale = 1;
	int hist_height = 256;
	Mat hist = Mat::zeros(hist_height, bins * 3, CV_8UC3);

	for (int i = 0; i < bins; i++) {
		auto bin_red = red.at<float>(i);
		auto bin_green = green.at<float>(i);
		auto bin_blue = blue.at<float>(i);

		int intensity_red = cvRound(bin_red * hist_height / red_max);
		int intensity_green = cvRound(bin_green * hist_height / green_max);
		int intensity_blue = cvRound(bin_blue * hist_height / blue_max);

		rectangle(hist, Point(i*scale, hist_height - 1), Point((i + 1)*scale - 1, hist_height - intensity_red), Scalar(255, 0, 0));
		rectangle(hist, Point((i + bins)*scale, hist_height - 1), Point((i + bins + 1)*scale - 1, hist_height - intensity_green), Scalar(0, 255, 0));
		rectangle(hist, Point((i + 2 * bins)*scale, hist_height - 1), Point((i + 2 * bins + 1)*scale - 1, hist_height - intensity_blue), Scalar(0, 0, 255));
	}

	imshow("rgb histogram", hist);
	waitKey();
}
}


// 9.3.2
namespace hist_comparision {
void test() 
{
	Mat src_base, src_test1, src_test2;
	Mat hsv_base, hsv_test1, hsv_test2;
	Mat hsv_halfdown;

	src_base = imread("base.jpg");
	src_test1 = imread("test1.jpg");
	src_test2 = imread("test2.jpg");

	imshow("origin", src_base);
	imshow("test1", src_test1);
	imshow("test2", src_test2);

	cvtColor(src_base, hsv_base, COLOR_BGR2HSV);
	cvtColor(src_test1, hsv_test1, COLOR_BGR2HSV);
	cvtColor(src_test2, hsv_test2, COLOR_BGR2HSV);

	hsv_halfdown = hsv_base(Range(hsv_base.rows / 2, hsv_base.rows - 1), Range(0, hsv_base.cols - 1));

	int hbins = 50; int sbins = 60;
	int hist_size[] = { hbins, sbins };
	float hranges[] = { 0,256 };
	float sranges[] = { 0,180 };
	const float* ranges[] = { hranges, sranges };
	int channels[] = { 0,1 };

	Mat base_hist, halfdown_hist, test1_hist, test2_hist;

	calcHist(&hsv_base, 1, channels, Mat(), base_hist, 2, hist_size, ranges, true, false);
	normalize(base_hist, base_hist, 0, 1, NORM_MINMAX, -1, noArray());

	calcHist(&hsv_halfdown, 1, channels, noArray(), halfdown_hist, 2, hist_size, ranges, true, false);
	normalize(halfdown_hist, halfdown_hist, 0, 1, NORM_MINMAX, -1, noArray());

	calcHist(&hsv_test1, 1, channels, noArray(), test1_hist, 2, hist_size, ranges, true, false);
	normalize(test1_hist, test1_hist, 0, 1, NORM_MINMAX, -1, noArray());

	calcHist(&hsv_test2, 1, channels, noArray(), test2_hist, 2, hist_size, ranges, true, false);
	normalize(test2_hist, test2_hist, 0, 1, NORM_MINMAX);

	const std::pair<int, const char*> compare_methods[] = { 
		{CV_COMP_CORREL, "CV_COMP_CORREL"},
		{CV_COMP_CHISQR, "CV_COMP_CHISQR"},
		{CV_COMP_BHATTACHARYYA, "CV_COMP_BHATTACHARYYA"},
		{CV_COMP_CHISQR_ALT, "CV_COMP_CHISQR_ALT "} 
	};

	for (auto compare_method : compare_methods) {
		double base_base = compareHist(base_hist, base_hist, compare_method.first);
		double base_half = compareHist(base_hist, halfdown_hist, compare_method.first);
		double base_test1 = compareHist(base_hist, test1_hist, compare_method.first);
		double base_test2 = compareHist(base_hist, test2_hist, compare_method.first);

		cout << "method " << compare_method.second << ":" << endl
			<< "\tbase-base: " << base_base << endl
			<< "\tbase-half: " << base_half << endl
			<< "\tbase-test1: " << base_test1 << endl
			<< "\tbase-test2: " << base_test2 << endl;
	}

	waitKey();
}
}


int main()
{
	//hue_saturation_histogram::test("hs_histogram.jpg");

	//one_dimension_histogram::test("one_dimension_histogram.jpg");

	//rgb_histogram::test("rgb_histogram.jpg");

	hist_comparision::test();
}
