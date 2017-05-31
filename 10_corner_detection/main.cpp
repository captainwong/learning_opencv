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


// 10.1.5
namespace harris {

auto win1 = "blend";
auto win2 = "scale";

Mat src, src2, gray;
int thresh = 30;
int max_thresh = 175;

void on_trackbar(int = 0, void* = nullptr)
{
	Mat dst, norm, scale;
	dst = Mat::zeros(src.size(), CV_32FC1);
	src2 = src.clone();

	cornerHarris(gray, dst, 2, 3, 0.04, BORDER_DEFAULT);
	imshow("dst", dst);
	normalize(dst, norm, 0, 255, NORM_MINMAX, CV_32FC1, noArray());
	imshow("norm", norm);
	convertScaleAbs(norm, scale);
	imshow("scale before circle", scale);
	for (int row = 0; row < norm.rows; row++) {
		for (int col = 0; col < norm.cols; col++) {
			if (cvRound(norm.at<float>(row, col)) > thresh + 80) {
				circle(src2, Point(col, row), 5, Scalar(10, 10, 255), 2, LINE_8);
				circle(scale, Point(col, row), 5, Scalar(0, 10, 255), 2, LINE_8);
			}
		}
	}

	imshow(win1, src2);
	imshow(win2, scale);
}

void test(const char* img)
{
	src = imread(img);
	src2 = src.clone();
	imshow("origin", src);

	cvtColor(src, gray, COLOR_BGR2GRAY);

	namedWindow(win1);
	createTrackbar("thresh", win1, &thresh, max_thresh, on_trackbar);
	on_trackbar();
	waitKey();
}

}


int main()
{
	harris::test("harris.jpg");
}
