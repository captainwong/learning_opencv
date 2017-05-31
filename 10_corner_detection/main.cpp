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


// 10.2.3
namespace shi_tomasi {
auto win = "Shi-Tomasi corner detection";

Mat src, gray;
int corner = 33;
int max_corner = 500;
mt19937 rng;
uniform_int_distribution<int> dist(0, 255);

void on_trackbar(int = 0, void* = nullptr)
{
	corner = max(corner, 1);

	vector<Point2f> corners;
	double qualiti_level = 0.01;
	double min_distance = 10;
	int block_size = 3;
	double k = 0.04;
	Mat copy = src.clone();

	goodFeaturesToTrack(gray, corners, corner, qualiti_level, min_distance, noArray(), block_size, false, k);

	cout << "detected corners: " << corners.size() << endl;

	int r = 4;
	for (size_t i = 0; i < corners.size(); i++) {
		circle(copy, corners[i], r, Scalar(dist(rng), dist(rng), dist(rng)), FILLED, LINE_8);
	}

	imshow(win, copy);
}

void test(const char* img)
{
	src = imread(img);
	cvtColor(src, gray, COLOR_BGR2GRAY);

	namedWindow(win);
	createTrackbar("max corner", win, &corner, max_corner, on_trackbar);
	on_trackbar();
	waitKey();
}

}


// 10.3.3
namespace subpix {
auto win = "sub-pix corner detection";

Mat src, gray;
int corner = 33;
int max_corner = 500;
mt19937 rng;
uniform_int_distribution<int> dist(0, 255);

void on_trackbar(int = 0, void* = nullptr)
{
	corner = max(corner, 1);

	vector<Point2f> corners;
	double qualiti_level = 0.01;
	double min_distance = 10;
	int block_size = 3;
	double k = 0.04;
	Mat copy = src.clone();

	goodFeaturesToTrack(gray, corners, corner, qualiti_level, min_distance, noArray(), block_size, false, k);

	cout << "detected corners: " << corners.size() << endl;

	int r = 4;
	for (size_t i = 0; i < corners.size(); i++) {
		circle(copy, corners[i], r, Scalar(dist(rng), dist(rng), dist(rng)), FILLED, LINE_8);
	}

	imshow(win, copy);

	Size winsize = Size(5, 5);
	Size zerozone = Size(-1, -1);
	TermCriteria criteria(TermCriteria::EPS + CV_TERMCRIT_ITER, 40, 0.001);

	cornerSubPix(gray, corners, winsize, zerozone, criteria);

	for (size_t i = 0; i < corners.size(); i++) {
		cout << "\tprecision coordinate[" << i << "](" << corners[i].x << ", " << corners[i].y << ")" << endl;
	}
}

void test(const char* img)
{
	src = imread(img);
	cvtColor(src, gray, COLOR_BGR2GRAY);

	namedWindow(win);
	createTrackbar("max corner", win, &corner, max_corner, on_trackbar);
	on_trackbar();
	waitKey();
}
}


int main()
{
	//harris::test("harris.jpg");

	//shi_tomasi::test("shi-tomasi.jpg");

	subpix::test("subpix.jpg");
}
