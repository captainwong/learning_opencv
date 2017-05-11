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


// 8.1
namespace contour_test {

auto win = "origin";
auto win_contour = "contour";

Mat src, gray;

int thresh = 80;
int thresh_max = 255;

std::mt19937 rng;
std::uniform_int_distribution<int> dist(0, 255);

Mat canny_mat_output;

vector<vector<Point>> contours;
vector<Vec4i> hierarchy;

void on_trackbar(int = 0, void* = nullptr)
{
	Canny(gray, canny_mat_output, thresh, thresh * 2, 3);

	findContours(canny_mat_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	Mat drawing = Mat::zeros(canny_mat_output.size(), CV_8UC3);
	for (size_t i = 0; i < contours.size(); i++) {
		Scalar color = Scalar(dist(rng), dist(rng), dist(rng));
		drawContours(drawing, contours, i, color, 2, LINE_8, hierarchy, 0);
	}

	imshow(win_contour, drawing);
}

void test(const char* img)
{
	src = imread(img);
	cvtColor(src, gray, COLOR_BGR2GRAY);
	blur(gray, gray, Size(2, 3));

	namedWindow(win);
	imshow(win, src);
	createTrackbar("canny threshhold", win, &thresh, thresh_max, on_trackbar);
	on_trackbar();
	waitKey();

}

}





int main()
{
	contour_test::test("contour.jpg");
}