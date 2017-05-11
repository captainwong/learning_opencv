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


// 8.2
namespace convex_hull_test {

// 8.2.3
namespace basic_test {

void test()
{
	Mat img(800, 800, CV_8UC3);
	mt19937 rng;
	uniform_int_distribution<int> dist(3, 100);
	uniform_int_distribution<int> dist_x(img.cols / 4, img.cols * 3 / 4);
	uniform_int_distribution<int> dist_y(img.rows / 4, img.rows * 3 / 4);
	uniform_int_distribution<int> dist_color(0, 255);

	while (true) {
		const int pts_count = dist(rng);

		vector<Point> pts;
		for (int i = 0; i < pts_count; i++) {
			pts.push_back(Point(dist_x(rng), dist_y(rng)));
		}

		vector<int> hull;
		convexHull(Mat(pts), hull, true);

		img = Scalar::all(0);
		
		for (int i = 0; i < pts_count; i++) {
			circle(img, pts[i], 3, Scalar(dist_color(rng), dist_color(rng), dist_color(rng)), FILLED, LINE_AA);
		}

		size_t hull_count = hull.size();
		Point pt0 = pts[hull[hull_count - 1]];

		for (size_t i = 0; i < hull_count; i++) {
			Point pt = pts[hull[i]];
			line(img, pt0, pt, Scalar(255, 255, 255), 2, LINE_AA);
			pt0 = pt;
		}

		imshow("result", img);

		if (waitKey() == 27) {
			break;
		}
	}
}

}

// 8.2.4
namespace comprehensive_test {

auto win = "origin";
auto win_result = "result";

Mat src, gray;
int thresh = 50;
const int thresh_max = 255;
mt19937 rng;
uniform_int_distribution<int> dist_color(0, 255);
Mat src_copy = src.clone();
Mat threshold_img_output;
vector<vector<Point>> contours;
vector<Vec4i> hierarchy;

void on_trackbar(int = 0, void* = nullptr)
{
	threshold(gray, threshold_img_output, thresh, thresh_max, THRESH_BINARY);
	findContours(threshold_img_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	vector<vector<Point>> hull(contours.size());
	for (size_t i = 0; i < contours.size(); i++) {
		convexHull(Mat(contours[i]), hull[i], false);
	}

	Mat drawing = Mat::zeros(threshold_img_output.size(), CV_8UC3);
	for (size_t i = 0; i < contours.size(); i++) {
		Scalar color(dist_color(rng), dist_color(rng), dist_color(rng));
		drawContours(drawing, contours, i, color, 1, LINE_8, noArray(), 0);
	}

	imshow(win_result, drawing);
}

void test(const char* img)
{
	src = imread(img);
	cvtColor(src, gray, COLOR_BGR2GRAY);
	blur(gray, gray, Size(3, 3));

	namedWindow(win);
	imshow(win, src);
	createTrackbar("thresh", win, &thresh, thresh_max, on_trackbar);
	on_trackbar();
	waitKey();
}

}

}


// 8.3
namespace convex_bounding {

// 8.3.6
namespace rect_bounding {



}

}

int main()
{
	//contour_test::test("contour.jpg");

	//convex_hull_test::basic_test::test();
	convex_hull_test::comprehensive_test::test("convexhull.jpg");
}