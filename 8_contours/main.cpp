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

void basic_test()
{
	Mat img(800, 800, CV_8UC3);
	mt19937 rng;
	uniform_int_distribution<int> dist_cnt(3, 100);
	uniform_int_distribution<int> dist_clr(0, 255);
	uniform_int_distribution<int> dist_x(img.cols / 4, img.cols * 3 / 4);
	uniform_int_distribution<int> dist_y(img.rows / 4, img.rows * 3 / 4);	

	while (true) {
		int count = dist_cnt(rng);
		vector<Point> pts;

		for (int i = 0; i < count; i++) {
			pts.push_back(Point(dist_x(rng), dist_y(rng)));
		}

		// rect bounding
		{
			RotatedRect box = minAreaRect(pts);
			Point2f vertex[4] = {};
			box.points(vertex);

			img = Scalar::all(0);

			for (int i = 0; i < count; i++) {
				circle(img, pts[i], 3, Scalar(dist_clr(rng), dist_clr(rng), dist_clr(rng)), FILLED, LINE_AA);
			}

			for (int i = 0; i < 4; i++) {
				line(img, vertex[i], vertex[(i + 1) % 4], Scalar(128, 128, 128), 1, LINE_AA);
			}

			imshow("rect bounding", img);
		}

		// circlr bounding
		{
			Point2f center;
			float radius = 0.0f;
			minEnclosingCircle(pts, center, radius);

			img = Scalar::all(0);
			for (int i = 0; i < count; i++) {
				circle(img, pts[i], 3, Scalar(dist_clr(rng), dist_clr(rng), dist_clr(rng)), FILLED, LINE_AA);
			}

			circle(img, center, cvRound(radius), Scalar(128, 128, 128), 1, LINE_AA);

			imshow("circle bounding", img);
		}

		if (27 == waitKey()) {
			break;
		}
	}
}


// 8.3.8
namespace comprehensive_test {

auto win = "origin";
auto win_intermedia = "intermedia";
auto win_result = "comprehensive bounding test";

Mat src, gray;
int thresh = 50;
const int max_thresh = 255;
mt19937 rng;
uniform_int_distribution<int> dist_clr(0, 255);

void on_trackbar(int = 0, void* = nullptr)
{
	Mat out;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	threshold(gray, out, thresh, max_thresh, THRESH_BINARY);
	imshow(win_intermedia, out);

	findContours(out, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	vector<vector<Point>> poly(contours.size());
	vector<Rect> bound(contours.size());
	vector<Point2f> center(contours.size());
	vector<float> radius(contours.size());

	for (size_t i = 0; i < contours.size(); i++) {
		approxPolyDP(contours[i], poly[i], 3, true);
		bound[i] = boundingRect(poly[i]);

		minEnclosingCircle(poly[i], center[i], radius[i]);
	}

	Mat result = Mat::zeros(out.size(), CV_8UC3);
	for (size_t i = 0; i < contours.size(); i++) {
		auto color = Scalar(dist_clr(rng), dist_clr(rng), dist_clr(rng));
		drawContours(result, poly, i, color, 1, LINE_AA, noArray(), 0);
		rectangle(result, bound[i], color, 1, LINE_AA);
		circle(result, center[i], static_cast<int>(radius[i]), color, 1, LINE_AA);
	}

	imshow(win_result, result);
}

void test(const char* img)
{
	src = imread(img);
	cvtColor(src, gray, COLOR_BGR2GRAY);
	blur(gray, gray, Size(3, 3));

	namedWindow(win);
	imshow(win, src);
	createTrackbar("thresh", win, &thresh, max_thresh, on_trackbar);

	on_trackbar();

	waitKey();
}

}

}


// 8.4 
namespace image_moments {

Mat src, gray;
int thresh = 100;
const int max_thresh = 255;

mt19937 rng;
uniform_int_distribution<int> dist(0, 255);

void on_trackbar(int = 0, void* = nullptr)
{
	Mat out;
	vector<vector<Point>> contours;
	vector<Vec4i> hierachy;

	// detect edges using canny
	Canny(gray, out, thresh, thresh * 2);
	// find contours
	findContours(out, contours, hierachy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	// get the moments
	vector<Moments> mu(contours.size());
	for (size_t i = 0; i < contours.size(); i++) {
		mu[i] = moments(contours[i], false);
	}

	// get the mass center
	vector<Point2f> mc(contours.size());
	for (size_t i = 0; i < contours.size(); i++) {
		mc[i] = Point2f(static_cast<float>(mu[i].m10 / mu[i].m00), 
						static_cast<float>(mu[i].m01 / mu[i].m00));
	}

	// draw contours
	Mat drawing = Mat::zeros(out.size(), out.type());
	for (size_t i = 0; i < contours.size(); i++) {
		Scalar color(dist(rng), dist(rng), dist(rng));
		drawContours(drawing, contours, i, color, 2, LINE_8, hierachy, 0);
		circle(drawing, mc[i], 4, color, FILLED, LINE_8);
	}

	auto win = "Contours";
	namedWindow(win);
	imshow(win, drawing);

	// calculate the area with the moments 00 and compare with the result of the OpenCV function
	cout << "\t Info: Area and Contour Length" << endl;
	for (size_t i = 0; i < contours.size(); i++) {
		cout << " * Contour[" << i <<
			"] - Area (M_00): " << mu[i].m00 <<
			" - Area OpenCV: " << contourArea(contours[i]) <<
			" - Length: " << arcLength(contours[i], true) << endl;
		Scalar color(dist(rng), dist(rng), dist(rng));
		drawContours(drawing, contours, i, color, 2, LINE_8, hierachy, 0);
		circle(drawing, mc[i], 4, color, FILLED, LINE_8);
	}
}

void test(const char* img)
{
	src = imread(img);
	cvtColor(src, gray, COLOR_BGR2GRAY);
	blur(gray, gray, Size(3, 3));

	auto win = "origin";
	namedWindow(win);
	imshow(win, src);

	createTrackbar("Canny Thresh:", win, &thresh, max_thresh, on_trackbar);
	on_trackbar();
	waitKey();
}

}


// 8.5
namespace watershed_test {

auto win = "Watershed";

Mat src, mask;
Point prevpt(-1, -1);
mt19937 rng;
uniform_int_distribution<int> dist(0, 255);

void on_mouse(int e, int x, int y, int flags = 0, void* = nullptr)
{
	if (x < 0 || x >= src.cols || y < 0 || y >= src.rows)return;

	if (e == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON)) {
		prevpt = Point(-1, -1);
	} else if (e == EVENT_LBUTTONDOWN) {
		prevpt = Point(x, y);
	} else if (e == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON)) {
		Point pt(x, y);
		if (prevpt.x < 0) {
			prevpt = pt;
		}

		line(mask, prevpt, pt, Scalar::all(255), 5, LINE_8);
		line(src, prevpt, pt, Scalar::all(255), 5, LINE_8);
		prevpt = pt;
		imshow(win, src);
	}


}

void test(const char* img)
{
	src = imread(img);
	imshow(win, src);
	Mat s, gray;
	src.copyTo(s);
	cvtColor(src, mask, COLOR_BGR2GRAY);
	cvtColor(mask, gray, COLOR_GRAY2BGR);
	mask = Scalar::all(0);

	setMouseCallback(win, on_mouse);

	while (true) {
		int c = waitKey();
		if (c == 27)break;

		if (c == '2') {
			mask = Scalar::all(0);
			s.copyTo(src);
			imshow(win, src);
		} else if (c == '1' || c == ' ') {
			vector<vector<Point>> contours;
			vector<Vec4i> hierachy;

			findContours(mask, contours, hierachy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
			if (contours.empty())continue;

			Mat m(mask.size(), CV_32S);
			m = Scalar::all(0);

			int comp_cnt = 0;
			for (int index = 0; index >= 0; index = hierachy[index][0], comp_cnt++) {
				drawContours(m, contours, index, Scalar::all(comp_cnt + 1), FILLED, LINE_8, hierachy);
			}

			if (comp_cnt == 0)continue;

			vector<Vec3b> color_tab;
			for (int i = 0; i < comp_cnt; i++) {
				color_tab.push_back(Vec3b(dist(rng), dist(rng), dist(rng)));
			}

			auto tc = getTickCount();

			watershed(s, m);
			tc = getTickCount() - tc;
			cout << "\tprocess time=" << tc*1000.0 / getTickFrequency() << "ms" << endl;

			Mat w(m.size(), CV_8UC3);
			for (int i = 0; i < m.rows; i++) {
				for (int j = 0; j < m.cols; j++) {
					int index = m.at<int>(i, j);
					if (index == -1) {
						w.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
					} else if (index <= 0 || index > comp_cnt) {
						w.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
					} else {
						w.at<Vec3b>(i, j) = color_tab[index - 1];
					}

				}
			}

			w = w*0.5 + gray*0.5;
			imshow("watershed transform", w);
		}
	}
}

}


int main()
{
	//contour_test::test("contour.jpg");

	//convex_hull_test::basic_test::test();
	//convex_hull_test::comprehensive_test::test("convexhull.jpg");

	//convex_bounding::basic_test();
	//convex_bounding::comprehensive_test::test("bounding.jpg");

	//image_moments::test("moments.jpg");

	watershed_test::test("watershed.jpg");
}