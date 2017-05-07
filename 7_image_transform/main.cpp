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


// 7.1
namespace edge_detect {

void test_laplacian(const char* img_name)
{
	Mat src, gray, dst, abs_dst;
	src = imread(img_name);
	imshow("laplacian origin", src);

	GaussianBlur(src, src, Size(3, 3), 0);

	cvtColor(src, gray, COLOR_BGR2GRAY);

	Laplacian(gray, dst, CV_16S, 3);

	convertScaleAbs(dst, abs_dst);

	imshow("laplacian result", abs_dst);

	waitKey();
}

void test_scharr(const char* img_name)
{
	Mat grad_x, grad_y, abs_grad_x, abs_grad_y;
	Mat src = imread(img_name);
	imshow("scharr origin", src);
	Mat dst;
	//dst.create(src.size(), src.type());
	
	Scharr(src, grad_x, CV_16S, 1, 0);
	convertScaleAbs(grad_x, abs_grad_x);
	imshow("scharr x direction", abs_grad_x);

	Scharr(src, grad_y, CV_16S, 1, 0);
	convertScaleAbs(grad_y, abs_grad_y);
	imshow("scharr y direction", abs_grad_y);

	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
	imshow("scharr x/y combination", dst);

	waitKey();
}


auto win_canny = "canny";
auto win_sobel = "sobel";

Mat src, gray, dst;

// for canny
Mat canny_edges;
int canny_low_threshhold = 1;

void on_canny(int = 0, void* = nullptr)
{
	blur(gray, canny_edges, Size(3, 3));
	Canny(canny_edges, canny_edges, canny_low_threshhold, canny_low_threshhold * 3, 3);
	dst = Scalar::all(0);
	src.copyTo(dst, canny_edges);
	imshow(win_canny, dst);
}

// for sobel
Mat sobel_gradient_x, sobel_gradient_y, sobel_abs_gradient_x, sobel_abs_gradient_y;
int sobel_kernel_size = 1;

void on_sobel(int = 0, void* = nullptr)
{
	Sobel(src, sobel_gradient_x, CV_16S, 1, 0, 2 * sobel_kernel_size + 1, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(sobel_gradient_x, sobel_abs_gradient_x);

	Sobel(src, sobel_gradient_y, CV_16S, 0, 1, 2 * sobel_kernel_size + 1, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(sobel_gradient_y, sobel_abs_gradient_y);

	addWeighted(sobel_abs_gradient_x, 0.5, sobel_abs_gradient_y, 0.5, 0, dst);

	imshow(win_sobel, dst);
}

void test(const char* img_name)
{
	src = imread(img_name);
	imshow("edge_detect", src);

	dst.create(src.size(), src.type());
	cvtColor(src, gray, COLOR_BGR2GRAY);

	namedWindow(win_canny);
	createTrackbar("param", win_canny, &canny_low_threshhold, 120, on_canny);
	on_canny();

	namedWindow(win_sobel);
	createTrackbar("param", win_sobel, &sobel_kernel_size, 3, on_sobel);
	on_sobel();

	waitKey();
}

}


// 7.2
namespace hough_transform {

Mat src, dst, mid;

int thresh_hold = 100;

auto win_name = "hough lines origin";
auto win_name1 = "HoughLines result";
auto win_name2 = "HoughLinesP result";

void on_hough_lines(int = 0, void* = nullptr)
{
	{
		Mat d = dst.clone();
		vector<Vec2f> lines;
		HoughLines(mid, lines, 1, CV_PI / 180, thresh_hold + 1);

		for (auto _line : lines) {
			float rho = _line[0];
			float theta = _line[1];
			double a = cos(theta);
			double b = sin(theta);
			double x0 = a * rho;
			double y0 = b * rho;
			
			Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * a));
			Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * a));

			line(d, pt1, pt2, Scalar(55, 100, 200), 1, LINE_AA);
		}

		imshow(win_name1, d);
	}

	{
		Mat d = dst.clone();
		vector<Vec4i> lines;
		HoughLinesP(mid, lines, 1, CV_PI / 180, thresh_hold + 1, 50, 10);
		for (auto& _line : lines) {
			line(d, Point(_line[0], _line[1]), Point(_line[2], _line[3]), Scalar(23, 180, 55), 1, LINE_AA);
		}

		imshow(win_name2, d);
	}
}

void test_hough_lines(const char* img)
{
	src = imread(img);
	namedWindow(win_name);
	imshow(win_name, src);
	createTrackbar("threshhold", win_name, &thresh_hold, 200, on_hough_lines);
	
	Canny(src, mid, 50, 200);
	cvtColor(mid, dst, COLOR_GRAY2BGR);

	on_hough_lines();

	waitKey();
}

void test_hough_circles(const char* img)
{
	Mat src = imread(img);
	imshow("hough circle origin", src);

	cvtColor(src, mid, COLOR_BGR2GRAY);
	GaussianBlur(mid, mid, Size(9, 9), 2, 2);

	vector<Vec3f> circles;
	HoughCircles(mid, circles, HOUGH_GRADIENT, 1.5, 10, 200, 100, 0, 0);

	for (auto& cc : circles) {
		Point center(cvRound(cc[0]), cvRound(cc[1]));
		int radius = cvRound(cc[2]);
		circle(src, center, radius, Scalar(155, 50, 255), 3, LINE_AA);

	}

	imshow("hough circles result", src);
	waitKey();
}

}

int main()
{
	//edge_detect::test_laplacian("laplacian.jpg");
	//edge_detect::test_scharr("scharr.jpg");
	//edge_detect::test("canny.jpg");

	hough_transform::test_hough_lines("hough.jpg");
	hough_transform::test_hough_circles("circle.jpg");
}