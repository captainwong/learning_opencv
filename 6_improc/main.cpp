#define _CRT_SECURE_NO_WARNINGS

#include <iostream>

#pragma warning(push)
#pragma warning(disable:4819)
#include <opencv2/opencv.hpp>
#pragma warning(pop)

#include <functional>
#include <vector>
#include <iomanip>

using namespace std;
using namespace cv;


// 6.1
namespace linear_filters {

Mat g_src, g_box_img, g_mean_img, g_gaussian_img;

int g_box = 3;
int g_mean = 3;
int g_gaussian = 3;

auto win_box = "box filter";
auto win_mean = "mean blur";
auto win_gaussian = "gaussian blur";

void on_box(int, void*)
{
	boxFilter(g_src, g_box_img, -1, Size(g_box + 1, g_box + 1));
	imshow(win_box, g_box_img);
}

void on_mean(int, void*)
{
	blur(g_src, g_mean_img, Size(g_mean + 1, g_mean + 1));
	imshow(win_mean, g_mean_img);
}

void on_gaussian(int, void*)
{
	GaussianBlur(g_src, g_gaussian_img, Size(g_gaussian * 2 + 1, g_gaussian * 2 + 1), 0);
	imshow(win_gaussian, g_gaussian_img);
}

void test()
{
	g_src = imread("filters.jpg", CV_LOAD_IMAGE_COLOR);

	g_box_img = g_src.clone();
	g_mean_img = g_src.clone();
	g_gaussian_img = g_src.clone();

	imshow("origin", g_src);

	namedWindow(win_box);
	createTrackbar("kernel: ", win_box, &g_box, 40, on_box);
	on_box(g_box, nullptr);

	namedWindow(win_mean);
	createTrackbar("kernel: ", win_mean, &g_mean, 40, on_mean);
	on_mean(g_mean, nullptr);

	namedWindow(win_gaussian);
	createTrackbar("kernel: ", win_gaussian, &g_gaussian, 40, on_gaussian);
	on_gaussian(g_gaussian, nullptr);

	while ('q' != waitKey());
}

}


int main()
{
	linear_filters::test();

}
