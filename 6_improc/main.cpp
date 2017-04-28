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

// 6.1 & 6.2
namespace filters {

Mat g_src;

// 6.1
namespace linear_filters {

Mat g_box_img, g_mean_img, g_gaussian_img;

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

} // end of namespace linear_filters


// 6.2
namespace non_linear_filters {

Mat g_median_img, g_bilateral_img;

int g_median = 10;
int g_bilateral = 10;

auto win_median = "median blur";
auto win_bilateral = "bilateral filter";

void on_median(int, void*)
{
	medianBlur(g_src, g_median_img, g_median * 2 + 1);
	imshow(win_median, g_median_img);
}

void on_bilateral(int, void*)
{
	bilateralFilter(g_src, g_bilateral_img, g_bilateral, g_bilateral * 2, g_bilateral / 2);
	imshow(win_bilateral, g_bilateral_img);
}

} // end of namespace non_linear_filters 


void test(const char* img_name)
{
	using namespace linear_filters;
	using namespace non_linear_filters;

	g_src = imread(img_name, CV_LOAD_IMAGE_COLOR);

	g_box_img = g_src.clone();
	g_mean_img = g_src.clone();
	g_gaussian_img = g_src.clone();
	g_median_img = g_src.clone();
	g_bilateral_img = g_src.clone();

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

	namedWindow(win_median);
	createTrackbar("param: ", win_median, &g_median, 50, on_median);
	on_median(g_median, nullptr);

	namedWindow(win_bilateral);
	createTrackbar("param: ", win_bilateral, &g_bilateral, 50, on_bilateral);
	on_bilateral(g_bilateral, nullptr);

	while ('q' != waitKey());
}

}


namespace morphology {

// 6.3
namespace eorde_and_dilate {

Mat g_src, g_dst;

// 0 for erode, 1 for dilate
int g_erode_or_dilate = 0;

int g_structuring_element_size = 3;

auto win_dst = "result";

void on_process(int, void*)
{
	Mat element = getStructuringElement(MORPH_RECT,
										Size(2 * g_structuring_element_size + 1, 2 * g_structuring_element_size + 1),
										Point(g_structuring_element_size, g_structuring_element_size));

	if (g_erode_or_dilate == 0) {
		erode(g_src, g_dst, element);
	} else {
		dilate(g_src, g_dst, element);
	}

	imshow(win_dst, g_dst);
}

void test(const char* img_name)
{
	g_src = imread(img_name);
	imshow("origin", g_src);

	Mat element = getStructuringElement(MORPH_RECT,
										Size(2 * g_structuring_element_size + 1, 2 * g_structuring_element_size + 1),
										Point(g_structuring_element_size, g_structuring_element_size));

	erode(g_src, g_dst, element);
	imshow(win_dst, g_dst);

	createTrackbar("erode/dilate", win_dst, &g_erode_or_dilate, 1, on_process);
	createTrackbar("kernel", win_dst, &g_structuring_element_size, 21, on_process);

	while ('q' != waitKey());
}

}


// 6.4 
namespace comprehensive {

Mat g_src, g_dst;

int g_element_shape = MORPH_RECT;

int g_trackbar_pos = 10;

auto win_origin = "origin";
auto win_erode = "erode";
auto win_dilate = "dilate";
auto win_open = "open = erode + dilate";
auto win_close = "close = dilate + erode";
auto win_gradient = "gradient = dilate - erode";
auto win_tophat = "top hat = src - open";
auto win_blackhat = "black hat = close - src";

void on_trackbar(int, void*)
{
	Mat element = getStructuringElement(g_element_shape,
										Size(g_trackbar_pos * 2 + 1, g_trackbar_pos * 2 + 1),
										Point(g_trackbar_pos, g_trackbar_pos));

	morphologyEx(g_src, g_dst, MORPH_ERODE, element);
	imshow(win_erode, g_dst);

	morphologyEx(g_src, g_dst, MORPH_DILATE, element);
	imshow(win_dilate, g_dst);

	morphologyEx(g_src, g_dst, MORPH_OPEN, element);
	imshow(win_open, g_dst);

	morphologyEx(g_src, g_dst, MORPH_CLOSE, element);
	imshow(win_close, g_dst);

	morphologyEx(g_src, g_dst, MORPH_GRADIENT, element);
	imshow(win_gradient, g_dst);

	morphologyEx(g_src, g_dst, MORPH_TOPHAT, element);
	imshow(win_tophat, g_dst);

	morphologyEx(g_src, g_dst, MORPH_BLACKHAT, element);
	imshow(win_blackhat, g_dst);
}

void test(const char* img_name)
{
	namedWindow(win_origin);
	createTrackbar("pos", win_origin,&g_trackbar_pos, 20, on_trackbar);
	g_src = imread(img_name);
	imshow(win_origin, g_src);

	while (true) {
		on_trackbar(g_trackbar_pos, nullptr);

		int c = waitKey();
		if ('q' == c || 27 == c) {
			break;
		} else if ('0' == c) {
			g_element_shape = MORPH_RECT;
		} else if ('1' == c) {
			g_element_shape = MORPH_CROSS;
		} else if ('2' == c) {
			g_element_shape = MORPH_ELLIPSE;
		} else if (' ' == c) {
			g_element_shape = (g_element_shape + 1) % 3;
		}
	}
}

}

}




int main()
{
	//filters::test("filters_2.jpg");

	//morphology::eorde_and_dilate::test("morphology.jpg");

	morphology::comprehensive::test("captain_america.jpg");
}
