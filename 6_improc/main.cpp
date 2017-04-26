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


// 6.3
namespace morphology {

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


int main()
{
	//filters::test("filters_2.jpg");

	morphology::test("morphology.jpg");
}
