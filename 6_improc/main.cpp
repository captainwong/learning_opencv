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

// 6.3 & 6.4
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


// 6.5
namespace flood_fill {

Mat src, dst, gray, mask;
int fill_mode = 1;
int low_diff = 20, up_diff = 20;
int connectivity = 4;
bool is_color = true;
bool use_mask = false;
int new_mask_val = 255;

auto win_name = "result";
auto win_mask = "mask";

void on_mouse(int e, int x, int y, int, void*)
{
	if (e != EVENT_LBUTTONDOWN) {
		return;
	}

	Point seed(x, y);
	int ld = fill_mode == 0 ? 0 : low_diff;
	int ud = fill_mode == 0 ? 0 : up_diff;
	int flags = connectivity | (new_mask_val << 8) | (fill_mode == 1 ? FLOODFILL_FIXED_RANGE : 0);

	mt19937 engine;
	uniform_int_distribution<int> dist(0, 255);
	
	int r = dist(engine);
	int g = dist(engine);
	int b = dist(engine);

	Mat d = is_color ? dst : gray;

	Scalar new_color = is_color ? Scalar(b, g, r) : Scalar(r*0.299 + g*0.587 + b*0.114);

	int area = 0; 

	if (use_mask) {
		threshold(mask, mask, 1, 128, THRESH_BINARY);
		area = floodFill(d, mask, seed, new_color, nullptr,
						 Scalar(ld, ld, ld), Scalar(ud, ud, ud), flags);
		imshow(win_mask, mask);
	} else {
		area = floodFill(d, seed, new_color, nullptr,
						 Scalar(ld, ld, ld), Scalar(ud, ud, ud), flags);
	}

	imshow(win_name, d);
	cout << area << " pixels repainted" << endl;
}

void test(const char* img_name)
{
	src = imread(img_name);
	src.copyTo(dst);
	cvtColor(src, gray, COLOR_BGR2GRAY);
	mask.create(src.rows + 2, src.cols + 2, CV_8UC1);

	namedWindow(win_name);
	createTrackbar("low difference", win_name, &low_diff, 255);
	createTrackbar("up difference", win_name, &up_diff, 255);

	setMouseCallback(win_name, on_mouse);

	while (true) {
		imshow(win_name, is_color ? dst : gray);

		int c = waitKey();
		if ((c & 255) == 27) {
			break;
		}

		switch (c) {
		case '1': // color/gray
			if (is_color) {
				cout << "switched to gray mode" << endl;
				cvtColor(src, gray, COLOR_BGR2GRAY);
			} else {
				cout << "switched to color mode" << endl;
				src.copyTo(dst);
			}
			mask = Scalar::all(0);
			is_color = !is_color;
			break;

		case '2': // diplay/hide mask
			if (use_mask) {
				destroyWindow(win_mask);
			} else {
				namedWindow(win_mask);
				mask = Scalar::all(0);
				imshow(win_mask, mask);
			}
			use_mask = !use_mask;
			break;

		case '3': // restore origin image
			cout << "restore to origin image" << endl;
			src.copyTo(dst);
			cvtColor(dst, gray, COLOR_BGR2GRAY);
			mask = Scalar::all(0);
			break;

		case '4': // flood fill with empty range
			cout << "flood fill with empty range" << endl;
			fill_mode = 0;
			break;

		case '5': // 
			cout << "flood fill with linear/fixed range" << endl;
			fill_mode = 1;
			break;

		case '6': 
			cout << "flood fill with linear/float range" << endl;
			fill_mode = 2;
			break;

		case '7':
			cout << "set connectivity to 4" << endl;
			connectivity = 4;
			break;

		case '8':
			cout << "set connectivity to 8" << endl;
			connectivity = 8;
			break;

		default:
			break;
		}
	}
}

}


// 6.6
namespace image_pyramid {

auto win_name = "result";

Mat src, dst, tmp;

void test(const char* img_name)
{
	src = imread(img_name);
	namedWindow(win_name);
	imshow(win_name, src);

	tmp = src;
	dst = tmp;

	while (true) {
		int key = waitKey(9);

		switch (key) {
		case 27:
		case 'q':
			return;
			break;

		case 'a':
		case '3':
			pyrUp(tmp, dst, Size(tmp.cols * 2, tmp.rows * 2));
			cout << "pyrUp, size*2" << endl;
			break;

		case 'w':
		case '1':
			resize(tmp, dst, Size(tmp.cols * 2, tmp.rows * 2));
			cout << "resize, size*2" << endl;
			break;

		case 'd':
		case '4':
			pyrDown(tmp, dst, Size(tmp.cols / 2, tmp.rows / 2));
			cout << "pyrDown, size/2" << endl;
			break;

		case 's':
		case '2':
			resize(tmp, dst, Size(tmp.cols / 2, tmp.rows / 2));
			cout << "resize, size/2";
			break;

		case 'r': // reset
			dst = src;
			break;

		default:
			break;
		}

		imshow(win_name, dst);
		tmp = dst;
	}
}


}


int main()
{
	//filters::test("filters_2.jpg");

	//morphology::eorde_and_dilate::test("morphology.jpg");

	//morphology::comprehensive::test("captain_america.jpg");

	//flood_fill::test("flood_fill.jpg");

	image_pyramid::test("pyramid.jpg");
}
