#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


namespace chapter3 {

// 3.1.8
void test_imwrite()
{
	auto create_alpha_mat = [](Mat& mat) {
		for (int i = 0; i < mat.rows; i++) {
			for (int j = 0; j < mat.cols; j++) {
				Vec4b& rgba = mat.at<Vec4b>(i, j);
				rgba[0] = UCHAR_MAX;
				rgba[1] = saturate_cast<uchar>(static_cast<float>(mat.cols - j) / static_cast<float>(mat.cols) * UCHAR_MAX);
				rgba[2] = saturate_cast<uchar>(static_cast<float>(mat.rows - i) / static_cast<float>(mat.rows) * UCHAR_MAX);
				rgba[3] = saturate_cast<uchar>(0.5 * (rgba[1] + rgba[2]));
			}
		}
	};

	Mat mat(480, 640, CV_8UC4);
	create_alpha_mat(mat);

	vector<int> compression_params;
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);

	try {
		imwrite("transparent_Alpha_image.png", mat, compression_params);
		imshow("generated png image", mat);
		cout << "done!" << endl;
		waitKey();
	} catch (runtime_error& e) {
		cerr << "converting to png format error: " << e.what() << endl;
	} catch (cv::Exception& e) {
		cerr << "cv exceptoin: " << e.what() << endl;
	}
}

// 3.1.9
void test_img_load_show_output()
{
	Mat girl = imread("girl.jpg", 19);
	imshow("1.girl", girl);

	Mat dota = imread("dota.jpg");
	imshow("2.dota", dota);

	Mat logo = imread("dota_logo.jpg");
	imshow("3.logo", logo);

	//Mat roi = dota(Rect(800, 350, logo.cols, logo.rows));
	Mat roi = dota(Range(350, 350 + logo.rows), Range(800, 800 + logo.cols));
	addWeighted(roi, 0.5, logo, 0.3, 0, roi);
	roi = dota(Range(100, 100 + girl.rows), Range(350, 350 + girl.cols));
	addWeighted(roi, 0.5, girl, 0.3, 0, roi);

	imshow("4.blend", dota);

	imwrite("dota_blend.jpg", dota);
	cout << "done!" << endl; 
	waitKey();

}


// 3.2.1
namespace track_bar {

auto win_name = "linear blend";
const int max_alpha = 100;
int g_alpha_value_slider = 0;
double g_alpha_value = 0.0;
double g_beta_value = 0.0;

Mat g_src1 = {};
Mat g_src2 = {};
Mat g_dst = {};

void on_trackbar(int n, void*)
{
	g_alpha_value = g_alpha_value_slider * 1.0 / max_alpha;
	g_beta_value = 1.0 - g_alpha_value;
	addWeighted(g_src1, g_alpha_value, g_src2, g_beta_value, 0.0, g_dst);
	imshow(win_name, g_dst);
}

void test()
{
	g_src1 = imread("1.jpg");
	g_src2 = imread("2.jpg");

	if (!g_src1.data || !g_src2.data) {
		cerr << "read image failed!" << endl;
		return;
	}

	g_alpha_value_slider = 70;

	namedWindow(win_name);
	char name[50] = { 0 };
	sprintf(name, "alpha: %d", max_alpha);
	createTrackbar(name, win_name, &g_alpha_value_slider, max_alpha, on_trackbar);

	on_trackbar(g_alpha_value_slider, nullptr);

	waitKey();
}
}; // end namespace track_bar

namespace mouse_operation {

auto win_name = "mouse operation";
Rect g_rect = {};
bool g_is_drawing_box = false;
RNG g_rng = { 12345 };

void draw_rectangle(Mat& img, Rect box)
{
	rectangle(img, box.tl(), box.br(), Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255)));
}

void on_mouse_event(int e, int x, int y, int flags, void* param)
{
	assert(param);
	Mat& img = *reinterpret_cast<Mat*>(param);
	switch (e) {
	case EVENT_MOUSEMOVE:
		if (g_is_drawing_box) {
			g_rect.width = x - g_rect.x;
			g_rect.height = y - g_rect.height;
		}
		break;

	case EVENT_LBUTTONDOWN:
		g_is_drawing_box = true;
		g_rect = Rect(x, y, 0, 0);
		break;

	case EVENT_LBUTTONUP:
		g_is_drawing_box = false;

		if (g_rect.width < 0) {
			g_rect.x += g_rect.width;
			g_rect.width *= -1;
		}

		if (g_rect.height < 0) {
			g_rect.y += g_rect.height;
			g_rect.height *= -1;
		}

		draw_rectangle(img, g_rect);
		break;

	default:
		break;
	}
}

void test()
{
	g_rect = Rect(-1, -1, 0, 0);

	Mat src(600, 800, CV_8UC3), tmp;
	src.copyTo(tmp);

	src = Scalar::all(0);

	namedWindow(win_name);
	setMouseCallback(win_name, on_mouse_event, reinterpret_cast<void*>(&src));

	while (true) {
		src.copyTo(tmp);

		if (g_is_drawing_box) {
			draw_rectangle(tmp, g_rect);
		}

		imshow(win_name, tmp);

		if (waitKey(10) == 27) {
			break;
		}
	}

}



}; // end namespace mouse_operation



};




int main()
{
	using namespace chapter3;

	//test_imwrite();

	//test_img_load_show_output();

	//track_bar::test();

	mouse_operation::test();


	system("pause");
}

