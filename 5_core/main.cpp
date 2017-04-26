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

// 5.1
namespace access_pixel {

const int NITERATIONS = 20;

// over designed, don't do this
typedef std::function<void(Mat&, int)> reduce_f1;
typedef std::function<void(Mat&, Mat&, int)> reduce_f2;

struct color_reduce {
	virtual void operator()(Mat& src, Mat& dst, int div = 64) = 0;
};

struct color_reduce_impl1 : public color_reduce {

	reduce_f1 func_ = {};

	explicit color_reduce_impl1(reduce_f1 func) { func_ = func; }

	virtual void operator()(Mat& src, Mat&, int div = 64) override {
		func_(src, div);
	}
};

struct color_reduce_impl2 : public color_reduce
{
	reduce_f2 func_ = {};

	explicit color_reduce_impl2(reduce_f2 func) { func_ = func; }

	virtual void operator()(Mat& src, Mat& dst, int div = 64) override {
		func_(src, dst, div);
	}
};

using color_reduce_impls = std::pair<string, color_reduce*>;

vector<color_reduce_impls> reduces = {
	{ 
		"【方法一】利用 .ptr 和 []", 
		new color_reduce_impl1([](Mat& src, int div = 64) {
			int lines = src.rows;
			int pixel_per_line = src.cols * src.channels();

			for (int line = 0; line < lines; line++) {
				uchar* data = src.ptr<uchar>(line);

				for (int i = 0; i < pixel_per_line; i++) {
					data[i] = data[i] / div * div + div / 2;
				}
			}
		})
	},

	{
		"【方法二】利用 .ptr 和 *++",
		new color_reduce_impl1([](Mat& src, int div = 64) {
			int lines = src.rows;
			int pixel_per_line = src.cols * src.channels();

			for (int line = 0; line < lines; line++) {
				uchar* data = src.ptr<uchar>(line);

				for (int i = 0; i < pixel_per_line; i++) {
					*data++ = *data / div * div + div / 2;
				}
			}
		})
	},

	{
		"【方法三】利用 .ptr 和 *++ 以及模运算",
		new color_reduce_impl1([](Mat& src, int div = 64) {
			int lines = src.rows;
			int pixel_per_line = src.cols * src.channels();

			for (int line = 0; line < lines; line++) {
				uchar* data = src.ptr<uchar>(line);

				for (int i = 0; i < pixel_per_line; i++) {
					int v = *data;
					*data++ = v - v % div + div / 2;
				}
			}
		})
	},

	{
		"【方法四】利用 .ptr 和 *++ 以及位运算",
		new color_reduce_impl1([](Mat& src, int div = 64) {
			int lines = src.rows;
			int pixel_per_line = src.cols * src.channels();
			int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
			uchar mask = 0xFF << n;

			for (int line = 0; line < lines; line++) {
				uchar* data = src.ptr<uchar>(line);

				for (int i = 0; i < pixel_per_line; i++) {
					*data++ = *data & mask + div / 2;
				}
			}
		})
	},

	{
		"【方法五】利用指针算术运算以及位运算",
		new color_reduce_impl1([](Mat& src, int div = 64) {
			int lines = src.rows;
			int pixel_per_line = src.cols * src.channels();
			int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
			int step = src.step;
			uchar mask = 0xFF << n;
			uchar* data = src.data;

			for (int line = 0; line < lines; line++) {
				for (int i = 0; i < pixel_per_line; i++) {
					*(data + i) = *(data + i) & mask + div / 2;
				}
				data += step;
			}
		})
	},

	{
		"【方法六】利用 .ptr 和 *++ 以及位运算、image.cols * image.channels()",
		new color_reduce_impl1([](Mat& src, int div = 64) {
			int lines = src.rows;
			int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
			uchar mask = 0xFF << n;

			for (int line = 0; line < lines; line++) {
				uchar* data = src.ptr<uchar>(line);
				for (int i = 0; i < src.cols * src.channels(); i++) {
					*data++ = *data & mask + div / 2;
				}
			}
		})
	},

	{
		"【方法七】利用 .ptr 和 *++ 以及位运算(continuous)",
		new color_reduce_impl1([](Mat& src, int div = 64) {
			int lines = src.rows;
			int pixel_per_line = src.cols * src.channels();

			if (src.isContinuous()) {
				pixel_per_line *= lines;
				lines = 1;
			}

			int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
			uchar mask = 0xFF << n;

			for (int line = 0; line < lines; line++) {
				uchar* data = src.ptr<uchar>(line);
				for (int i = 0; i < pixel_per_line; i++) {
					*data++ = *data & mask + div / 2;
				}
			}
		})
	},

	{ 
		// fastest solution
		"【方法八】利用 .ptr 和 *++ 以及位运算(continuous + channels)",
		new color_reduce_impl1([](Mat& src, int div = 64) {
			int lines = src.rows; 
			int cols = src.cols;

			if (src.isContinuous()) {
				cols *= lines;
				lines = 1;
			}

			int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
			uchar mask = 0xFF << n;

			for (int line = 0; line < lines; line++) {
				uchar* data = src.ptr<uchar>(line);
				for (int i = 0; i < cols; i++) {
					*data++ = *data & mask + div / 2;
					*data++ = *data & mask + div / 2;
					*data++ = *data & mask + div / 2;
				}
			}
		})
	},

	{
		"【方法九】利用 Mat_ iterator",
		new color_reduce_impl1([](Mat& src, int div = 64) {
			for (auto iter = src.begin<Vec3b>(); iter != src.end<Vec3b>(); iter++) {
				(*iter)[0] = (*iter)[0] / div * div + div / 2;
				(*iter)[1] = (*iter)[1] / div * div + div / 2;
				(*iter)[2] = (*iter)[2] / div * div + div / 2;
			}
		})
	},

	{
		"【方法十】利用 Mat_ iterator 以及位运算",
		new color_reduce_impl1([](Mat& src, int div = 64) {
			int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
			uchar mask = 0xFF << n;
			for (auto iter = src.begin<Vec3b>(); iter != src.end<Vec3b>(); iter++) {
				(*iter)[0] = (*iter)[0] & mask + div / 2;
				(*iter)[1] = (*iter)[1] & mask + div / 2;
				(*iter)[2] = (*iter)[2] & mask + div / 2;
			}
		})
	},

	{
		"【方法十一】利用 Mat_ iterator 并借助临时 Mat_<Vec3b> 对象",
		new color_reduce_impl1([](Mat& src, int div = 64) {
			Mat_<Vec3b> cimage = src;
			for (auto iter = cimage.begin(); iter != cimage.end(); iter++) {
				(*iter)[0] = (*iter)[0] / div * div + div / 2;
				(*iter)[1] = (*iter)[1] / div * div + div / 2;
				(*iter)[2] = (*iter)[2] / div * div + div / 2;
			}
		})
	},

	{
		"【方法十二】利用动态地址计算配合 at",
		new color_reduce_impl1([](Mat& src, int div = 64) {
			int rows = src.rows;
			int cols = src.cols;

			for (int row = 0; row < rows; row++) {
				for (int col = 0; col < cols; col++) {
					src.at<Vec3b>(row, col)[0] = src.at<Vec3b>(row, col)[0] / div * div + div / 2;
					src.at<Vec3b>(row, col)[1] = src.at<Vec3b>(row, col)[1] / div * div + div / 2;
					src.at<Vec3b>(row, col)[2] = src.at<Vec3b>(row, col)[2] / div * div + div / 2;
				}
			}
		})
	},

	{ 
		"【方法十三】利用图像的输入与输出", 
		new color_reduce_impl2([](Mat& src, Mat& dst, int div = 64) {
			dst.create(src.rows, src.cols, src.type());

			int pixels = src.rows * src.cols;
			int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
			uchar mask = 0xFF << n;

			uchar* data = dst.ptr<uchar>(0);
			const uchar* idata = src.ptr<uchar>(0);

			for (int i = 0; i < pixels; i++) {
				*data++ = (*idata++) & mask + div / 2;
				*data++ = (*idata++) & mask + div / 2;
				*data++ = (*idata++) & mask + div / 2;
			}
		})
	},

};

auto test_one = [](color_reduce* reduce, Mat& src, Mat& dst, int div = 64)
{
	auto time0 = static_cast<double>(getTickCount());
	(*reduce)(src, dst, div);
	time0 = (static_cast<double>(getTickCount()) - time0) / getTickFrequency();
	return time0;
};

void test()
{
	const int WIN_WIDTH = 480;
	const int WIN_HEIGHT = 320;

	Mat origin = imread("1.png");
	namedWindow("origin", WINDOW_NORMAL);
	resizeWindow("origin", WIN_WIDTH, WIN_HEIGHT);
	moveWindow("origin", 0, 0);
	imshow("origin", origin);

	Mat src, dst;
	//origin.copyTo(src);
	dst.create(origin.rows, origin.cols, origin.type());

	//vector<double> t(reduces.size(), 0.0);

	using reduce_speed = std::pair<double, string>;
	vector<reduce_speed> sorted_result;

	int index = 0;
	for (const auto& reduce : reduces) {
		cout << ++index << "/" << reduces.size() << " " << reduce.first << endl;
		double t = 0.0;
		for (int i = 0; i < NITERATIONS; i++) {
			src = imread("1.png");
			t += test_one(reduce.second, src, dst);
		}

		t /= NITERATIONS;
		t *= 1000;

		sorted_result.emplace_back(reduce_speed(t, reduce.first));

		cout << t << "ms" << endl << endl;

		auto winname = reduce.first + " result";

		namedWindow(winname, WINDOW_NORMAL);
		resizeWindow(winname, WIN_WIDTH, WIN_HEIGHT);
		moveWindow(winname, WIN_WIDTH, 0);

		if (typeid(*reduce.second) == typeid(color_reduce_impl1)) {
			imshow(winname, src);
		} else if (typeid(*reduce.second) == typeid(color_reduce_impl2)) {
			imshow(winname, dst);
		} else if (typeid(*reduce.second) == typeid(color_reduce)) {
			cout << "oops" << endl;
		} else {
			assert(0);
		}		

		//cout << "Press a key to continue..." << endl;
		//waitKey();
	}

	std::sort(sorted_result.begin(), sorted_result.end(), [](const reduce_speed& s1, const reduce_speed& s2) {
		return s1.first < s2.first;
	});

	index = 0;
	cout << "sorted result:" << endl;
	cout.setf(ios::fixed);
	for (auto item : sorted_result) {
		cout << "#" << ++index << " " << setprecision(3) << item.first << " " << item.second << endl;
	}

	
	waitKey();
}

}


// 5.2
namespace basic_graph_blend {

bool roi_linear_blend()
{
	Mat src = imread("dota_pa.jpg");
	Mat logo = imread("dota_logo.jpg");
	if (!src.data || !logo.data) {
		return false;
	}

	Mat roi = src(Rect(200, 250, logo.cols, logo.rows));
	addWeighted(roi, 0.5, logo, 0.3, 0, roi);
	imshow("roi linear blend", src);
	return true;
}

bool roi_add_image()
{
	Mat src = imread("dota_pa.jpg");
	Mat logo = imread("dota_logo.jpg");
	if (!src.data || !logo.data) {
		return false;
	}

	Mat roi = src(Rect(200, 250, logo.cols, logo.rows));
	Mat mask = imread("dota_logo.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	logo.copyTo(roi, mask);
	imshow("roi blending", src);
	return true;
}

bool linear_blending()
{
	double alpha = 0.5;
	double beta = 1.0 - alpha;

	Mat src1, src2, dst;
	src1 = imread("mogu.jpg");
	src2 = imread("rain.jpg");
	if (!src1.data || !src2.data) {
		return false;
	}

	addWeighted(src1, alpha, src2, beta, 0, dst);
	imshow("linear blending origin", src1);
	imshow("linear blending result", dst);
	return true;
}

void test()
{
	roi_linear_blend();
	roi_add_image();
	linear_blending();
	waitKey();
}

}


// 5.3
namespace multi_channel_blend {

void test()
{
	const double alpha = 0.5;
	const double beta = 1.0 - alpha;
	const double gamma = 0.;

	Mat src = imread("dota_jugg.jpg");
	Mat logo = imread("dota_logo.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	vector<Mat> channels;
	split(src, channels);

	Mat blue_channel = channels.at(0);
	addWeighted(blue_channel(Rect(500, 250, logo.cols, logo.rows)), 
				alpha, logo, beta, gamma,
				blue_channel(Rect(500, 250, logo.cols, logo.rows)));

	merge(channels, src);
	imshow("origin blue channel merge with logo", src);


	src = imread("dota_jugg.jpg");
	split(src, channels);
	Mat green_channel = channels.at(1);
	addWeighted(green_channel(Rect(500, 250, logo.cols, logo.rows)),
				alpha, logo, beta, gamma,
				green_channel(Rect(500, 250, logo.cols, logo.rows)));
	merge(channels, src);
	imshow("origin green channel merge with logo", src);


	src = imread("dota_jugg.jpg");
	split(src, channels);
	Mat red_channel = channels.at(2);
	addWeighted(red_channel(Rect(500, 250, logo.cols, logo.rows)),
				alpha, logo, beta, gamma,
				red_channel(Rect(500, 250, logo.cols, logo.rows)));
	merge(channels, src);
	imshow("origin red channel merge with logo", src);

	waitKey();
}

}


// 5.4
namespace adjust_contrast_and_bright {

int g_contrast = 0;
int g_bright = 0;
Mat g_src, g_dst;

auto winname_origin = "adjust contrast and bright origin";
auto winname_result = "adjust contrast and bright result";

void on_track_bar(int, void*)
{
	int rows = g_src.rows;
	int cols = g_src.cols;

	if (g_src.isContinuous() && g_dst.isContinuous()) {
		cols *= rows;
		rows = 1;
	}

	for (int y = 0; y < rows; y++) {
		auto sdata = g_src.ptr<uchar>(y);
		auto ddata = g_dst.ptr<uchar>(y);
		for (int x = 0; x < cols; x++) {
			for (int c = 0; c < 3; c++) {
				//g_dst.at<Vec3b>(y, x)[c] = saturate_cast<uchar>((g_contrast*0.01) * (g_src.at<Vec3b>(y, x)[c]) + g_bright);

				*ddata++ = saturate_cast<uchar>((g_contrast*0.01) * (*sdata++) + g_bright);
			}
		}
	}

	imshow(winname_origin, g_src);
	imshow(winname_result, g_dst);
}

void test()
{
	g_src = imread("1.png");

	g_dst = Mat::zeros(g_src.size(), g_src.type());

	g_contrast = 80;
	g_bright = 80;

	namedWindow(winname_origin);
	//resizeWindow(winname_origin, 800, 600);
	moveWindow(winname_origin, 0, 0);
	imshow(winname_origin, g_src);
	
	
	namedWindow(winname_result);
	//resizeWindow(winname_result, 800, 600);
	//moveWindow(winname_result, 805, 0);
	moveWindow(winname_result, g_src.cols, 0);
	createTrackbar("contrast: ", winname_result, &g_contrast, 300, on_track_bar);
	createTrackbar("bright  : ", winname_result, &g_bright, 200, on_track_bar);

	on_track_bar(0, nullptr);
	while ('q' != waitKey());
}

}


// 5.5
namespace discrete_fourier_transform {

void test()
{
	Mat src = imread("dfs.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("origin", src);

	int r = getOptimalDFTSize(src.rows);
	int c = getOptimalDFTSize(src.cols);

	Mat padded;
	copyMakeBorder(src, padded, 0, r - src.rows, 0, c - src.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complex_i;
	merge(planes, 2, complex_i);
	dft(complex_i, complex_i);
	split(complex_i, planes);


	Mat magnituded_img;
	magnitude(planes[0], planes[1], magnituded_img);
	log(magnituded_img + Scalar::all(1), magnituded_img);

	magnituded_img = magnituded_img(Rect(0, 0, magnituded_img.cols & -2, magnituded_img.rows & -2));

	int cx = magnituded_img.cols / 2;
	int cy = magnituded_img.rows / 2;

	Mat q0(magnituded_img, Rect(0, 0, cx, cy));
	Mat q1(magnituded_img, Rect(cx, 0, cx, cy));
	Mat q2(magnituded_img, Rect(0, cy, cx, cy));
	Mat q3(magnituded_img, Rect(cx, cy, cx, cy));

	//swap(q0, q3);
	//swap(q1, q2);

	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(magnituded_img, magnituded_img, 0.0, 1.0, NORM_MINMAX);

	imshow("magnituded image", magnituded_img);

	waitKey();
}

}


// 5.6 
namespace xml_and_yaml_io {

auto filename = "test.yaml";

void test_write()
{
	FileStorage fs(filename, FileStorage::WRITE);
	fs << "frame_count" << 5;
	time_t raw_time;
	time(&raw_time);
	fs << "calibration_date" << asctime(localtime(&raw_time));

	Mat camera_matrix = (Mat_<double>(3, 3) << 1000, 0, 320, 0, 1000, 240, 0, 0, 1);
	Mat dist_coeffs = (Mat_<double>(5, 1) << 0.1, 0.01, -0.001, 0, 0);
	fs << "camera_matrix" << camera_matrix << "dist_coeffs" << dist_coeffs;

	fs << "features" << "[";
	for (int i = 0; i < 3; i++) {
		int x = rand() % 640;
		int y = rand() % 480;
		uchar lbp = rand() % 256;

		fs << "{:" << "x" << x << "y" << y << "lbp" << "[:";
		for (int j = 0; j < 8; j++) {
			fs << ((lbp >> j) & 1);
		}

		fs << "]" << "}";
	}

	fs << "]";
}

void test_read()
{
	FileStorage fs(filename, FileStorage::READ);

	int frame_count = fs["frame_count"];

	string date;
	fs["calibration_date"] >> date;

	Mat camera_matrix, dist_coeffs;
	fs["camera_matrix"] >> camera_matrix;
	fs["dist_coeffs"] >> dist_coeffs;

	cout << "frame count:" << frame_count << endl
		<< "calibration date: " << date << endl
		<< "camera matrix: " << camera_matrix << endl
		<< "distortion coeffs: " << dist_coeffs << endl;

	FileNode features = fs["features"];
	int idx = 0;
	vector<uchar> lbps;

	for (auto iter = features.begin(); iter != features.end(); iter++, idx++) {
		cout << "feature #" << idx << ": ";
		cout << "x=" << (int)(*iter)["x"] << ", y=" << (int)(*iter)["y"] << ", lbp: (";
		(*iter)["lbp"] >> lbps;
		for (auto lbp : lbps) {
			cout << " " << (int)lbp;
		}
		cout << ")" << endl;
	}
}

void test()
{
	test_write();
	system("pause");
	test_read();
	system("pause");
}

}


int main()
{
	//access_pixel::test();

	//basic_graph_blend::test();

	//multi_channel_blend::test();

	//adjust_contrast_and_bright::test();

	//discrete_fourier_transform::test();

	xml_and_yaml_io::test();
}
