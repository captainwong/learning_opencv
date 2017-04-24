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

// 5.1.5
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


int main()
{
	access_pixel::test();
}
