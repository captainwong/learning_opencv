#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

auto image_name = "1.jpg";
auto video_name = "1.avi";

// 1.5.1
void hello_opencv(const Mat& src)
{
	imshow("origin", src);
}

// 1.5.2
void test_erode(const Mat& src)
{
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	Mat dst;
	erode(src, dst, element);
	imshow("eroded", dst);
}

// 1.5.3
void test_blur(const Mat& src)
{
	Mat dst;
	blur(src, dst, Size(7, 7));
	imshow("blurred", dst);
}

// 1.5.4
void test_canny(const Mat& src)
{
	Mat edge, gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	blur(gray, edge, Size(3, 3));
	Canny(edge, edge, 3, 9, 3);
	imshow("Cannied", edge);
}

// 1.6.1
void test_play_video()
{
	VideoCapture capture(video_name);
	Mat frame;
	while (true) {
		capture >> frame;
		if (frame.empty()) {
			break;
		}

		imshow("video", frame);
		waitKey(30);
	}
}

// 1.6.2
void test_capture_camera()
{
	VideoCapture capture(0);
	Mat edges, frame;

	while (true) {
		capture >> frame;

		test_canny(frame);

		if (waitKey(30) >= 0) {
			break;
		}
	}
}

int main()
{
	cout << "Hello OpenCV! " << CV_VERSION << endl;

	Mat img = imread(image_name);
	hello_opencv(img);
	test_erode(img);
	test_blur(img);
	//test_canny(img);

	//test_play_video();

	test_capture_camera();
}
