#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	cout << "Hello OpenCV!" << endl;

	Mat img = imread("pp.jpg");
	if (img.empty()) {
		cout << "error";
		return -1;
	}
	imshow("ppµÄö¦ÕÕ", img);
	waitKey();
}
