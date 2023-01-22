#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void overlayImage(Mat* src, Mat* overlay, const Point& location)
{
	for (int y = max(location.y, 0); y < src->rows; ++y) {
		int fY = y - location.y;

		if (fY >= overlay->rows)
			break;

		for (int x = max(location.x, 0); x < src->cols; ++x) {
			int fX = x - location.x;

			if (fX >= overlay->cols)
				break;

			double opacity = ((double)overlay->data[fY * overlay->step + fX * overlay->channels() + 3]) / 255;

			for (int c = 0; opacity > 0 && c < src->channels(); ++c) {
				unsigned char overlayPx = overlay->data[fY * overlay->step + fX * overlay->channels() + c];
				unsigned char srcPx = src->data[y * src->step + x * src->channels() + c];
				src->data[y * src->step + src->channels() * x + c] = srcPx * (1. - opacity) + overlayPx * opacity;
			}
		}
	}
}

int main(int argc, char** argv)
{
	Mat underlay = imread("14447300874345549.png", IMREAD_UNCHANGED);
	Mat overlay = imread("14447300692773615.png", IMREAD_UNCHANGED);
	Mat test = imread("1429209553641108.png", IMREAD_UNCHANGED);

	if (underlay.empty() || overlay.empty() || test.empty()) {
		cout << "Could not read input image files " << endl;
		return -1;
	}

	//Mat rgba[4];
	//split(underlay, rgba);
	//imshow("alpha1.png", rgba[3]);
	//imwrite("alpha1.png", rgba[3]);

	//split(overlay, rgba);
	//imshow("alpha2.png", rgba[3]);
	//imwrite("alpha2.png", rgba[3]);

	overlayImage(&underlay, &overlay, Point());
	//overlayImage(&test, &underlay, Point(120, 180));

	//split(underlay, rgba);
	//imshow("alpha3.png", rgba[3]);
	//imwrite("alpha3.png", rgba[3]);

	imshow("result1", underlay);
	//imwrite("result1.png", underlay);
	//imshow("result2", test);
	//imwrite("result2.png", test);

	waitKey();

	return 0;
}