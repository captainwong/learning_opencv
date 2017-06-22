#define _CRT_SECURE_NO_WARNINGS

#include <iostream>

#pragma warning(push)
#pragma warning(disable:4819)
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#pragma warning(pop)

#include <functional>
#include <vector>
#include <iomanip>
#include <random>
#include <sal.h>



using namespace std;
using namespace cv;
using namespace xfeatures2d;

// 11.1.6
namespace SURF_feature_detector {

void test(const char* img1, const char* img2)
{
	Mat src1 = imread(img1);
	Mat src2 = imread(img2);

	imshow("origin1", src1);
	imshow("origin2", src2);

	int min_hessian = 400;
	auto f2d = SURF::create(min_hessian);
	std::vector<KeyPoint> kpv1, kpv2;
	f2d->detect(src1, kpv1);
	f2d->detect(src2, kpv2);

	Mat kpm1, kpm2;
	drawKeypoints(src1, kpv1, kpm1);
	drawKeypoints(src2, kpv2, kpm2);

	imshow("SURF 1", kpm1);
	imshow("SURF 2", kpm2);
	waitKey();
}

}



int main()
{
	SURF_feature_detector::test("11.1.6-1.jpg", "11.1.6-2.jpg");
}