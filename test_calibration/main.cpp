// <Learning opencv> 3rd edition, Gary Bradski & Adrian Kaebler
// Chapter 11: Camera Modules and Calibration
// Putting Calibration All Together

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>

#pragma warning(push)
#pragma warning(disable:4819)
#include <opencv2/opencv.hpp>
#pragma warning(pop)

using namespace std;
using namespace cv;




//int main(int argc, char** argv)
//{
//	int n_boards = 0; // 纸板数量
//	float image_sf = 0.5f;
//	float delay = 1.0f;
//	int board_w = 0;
//	int board_h = 0;
//
//
//	if (argc < 4 || argc > 6) {
//		printf("Error: Wrong number of input params!\n");
//		return -1;
//	}
//
//	board_w = atoi(argv[1]);
//	board_h = atoi(argv[2]);
//	n_boards = atoi(argv[3]);
//
//	int board_n = board_w* board_h; // 纸板像素数量
//	Size board_sz = Size(board_w, board_h);
//	VideoCapture cap;
//	if (!cap.open(0)) {
//		printf("Cannot open camero 0!\n");
//		return -1;
//	}
//
//	namedWindow("Calibration");
//
//	auto image_points = Mat(n_boards * board_n, 2, CV_32FC1);
//	auto object_points = Mat(n_boards * board_n, 3, CV_32FC1);
//	auto point_counts = Mat(n_boards, 1, CV_32SC1);
//	auto intrinsic_matrix = Mat(3, 3, CV_32FC1); // 内参数矩阵
//	auto distortion_coeffs = Mat(5, 1, CV_32FC1); // 畸变系数
//
//	auto corners = new CvPoint2D32f[board_n];
//	int corner_count = 0;
//	int successes = 0;
//	int step = 0, frame = 0;
//	
//	Mat image;
//	cap >> image;
//	auto gray_image = Mat(image.size(), 8, 1);
//
//	// capture corner loop untill we've got n_boards
//	// successful captures (all corners on the board are found)
//	while (successes < n_boards) {
//		// skip every board_dt frames to allow user to move chessboard
//		if (frame++ % board_dt == 0) {
//			// find chessboard corners:
//			int found = findChessboardCorners(image, board_sz, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);
//		}
//	}
//}
