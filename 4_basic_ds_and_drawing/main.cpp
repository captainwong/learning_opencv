#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

auto winname1 = "graphic 1";
auto winname2 = "graphic 2";
const int winwidth = 600;


void draw_ellipse(Mat img, double angle)
{
	const int thickness = 2;
	const int line_type = LINE_AA;

	ellipse(img,
			Point(winwidth / 2, winwidth / 2),
			Size(winwidth / 4, winwidth / 16),
			angle, 0, 360,
			Scalar(255, 129, 0),
			thickness,
			line_type);
}

void draw_filled_circle(Mat img, Point center)
{
	const int thickness = -1;
	const int line_type = LINE_AA;

	circle(img, center, winwidth / 32, Scalar(0, 0, 255), thickness, line_type);
}

void draw_polygon(Mat img)
{
	const int line_type = 8;

	const Point pts[20] = {
		Point(winwidth / 4, 7 * winwidth / 8),
		Point(3 * winwidth / 4, 7 * winwidth / 8),
		Point(3 * winwidth / 4, 13 * winwidth / 16),
		Point(11 * winwidth / 16, 13 * winwidth / 16),
		Point(19 * winwidth / 32, 3 * winwidth / 8),
		Point(3 * winwidth / 4, 3 * winwidth / 8),
		Point(3 * winwidth / 4, winwidth / 8),
		Point(26 * winwidth / 40, winwidth / 8),
		Point(26 * winwidth / 40, winwidth / 4),
		Point(22 * winwidth / 40, winwidth / 4),
		Point(22 * winwidth / 40, winwidth / 8),
		Point(18 * winwidth / 40, winwidth / 8),
		Point(18 * winwidth / 40, winwidth / 4),
		Point(14 * winwidth / 40, winwidth / 4),
		Point(14 * winwidth / 40, winwidth / 8),
		Point(winwidth / 4, winwidth / 8),
		Point(winwidth / 4, 3 * winwidth / 8),
		Point(13 * winwidth / 32, 3 * winwidth / 8),
		Point(5 * winwidth / 16, 13 * winwidth / 16),
		Point(winwidth / 4, 13 * winwidth / 16)
	};

	//const Point* ppt[1] = { rook_points[0] };
	const Point* ppt[1] = { pts };
	const int npt[] = { 20 };

	fillPoly(img, ppt, npt, 1, Scalar(255, 255, 255), line_type);
}

void draw_line(Mat img, Point start, Point end)
{
	const int thickness = 2;
	const int line_type = LINE_AA;

	line(img, start, end, Scalar(0, 0, 0), thickness, line_type);
}



int main()
{
	Mat atom_img = Mat::zeros(winwidth, winwidth, CV_8UC3);
	Mat rook_img = Mat::zeros(winwidth, winwidth, CV_8UC3);

	draw_ellipse(atom_img, 90);
	draw_ellipse(atom_img, 0);
	draw_ellipse(atom_img, 45);
	draw_ellipse(atom_img, -45);

	draw_filled_circle(atom_img, Point(winwidth / 2, winwidth / 2));

	
	draw_polygon(rook_img);
	rectangle(rook_img, Point(0, 7 * winwidth / 8), Point(winwidth, winwidth), Scalar(0, 255, 255), -1, LINE_AA);
	
	draw_line(rook_img, Point(0, 15 * winwidth / 16), Point(winwidth, 15 * winwidth / 16));
	draw_line(rook_img, Point(winwidth / 4, 7 * winwidth / 8), Point(winwidth / 4, winwidth));
	draw_line(rook_img, Point(winwidth / 2, 7 * winwidth / 8), Point(winwidth / 2, winwidth));
	draw_line(rook_img, Point(3 * winwidth / 4, 7 * winwidth / 8), Point(3 * winwidth / 4, winwidth));


	imshow(winname1, atom_img);
	moveWindow(winname1, 0, 200);
	imshow(winname2, rook_img);
	moveWindow(winname2, winwidth, 200);

	waitKey();
}