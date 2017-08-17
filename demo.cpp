#include <opencv2\opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <string>

using namespace std;
using namespace cv;

/*
int main()
{
	const string s = "D://1.jpg";
	Mat img = imread(s);
	if (img.empty())
	{
		cout << "error";
		system("pause");
		return -1;
	}
	namedWindow("Display window", CV_WINDOW_AUTOSIZE);
	imshow("Display window", img);
	waitKey(0);  // waitKey(2000) will generate an error!
	system("pause");
	return 0;
}
*/

int main(void)
{
	Mat src1, src2, a, b, temp;
	src1 = imread("D://from_ImageNet/marble1.jpg");
	src2 = imread("D://from_ImageNet/marble2.jpg");
	cout << src1.rows << ", " << src1.cols << endl;
	cout << src2.rows << ", " << src2.cols << endl;
	a = src1(Rect(0, 0, 175, 190));
	b = src2(Rect(0, 0, 175, 190));
	cout << a.rows << ", " << a.cols << endl;
	absdiff(a, b, temp);
	cout << sum(temp) << endl;
	cout << sum(sum(temp))[0] << endl;
	Scalar s = sum(temp);
	cout << s << endl;

	system("pause");
	return 0;
}
