#include "patch.h"
#include "patchmatch.h"
#include <opencv2\opencv.hpp>
#include <ctime>

using namespace std;
using namespace cv;

// the 3-line code below cannot be put in their headers:
// repetively included
// int patch::side_len = 3;  // 5 * 5 patch
int PatchMatch::step = 1;
float PatchMatch::alpha = 0.5;

int main(void)
{
	clock_t start, finish;
	double duration;

	PatchMatch solution("test1.jpg", "test2.jpg");
	INIT_METHOD mode = direct;
	start = clock();
	solution.init(mode);
	cout << "initialization done." << endl;
	solution.propagation_search(mode);
	Mat rst = solution.reshuffle();
	finish = clock();
	duration = static_cast<double>(finish - start) / CLOCKS_PER_SEC;
	cout << duration << " seconds." << endl;
	namedWindow("Display window", CV_WINDOW_AUTOSIZE);
	imshow("Display window", rst);
	// imwrite("D://from_ImageNet/result.jpg", rst);
	waitKey(0);

	system("pause");
	return 0;
}