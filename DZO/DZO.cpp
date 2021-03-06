#include <iostream>
#include <opencv2/opencv.hpp>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;
using namespace cv;

Vec3b bilinInterpol(const Mat& srcImg, float x, float y) {
	if (x - 1 < 0 || x + 1 > srcImg.cols || y - 1 < 0 || y + 1 > srcImg.rows) {
		return srcImg.at<Vec3b>(y, x);
	}

	Vec3b v1 = srcImg.at<Vec3b>(floor(y), floor(x));
	Vec3b v2 = srcImg.at<Vec3b>(ceil(y), floor(x));
	Vec3b v3 = srcImg.at<Vec3b>(floor(y), ceil(x));
	Vec3b v4 = srcImg.at<Vec3b>(ceil(y), ceil(x));

	double tmp;
	double q1 = (1 - modf(x, &tmp)) * (1 - modf(y, &tmp));
	double q2 = (1 - modf(x, &tmp)) * (modf(y, &tmp));
	double q3 = (modf(x, &tmp)) * (1 - modf(y, &tmp));
	double q4 = (modf(x, &tmp)) * (modf(y, &tmp));

	return Vec3b((v1 * q1) + (v2 * q2) + (v3 * q3) + (v4 * q4));
}

struct RLDUserData {
	cv::Mat& src_8uc3_img;
	cv::Mat& undistorted_8uc3_img;
	int K1;
	int K2;

	RLDUserData(const int K1, const int K2, cv::Mat& src_8uc3_img, cv::Mat& undistorted_8uc3_img) : K1(K1), K2(K2), src_8uc3_img(src_8uc3_img), undistorted_8uc3_img(undistorted_8uc3_img) {

	}
};

void geom_dist(cv::Mat& src_8uc3_img, cv::Mat& dst_8uc3_img, bool bili, double K1 = 1.0, double K2 = 1.0)
{
	float C_u = src_8uc3_img.cols / 2;
	float C_v = src_8uc3_img.rows / 2;
	float R = sqrt(pow(C_u, 2) + pow(C_v, 2));
	for (int r = 0; r < src_8uc3_img.rows; r++) {
		for (int c = 0; c < src_8uc3_img.cols; c++) {
			float X_n = c - C_u;
			float Y_n = r - C_v;
			float x_ = X_n / R;
			float y_ = Y_n / R;
			float r_2 = pow(x_, 2) + pow(y_, 2);
			float fi = 1 + K1 * r_2 + K2 * pow(r_2, 2);
			float X_d = (X_n * pow(fi, -1)) + C_u;
			float Y_d = (Y_n * pow(fi, -1)) + C_v;
			//int Y = Y_d;
			//int X = X_d;
			if (Y_d >= src_8uc3_img.rows || X_d >= src_8uc3_img.cols || X_d < 0 || Y_d < 0)
				continue;
			//Vec3b src = src_8uc3_img.at<Vec3b>(Y, X);
			//dst_8uc3_img.at<Vec3b>(r, c) = src_8uc3_img.at<Vec3b>(Y, X);
			dst_8uc3_img.at<Vec3b>(r, c) = bilinInterpol(src_8uc3_img, X_d, Y_d);
		}
	}
}

void apply_rld(int id, void* user_data)
{
	RLDUserData* rld_user_data = (RLDUserData*)user_data;

	geom_dist(rld_user_data->src_8uc3_img, rld_user_data->undistorted_8uc3_img, !false, rld_user_data->K1 / 100.0, rld_user_data->K2 / 100.0);
	cv::imshow("Geom Dist", rld_user_data->undistorted_8uc3_img);
}

void convulation(Mat& img) {
	Mat con = Mat(Size(3,3), img.type(), 1);
	Mat tmp;
	for (int c = 1; c < img.cols-1; c++) {
		for (int r = 1; r < img.rows - 1; r++) {
			tmp = (img(cv::Rect(Point(c - 1, r - 1), Point(c + 2, r + 2))));
			//cout << (img(cv::Rect(Point(0, 0), Point(c + 2, r + 2)))) << endl;
			img.at<uchar>(r, c) = (tmp.dot(con) / 9 );
			//cout << (tmp.dot(con) / 9) << endl;
		}
	}
}

template <typename T>
void convulation(const Mat& srcImg, Mat& dstImg , Mat& convMask, T convK) {
	Mat matInImg;
	//Size sizeOfMath = convMask.size();
	for (int r = (convMask.rows / 2); r < srcImg.rows - (convMask.rows / 2); r++) {
		for (int c = (convMask.cols / 2); c < srcImg.cols - (convMask.cols / 2); c++) {
			matInImg = (srcImg(cv::Rect(Point(c - 1, r - 1), Point(c + 2, r + 2))));
			dstImg.at<T>(r, c) = (matInImg.dot(convMask) / convK);
		}
	}
}

void ant(Mat img, Mat& out, int iteration) {
	double sigma = 0.015;
	double lambda = 0.1;

	for (int iter = 0; iter < iteration; iter++) {
		for (int r = 1; r < img.rows - 1; r++) {
			for (int c = 1; c < img.cols - 1; c++) {
				double in = img.at<double>(r, c - 1) - img.at<double>(r, c);
				double is = img.at<double>(r, c + 1) - img.at<double>(r, c);
				double ie = img.at<double>(r + 1, c) - img.at<double>(r, c);
				double iw = img.at<double>(r - 1, c) - img.at<double>(r, c);
		
				double cn = (exp(-1 * (pow(abs(in), 2) / pow(sigma, 2))));
				double cs = (exp(-1 * (pow(abs(is), 2) / pow(sigma, 2))));
				double ce = (exp(-1 * (pow(abs(ie), 2) / pow(sigma, 2))));
				double cw = (exp(-1 * (pow(abs(iw), 2) / pow(sigma, 2))));

				out.at<double>(r, c) = (img.at<double>(r, c) * (1 - lambda * (cn + cs + ce + cw)) + lambda * 
					(cn * img.at<double>(r, c - 1) + cs * img.at<double>(r, c + 1) + ce * img.at<double>(r + 1, c) + cw * img.at<double>(r - 1, c)));
			}
		}
		img = out;
	}
}

void fourierTransform(Mat& srcImg, Mat* outImgArray) {
	// Norm sqrt NM
	Mat DFT;
	double normMN = 1.0 / sqrt(srcImg.rows * srcImg.cols);
	srcImg.convertTo(DFT, CV_64FC1, (1.0 / 255.0) * normMN);

	// Base matrix - complex values
	Mat baseMatrix = Mat(DFT.size(), CV_64FC2);
	// Amplitude matrix
	Mat amp = Mat(DFT.size(), CV_64FC1);
	// Phase matrix
	Mat phase = Mat(DFT.size(), CV_64FC1);
	// Spectrum matrix
	Mat powerSpec = Mat(DFT.size(), CV_64FC1);

	for (int r = 0; r < DFT.rows; r++) {
		for (int c = 0; c < DFT.cols; c++) {

			// Real and imagine part
			double realP = 0;
			double imagin = 0;

			for (int m = 0; m < DFT.rows; m++) {
				for (int n = 0; n < DFT.cols; n++) {
					realP += cos(-(2 * M_PI) * ((((double)(c * n) / (DFT.cols)) + ((double)(r * m) / (DFT.rows))))) * DFT.at<double>(m, n);
					imagin += sin(-(2 * M_PI) * ((((double)(c * n) / (DFT.cols)) + ((double)(r * m) / (DFT.rows))))) * DFT.at<double>(m, n);
				}
			}

			// Add real and imagine part to base matrix
			baseMatrix.at<Vec2d>(r, c) = Vec2d{ realP, imagin };

			// Calculate amplitude
			amp.at<double>(r, c) = sqrt(pow(realP, 2) + pow(imagin, 2));

			// Calculate phase
			phase.at<double>(r, c) = atan((imagin / realP));

			// Calculate power spectrum
			powerSpec.at<double>(r, c) = pow(amp.at<double>(r, c), 2);
		}
	}
	outImgArray[0] = baseMatrix;
	outImgArray[1] = amp;
	outImgArray[2] = phase;
	outImgArray[3] = powerSpec;
}

void swapQuadrants(Mat& img) {
	int cx = img.cols / 2;
	int cy = img.rows / 2;

	Mat q0(img, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(img, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(img, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(img, Rect(cx, cy, cx, cy)); // Bottom-Right

	// swap quadrants (Top-Left with Bottom-Right)
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	// swap quadrant (Top-Right with Bottom-Left)
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

void showPowerSpec(Mat& powerSpec) {
	// Change to logaritmic scale
	log(powerSpec, powerSpec);

	swapQuadrants(powerSpec);

	normalize(powerSpec, powerSpec, 0, 1, NORM_MINMAX);

	imshow("Power spectre", powerSpec);
}

Mat reverseFourierTransform(const Mat& srcImg) {
	Mat outImg = Mat(srcImg.size(), CV_64FC1);

	double normMN = 1.0 / sqrt(srcImg.rows * srcImg.cols);

	for (int r = 0; r < srcImg.rows; r++) {
		for (int c = 0; c < srcImg.cols; c++) {
			double realP = 0;
			for (int m = 0; m < srcImg.rows; m++) {
				for (int n = 0; n < srcImg.cols; n++) {
					double baseReal = cos((2 * M_PI) * ((((double)(c * n) / (srcImg.cols)) + ((double)(r * m) / (srcImg.rows))))) * normMN;
					double baseImagin = sin((2 * M_PI) * ((((double)(c * n) / (srcImg.cols)) + ((double)(r * m) / (srcImg.rows))))) * normMN;
					realP += (srcImg.at<Vec2d>(m, n)[0] * baseReal) - (srcImg.at<Vec2d>(m, n)[1] * baseImagin);
				}
			}
			outImg.at<double>(r, c) = realP;
		}
	}
	return outImg;
}

void lowPass(Mat& baseMatrix, int radius) {
	Mat mask = Mat(baseMatrix.size(), CV_8UC1, Scalar(0));
	cv::circle(mask, Point2f(baseMatrix.cols / 2, baseMatrix.rows / 2), radius, cv::Scalar(255), -1);
	swapQuadrants(baseMatrix);
	for (int r = 0; r < mask.rows; r++) {
		for (int c = 0; c < mask.cols; c++) {
			if (mask.at<uchar>(r, c) < 1){
				baseMatrix.at<cv::Vec2d>(r, c) = cv::Vec2d(0, 0);
			}
		}
	}
	swapQuadrants(baseMatrix);
}

void highPass(Mat& baseMatrix, int radius) {
	Mat mask = Mat(baseMatrix.size(), CV_8UC1, Scalar(0));
	cv::circle(mask, Point2f(baseMatrix.cols / 2, baseMatrix.rows / 2), radius, cv::Scalar(255), -1);
	swapQuadrants(baseMatrix);
	for (int r = 0; r < mask.rows; r++) {
		for (int c = 0; c < mask.cols; c++) {
			if (mask.at<uchar>(r, c) == 1) {
				baseMatrix.at<cv::Vec2d>(r, c) = cv::Vec2d(0, 0);
			}
		}
	}
	swapQuadrants(baseMatrix);
}

void pixelPass(Mat& baseMatrix) {
	Mat mask = Mat(baseMatrix.size(), CV_8UC1, Scalar(0));

	rectangle(mask, Point(2, mask.rows/2 - 2), Point(30, mask.rows/2 + 2), Scalar(255), -1);
	rectangle(mask, Point(36, mask.rows / 2 - 2), Point(64, mask.rows / 2 + 2), Scalar(255), -1);

	imshow("mask", mask);
	swapQuadrants(baseMatrix);

	for (int r = 0; r < mask.rows; r++) {
		for (int c = 0; c < mask.cols; c++) {
			if (mask.at<uchar>(r, c) == 255) {
				baseMatrix.at<cv::Vec2d>(r, c) = cv::Vec2d(0, 0);
			}
		}
	}
	swapQuadrants(baseMatrix);
}

int* getAndShowHisto(Mat& srcImg, string imgName) {
	int cntOfVals[256] = { 0 };		// Count pixels with same values
	int cntOfCumVals[256] = { 0 };	// Count of cumulative distribution function

	for (int r = 0; r < srcImg.rows; r++) {
		for (int c = 0; c < srcImg.cols; c++) {
			cntOfVals[srcImg.at<uchar>(r, c)]++;
		}
	}

	// Find max val in array and create matrix (256 x maxOccurrence) - ITS NOT OK! Matrix will grow so much with bigger img 
	Mat histo = Mat(Size(256, *max_element(cntOfVals, cntOfVals + 255)), CV_8UC1, Scalar(0));
	// Cumulative distribution function. Same problem SIZE OF MATRIX!!!
	Mat cumHisto = Mat(Size(256, (srcImg.rows * srcImg.cols)), CV_8UC1, Scalar(0));
	for (int i = 0; i < 256; i++) {
		unsigned int tmp = 0;
		for (int c = 0; c < i; c++) {
			tmp += cntOfVals[c];
		}
		cntOfCumVals[i] = tmp;
		line(cumHisto, Point(i, cumHisto.rows), Point(i, cumHisto.rows - tmp), Scalar(255), 1);
		line(histo, Point(i, histo.rows), Point(i, histo.rows - cntOfVals[i]), Scalar(255), 1);
	}

	resize(cumHisto, cumHisto, Size(256, 500));
	resize(histo, histo, Size(256, 500));

	imshow(imgName + " Cumuluative Histogram", cumHisto);
	imshow(imgName + " Histogram", histo);

	return cntOfCumVals;
}

void histogramEqualisation(Mat& srcImg, Mat& dstImg, int* cumulativeDisFun, int L) {
	//int L = 251;
	for (int r = 0; r < srcImg.rows; r++) {
		for (int c = 0; c < srcImg.cols; c++) {
			int cdf_min = *min_element(cumulativeDisFun, cumulativeDisFun + 255);
			dstImg.at<uchar>(r, c) = round((((double)cumulativeDisFun[srcImg.at<uchar>(r, c)] - cdf_min) /
				((srcImg.cols * srcImg.rows) - cdf_min)) *
				(L - 1)
			);
		}
	}
}

void projectiveTransform(const Mat& srcImg, Mat& dstImg, Point2d* mapPointsFrom, Point2d* mapPointsTo) {
	Mat A = Mat(Size(8, 8), CV_64FC1);
	Mat B = Mat(Size(1, 8), CV_64FC1);

	for (int i = 0; i < 4; i++) {
		int row = i * 2;

		A.at<double>(row, 0) = mapPointsTo[i].y;
		A.at<double>(row, 1) = 1;
		A.at<double>(row, 2) = 0;
		A.at<double>(row, 3) = 0;
		A.at<double>(row, 4) = 0;
		A.at<double>(row, 5) = -mapPointsFrom[i].x * mapPointsTo[i].x;
		A.at<double>(row, 6) = -mapPointsFrom[i].x * mapPointsTo[i].y;
		A.at<double>(row, 7) = -mapPointsFrom[i].x;

		B.at<double>(row, 0) = -mapPointsTo[i].x;

		A.at<double>(row + 1, 0) = 0;
		A.at<double>(row + 1, 1) = 0;
		A.at<double>(row + 1, 2) = mapPointsTo[i].x;
		A.at<double>(row + 1, 3) = mapPointsTo[i].y;
		A.at<double>(row + 1, 4) = 1;
		A.at<double>(row + 1, 5) = -mapPointsFrom[i].y * mapPointsTo[i].x;
		A.at<double>(row + 1, 6) = -mapPointsFrom[i].y * mapPointsTo[i].y;
		A.at<double>(row + 1, 7) = -mapPointsFrom[i].y;

		B.at<double>(row + 1, 0) = 0;
	}

	Mat res;
	solve(A, B, res);
	Mat perMat = Mat(Size(3, 3), CV_64FC1);
	perMat.at<double>(0, 0) = 1; 

	for (int i = 1; i < 9; i++) {
		perMat.at<double>(i) = res.at<double>(i - 1);
	}

	Mat transMat = cv::Mat::eye(3, 3, CV_64FC1); 
	transMat = transMat * perMat; 

	for (int x = 0; x < dstImg.rows; x++) {
		for (int y = 0; y < dstImg.cols; y++) {
			Mat homPos = Mat(Matx31d(x, y, 1.0f)); 
			Mat tranPosMat = transMat * homPos; 

			Vec2d tranPos = Vec2d(tranPosMat.at<double>(0, 0) / tranPosMat.at<double>(2, 0), tranPosMat.at<double>(1, 0) / tranPosMat.at<double>(2, 0)); 

			if (tranPos.val[0] >= 0 &&
				tranPos.val[0] < srcImg.rows &&
				tranPos.val[1] >= 0 &&
				tranPos.val[1] < srcImg.cols)
			{
				dstImg.at<Vec3b>(x, y) = srcImg.at<Vec3b>(tranPos.val[0], tranPos.val[1]);
			}
		}
	}
}

static void exercise7() {
	Mat uneq = imread("images/uneq.jpg", IMREAD_GRAYSCALE);
	if (uneq.empty())
		cout << "Faild to load img" << endl;

	imshow("No equalized img", uneq);
	histogramEqualisation(uneq, uneq, getAndShowHisto(uneq, "input"), 251);
	imshow("Equalized img", uneq);
	getAndShowHisto(uneq, "output");
}

static void exercise8() {
	Mat flag = imread("images/flag.png", IMREAD_COLOR);
	Mat vsb = imread("images/vsb.jpg", IMREAD_COLOR);

	Point2d to[4];
	Point2d from[4];
	from[0] = Point2d(0, 0);
	from[1] = Point2d(215, 0);
	from[2] = Point2d(215, 323);
	from[3] = Point2d(0, 323);

	to[0] = Point2d(107, 69);
	to[1] = Point2d(134, 66);
	to[2] = Point2d(122, 228);
	to[3] = Point2d(76, 227);

	projectiveTransform(flag, vsb, from, to);
	imshow("vsb", vsb);
}

static void exercise6(Mat& src_8uc3_img1, Mat& geom_8uc3_img, RLDUserData& rld_user_data) {

	src_8uc3_img1 = cv::imread("images/distorted_window.jpg", cv::IMREAD_COLOR);
	if (src_8uc3_img1.empty())
	{
		printf("Unable to load image!\n");
		exit(-1);
	}

	cv::namedWindow("Original Image");
	cv::imshow("Original Image", src_8uc3_img1);

	//geom_8uc3_img = src_8uc3_img1.clone();
	geom_8uc3_img = Mat(src_8uc3_img1.size(), src_8uc3_img1.type(), Scalar(0, 0, 0));
	apply_rld(0, (void*)&rld_user_data);

	cv::namedWindow("Geom Dist");
	cv::imshow("Geom Dist", geom_8uc3_img);

	cv::createTrackbar("K1", "Geom Dist", &rld_user_data.K1, 100, apply_rld, &rld_user_data);
	cv::createTrackbar("K2", "Geom Dist", &rld_user_data.K2, 100, apply_rld, &rld_user_data);
}

void sobel(const Mat& srcImg, Mat& dstImg) {
	Mat G_x = Mat(srcImg.size(), srcImg.type());
	Mat G_y = Mat(srcImg.size(), srcImg.type());
	Mat mask = Mat(Size(3, 3), CV_64FC1);

	// (row,col)
	mask.at<double>(0, 0) = 1.0;
	mask.at<double>(0, 1) = 0;
	mask.at<double>(0, 2) = -1.0;

	mask.at<double>(1, 0) = 2.0;
	mask.at<double>(1, 1) = 0;
	mask.at<double>(1, 2) = -2.0;

	mask.at<double>(2, 0) = 1.0;
	mask.at<double>(2, 1) = 0;
	mask.at<double>(2, 2) = -1.0;

	convulation<double>(srcImg, G_x, mask, 1.0);
	transpose(mask, mask);
	convulation<double>(srcImg, G_y, mask, 1.0);
	
	magnitude(G_x, G_y, dstImg);
}

static void exercise9() {
	Mat valve = cv::imread("images/valve.png", IMREAD_GRAYSCALE);
	if (valve.empty()) {
		cout << "Faild to load valve img!" << endl;
		return;
	}
	imshow("valve orig", valve);
	valve.convertTo(valve, CV_64FC1, 1.0 / 255.0);
	sobel(valve, valve);
	imshow("Valve sobel", valve);
}

// Without linear interpolation and fixed values
void nonMaximaSuppression(const Mat& srcImg, Mat& dstImg) {
	Mat mag = Mat(srcImg.rows, srcImg.cols, srcImg.type());
	for (int r = 1; r < srcImg.rows - 1; r++) {
		for (int c = 1; c < srcImg.cols - 1; c++) {
			double F_x = (srcImg.at<double>(r, c - 1) - srcImg.at<double>(r, c + 1)) / 2;
			double F_y = (srcImg.at<double>(r - 1, c) - srcImg.at<double>(r + 1, c)) / 2;
			mag.at<double>(r, c) = sqrt((F_x * F_x) + (F_y * F_y));
		}
	}
	//Mat res = Mat(srcImg.rows, srcImg.cols, srcImg.type());
	for (int r = 1; r < mag.rows - 1; r++) {
		for (int c = 1; c < mag.cols - 1; c++) {
			if ((mag.at<double>(r, c) > mag.at<double>(r, c + 1)) && (mag.at<double>(r, c) < mag.at<double>(r, c - 1))) {
				dstImg.at<double>(r, c) = 1.0;
			}
			else {
				dstImg.at<double>(r, c) = 0;
			}
		}
	}
}

// Without linear interpolation
void nonMaximaSuppressionDoubleThres(const Mat& srcImg, Mat& dstImg) {
	Mat mag = Mat(srcImg.rows, srcImg.cols, srcImg.type());
	for (int r = 1; r < srcImg.rows - 1; r++) {
		for (int c = 1; c < srcImg.cols - 1; c++) {
			double F_x = (srcImg.at<double>(r, c - 1) - srcImg.at<double>(r, c + 1)) / 2;
			double F_y = (srcImg.at<double>(r - 1, c) - srcImg.at<double>(r + 1, c)) / 2;
			mag.at<double>(r, c) = sqrt((F_x * F_x) + (F_y * F_y));
		}
	}
	//Mat res = Mat(srcImg.rows, srcImg.cols, srcImg.type());
	for (int r = 1; r < mag.rows - 1; r++) {
		for (int c = 1; c < mag.cols - 1; c++) {
			if ((mag.at<double>(r, c) > 0.05) && (mag.at<double>(r, c) < 0.1)) {
				dstImg.at<double>(r, c) = 1.0;
			}
			else {
				dstImg.at<double>(r, c) = 0;
			}
		}
	}
}

static void exercise10() {
	Mat valve = cv::imread("images/valve.png", IMREAD_GRAYSCALE);
	valve.convertTo(valve, CV_64FC1, 1.0 / 255.0);
	//Mat valveEdit = Mat(valveEdit.size(), valveEdit.type());
	if (valve.empty()) {
		cout << "Faild to load valve img!" << endl;
	}

	//nonMaximaSuppression(valve, valve);
	nonMaximaSuppressionDoubleThres(valve, valve);
	imshow("mag", valve);
}

int main()
{
	cv::Mat src_8uc3_img = cv::imread("images/lena.png", IMREAD_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	
	if (src_8uc3_img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
	}

#pragma region exercise 1

	Mat moon = imread("images/moon.jpg", IMREAD_GRAYSCALE);

	if (moon.empty())
		cout << "Faild to load img" << endl;

	double min, max;
	Point minLoc;
	Point maxLoc;
	cv::minMaxLoc(moon, &min, &max, &minLoc, &maxLoc);		// Find min and max in img
	cout << "min val: " << min << endl;		// 101
	cout << "max val: " << max << endl;		// 160

	int oldRange = max - min;
	float q = 255 / (float)oldRange;	// Quotient of range

	for (int r = 0; r < moon.rows; r++) {
		for (int c = 0; c < moon.cols; c++) {
			moon.at<uchar>(r, c) = round((moon.at<uchar>(r, c) - min) * q);		// Set new range 0-255
		}
	}
	//imshow("new m", moon);

	moon.convertTo(moon, CV_32FC1, 1.0 / 255.0);	// Convert to float
	cv::pow(moon, 1.5, moon);		// Power all pixel by some value
	moon.convertTo(moon, CV_8UC1, 255 / 1.0);		// Convert back to int
	//imshow("Gama", moon);
	
#pragma endregion Gamma correction

#pragma region exercise 2

	//imshow("moon orig", moon);
	//Mat con = Mat(Size(3, 3), moon.type(), 1);
	//convulation<uchar>(moon, moon, con, 9);
	//imshow("con", moon);

#pragma endregion convolution

#pragma region exercise 3

	cv::Mat src_8uc1_img_gray = cv::imread("images/lena.png", IMREAD_GRAYSCALE);
	Mat lena;
	src_8uc1_img_gray.copyTo(lena);
	lena.convertTo(lena, CV_64FC1, 1.0 / 255.0);

	Mat lenaNew = Mat(lena.size(), lena.type(), 0.f);
	//ant(lena, lenaNew, 100);
	//imshow("lena new", lenaNew);
	
#pragma endregion Anisotropic filtering
	
#pragma region exercise 4-5

	cv::Mat lena64 = cv::imread("images/lena64_bars.png", IMREAD_GRAYSCALE);
	Mat partsOfDFT[4];
	//fourierTransform(lena64, partsOfDFT);
	//showPowerSpec(partsOfDFT[3]);
	//lowPass(partsOfDFT[0], 20);
	//pixelPass(partsOfDFT[0]);
	//imshow("Back to img", reverseFourierTransform(partsOfDFT[0]));
	//showPowerSpec(partsOfDFT[3]);
	//imshow("phase", partsOfDFT[2]);

#pragma endregion Discrete Fourier Transform, inverse Discrete Fourier Transform and filtering in the frequency domain

#pragma region exercise 6
	
	cv::Mat src_8uc3_img1, geom_8uc3_img;
	RLDUserData rld_user_data(3.0, 1.0, src_8uc3_img1, geom_8uc3_img);
	exercise6(src_8uc3_img1, geom_8uc3_img, rld_user_data);

#pragma endregion Simple removal of geometric distortion

#pragma region exercise 7

	//exercise7();

#pragma endregion Histogram equalisation

#pragma region exercise 8
	
	//exercise8();

#pragma endregion Perspective transform

#pragma region exercise 9

	//exercise9();

#pragma endregion Sobel operator

#pragma region exercise 10

	//exercise10();

#pragma endregion Edge Thinning and Double Thresholding

	/*
	cv::Mat gray_8uc1_img; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 8 bits (uchar)
	cv::Mat gray_32fc1_img; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 32 bits (float)

	cv::cvtColor(src_8uc3_img, gray_8uc1_img, COLOR_BGRA2GRAY); // convert input color image to grayscale one, CV_BGR2GRAY specifies direction of conversion
	gray_8uc1_img.convertTo(gray_32fc1_img, CV_32FC1, 1.0 / 255.0); // convert grayscale image from 8 bits to 32 bits, resulting values will be in the interval 0.0 - 1.0

	int x = 10, y = 15; // pixel coordinates

	uchar p1 = gray_8uc1_img.at<uchar>(y, x); // read grayscale value of a pixel, image represented using 8 bits
	float p2 = gray_32fc1_img.at<float>(y, x); // read grayscale value of a pixel, image represented using 32 bits
	cv::Vec3b p3 = src_8uc3_img.at<cv::Vec3b>(y, x); // read color value of a pixel, image represented using 8 bits per color channel

	// print values of pixels
	printf("p1 = %d\n", p1);
	printf("p2 = %f\n", p2);
	printf("p3[ 0 ] = %d, p3[ 1 ] = %d, p3[ 2 ] = %d\n", p3[0], p3[1], p3[2]);

	gray_8uc1_img.at<uchar>(y, x) = 0; // set pixel value to 0 (black)

	// draw a rectangle
	cv::rectangle(gray_8uc1_img, cv::Point(65, 84), cv::Point(75, 94),
		cv::Scalar(50), cv::FILLED);

	// declare variable to hold gradient image with dimensions: width= 256 pixels, height= 50 pixels.
	// Gray levels wil be represented using 8 bits (uchar)
	cv::Mat gradient_8uc1_img(50, 256, CV_8UC1);

	// For every pixel in image, assign a brightness value according to the x coordinate.
	// This wil create a horizontal gradient.
	for (int y = 0; y < gradient_8uc1_img.rows; y++) {
		for (int x = 0; x < gradient_8uc1_img.cols; x++) {
			gradient_8uc1_img.at<uchar>(y, x) = x;
		}
	}

	// diplay images
	cv::imshow("Gradient", gradient_8uc1_img);
	cv::imshow("Lena gray", gray_8uc1_img);
	cv::imshow("Lena gray 32f", gray_32fc1_img);
	*/

	cv::waitKey(0); // wait until keypressed
	//destroyAllWindows();

	return 0;
}

