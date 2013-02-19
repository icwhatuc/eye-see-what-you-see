#include <cv.h> 
#include <highgui.h> 
#include <cstdio>  
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <ml.h>
#include <sys/time.h>
#include <unistd.h>
#include <cmath>
#include "opencv2/gpu/gpu.hpp"
#include "eyepair.h"

#define CAM_WIDTH   1920
#define CAM_HEIGHT  1080
#define HIST_BINS   256
#define HIST_THRESHOLD 0.001

// SVM and Haar
#define SVM_IMG_SIZE 75
#define SVM_FILE          "eye_classify_withweights.svm"
#define HAAR_EYE_FILE     "cascades/haarcascade_eye.xml"
#define HAAR_EYEPAIR_FILE "cascades/haarcascade_eyepair.xml"

// Window names
#define W_COLOR  "color"
#define W_DIFF   "difference"
#define W_THRESH "thresholded difference"

using namespace cv;
using namespace std;

void timing(bool start, string what="");

int main(int argc, const char *argv[])
{
	// Initialize video source
	VideoCapture capture;
	if (argc == 1)
		capture.open(0); 
	else
		capture.open(argv[1]); // If no input file specified, live capture

	if (!capture.isOpened()) {
		fprintf(stderr, "ERROR: capture is NULL \n");
		return -1;
	}
	capture.set(CV_CAP_PROP_FRAME_WIDTH, CAM_WIDTH);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT);
	
	// Initialize matrix objects
	gpu::GpuMat prevframe(CAM_WIDTH, CAM_HEIGHT, CV_8UC3), 
				prevframe_gray(CAM_WIDTH, CAM_HEIGHT, CV_8UC1), 
				currframe(CAM_WIDTH, CAM_HEIGHT, CV_8UC3),
				currframe_gray(CAM_WIDTH, CAM_HEIGHT, CV_8UC1),
				diff_img, thresh_img,
				hist(1,256,CV_32SC1);
	Mat mat_prevframe(CAM_WIDTH, CAM_HEIGHT, CV_8UC3),
		mat_currframe(CAM_WIDTH, CAM_HEIGHT, CV_8UC3),
		mat_diff, mat_thresh, mat_hist;

	//DEBUG!!
	Mat mat_prevframe_gray(CAM_WIDTH, CAM_HEIGHT, CV_8UC1),
	    mat_currframe_gray(CAM_WIDTH, CAM_HEIGHT, CV_8UC1);

	// SVM
	CvSVM svm;
	svm.load(SVM_FILE);

	// Haar
	gpu::CascadeClassifier_GPU eye_cascade, eyepair_cascade;
	eye_cascade.load(HAAR_EYE_FILE);
	eyepair_cascade.load(HAAR_EYEPAIR_FILE);

	// Blob detection parameters
	SimpleBlobDetector::Params params;
	params.minDistBetweenBlobs = 50.0f;
	params.filterByInertia = false;
	params.filterByConvexity = false;
	params.filterByColor = false;
	params.filterByCircularity = true;
	params.filterByArea = true;
	params.minArea = 4.0f;
	params.maxArea = 200.0f;
	params.minCircularity = 0.3f;
	params.maxCircularity = 1.0f;
	params.thresholdStep = 1;
	vector<KeyPoint> keypoints;

	// Histogram-related constants
	const int histThresholdPixels = CAM_WIDTH*CAM_HEIGHT*HIST_THRESHOLD;
	const float range[] = { 0, HIST_BINS } ;
	const float* histRange = { range };
	int bins = HIST_BINS;
	int histCount, threshBin;

	// Display windows
	cvNamedWindow(W_COLOR, CV_WINDOW_NORMAL);
	cvNamedWindow(W_DIFF, CV_WINDOW_NORMAL);
	cvNamedWindow(W_THRESH, CV_WINDOW_NORMAL);

	for(;;) {
		timing(true,"loop");
		// Grab two consecutive frames, convert to grayscale
		capture >> mat_prevframe;
		prevframe.upload(mat_prevframe);
		gpu::cvtColor(prevframe, prevframe_gray, CV_BGR2GRAY);
		
		capture >> mat_currframe;
		currframe.upload(mat_currframe);
		gpu::cvtColor(currframe, currframe_gray, CV_BGR2GRAY);
		
		// Obtain difference image and blur
		gpu::absdiff(currframe_gray, prevframe_gray, diff_img);
		//gpu::blur(diff_img, diff_img, Size(3, 3), Point(-1,-1));

		// Find threshold based on histogram
		gpu::calcHist(diff_img,hist);
		hist.download(mat_hist);
		histCount = 0;
		for (threshBin = HIST_BINS-1; threshBin > 0; threshBin--) {
			histCount += mat_hist.at<uchar>(threshBin);
			if (histCount > histThresholdPixels)
				break;
		}
		gpu::threshold(diff_img,thresh_img,threshBin,255,CV_THRESH_BINARY);

		// Set blob thresholds, then detect blobs
		params.minThreshold = threshBin-1;
		params.maxThreshold = threshBin+1;
		Ptr<FeatureDetector> blob_detector = new SimpleBlobDetector(params);
		blob_detector->create("SimpleBlob");
		diff_img.download(mat_diff); // Blob detect doesn't work on GpuMat objects
		blob_detector->detect(mat_diff, keypoints);
		drawKeypoints(mat_currframe,keypoints,mat_currframe,CV_RGB(0,0,255));

		// Show images
		imshow(W_COLOR,mat_currframe);
		imshow(W_DIFF,mat_diff);
		thresh_img.download(mat_thresh);
		imshow(W_THRESH,mat_thresh);

		// End of loop -- check for inputs
		switch(waitKey(1)) {
		case 27: // ESC
		case 'x': 
			return 0;
		}
		timing(false);
	}
	return 0;
}

void timing(bool start, string what) {
	static struct timeval starttime, endtime;
	static long mtime, seconds, useconds;
	if (start) {
		gettimeofday(&starttime, NULL);
		cout << "timing " << what << endl;
	}
	else {
		gettimeofday(&endtime, NULL);
		seconds  = endtime.tv_sec  - starttime.tv_sec;
		useconds = endtime.tv_usec - starttime.tv_usec;

		mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;

		printf("Elapsed time: %ld milliseconds\n", mtime);
	}
}

/*
Rect *haarEyeClassify(gpu::CascadeClassifier_GPU &cascade, gpu::GpuMat &image) {
	gpu::GpuMat eyes, tmpImage;
	const int scaleFactor = 1.2;
	const int minNeighbors = 3;
	const Size minSize = Size(20,20);

	equalizeHist(image,tmpImage);
	int numDetected = cascade.detectMultiScale(tmpImage, eyes, scaleFactor, minNeighbors, minSize);
	if (numDetected == 0)
		return 0;
	
	Mat tmpMat;
	eyes.colRange(0, numDetected).download(tmpMat);
	Rect *eyeRect = tmpMat.ptr<Rect>();
	return eyeRect[0];
}
*/
