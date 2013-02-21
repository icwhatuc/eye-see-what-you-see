#include <cv.h> 
#include <highgui.h> 
#include <cstdio>  
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
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
Rect haarEyeClassify(gpu::CascadeClassifier_GPU &cascade, gpu::GpuMat &image);
bool svmEyeClassify(CvSVM &svm, Mat &image);
bool pointsOverlap(Point2f &p1, Point2f &p2);

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
		mat_prevframe_gray(CAM_WIDTH, CAM_HEIGHT, CV_8UC1),
		mat_currframe(CAM_WIDTH, CAM_HEIGHT, CV_8UC3),
		mat_currframe_gray(CAM_WIDTH, CAM_HEIGHT, CV_8UC1),
		mat_diff, mat_thresh, mat_hist;

	// SVM
	CvSVM svm;
	svm.load(SVM_FILE);

	// Haar
	gpu::CascadeClassifier_GPU eye_cascade, eyepair_cascade;
	if (!eye_cascade.load(HAAR_EYE_FILE)) {
		fprintf(stderr, "ERROR: could not load haar cascade xml for eyes\n");
		return -1;
	}
	if (!eyepair_cascade.load(HAAR_EYEPAIR_FILE)) {
		fprintf(stderr, "ERROR: could not load haar cascade xml for eye pairs\n");
		return -1;
	}

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
	list<KeyPoint> prev_blobs, curr_blobs;

	// Histogram-related constants
	const int histThresholdPixels = CAM_WIDTH*CAM_HEIGHT*HIST_THRESHOLD;
	const float range[] = { 0, HIST_BINS } ;
	const float* histRange = { range };
	int bins = HIST_BINS;
	int hist_count, thresh_bin;

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
		currframe_gray.download(mat_currframe_gray);
		
		// Obtain difference image and blur
		gpu::absdiff(currframe_gray, prevframe_gray, diff_img);
		//gpu::blur(diff_img, diff_img, Size(3, 3), Point(-1,-1));

		// Find threshold based on histogram
		gpu::calcHist(diff_img,hist);
		hist.download(mat_hist);
		hist_count = 0;
		for (thresh_bin = HIST_BINS-1; thresh_bin > 0; thresh_bin--) {
			hist_count += mat_hist.at<uchar>(thresh_bin);
			if (hist_count > histThresholdPixels)
				break;
		}
		gpu::threshold(diff_img,thresh_img,thresh_bin,255,CV_THRESH_BINARY);

		// Set blob thresholds, then detect blobs
		params.minThreshold = thresh_bin-1;
		params.maxThreshold = thresh_bin+1;
		Ptr<FeatureDetector> blob_detector = new SimpleBlobDetector(params);
		blob_detector->create("SimpleBlob");
		diff_img.download(mat_diff); // Blob detect doesn't work on GpuMat objects
		blob_detector->detect(mat_diff, keypoints);
		drawKeypoints(mat_currframe,keypoints,mat_currframe,CV_RGB(0,0,255));

		// Iterate through keypoints and check using SVM and Haar cascades
		bool svm_detected, haar_detected;
		swap(curr_blobs, prev_blobs);
		curr_blobs.clear();
		for (int curr_blob=0; curr_blob<keypoints.size(); curr_blob++) {
			// Check if current keypoint overlaps with previously known ones
			for (list<KeyPoint>::iterator it = prev_blobs.begin();
			     it != prev_blobs.end(); ) 
			{
				if (pointsOverlap(it->pt, keypoints[curr_blob].pt))
					it = prev_blobs.erase(it);
				else
					it++;
			}

			// Top left corner of square region
			int x = keypoints[curr_blob].pt.x - SVM_IMG_SIZE/2;
			int y = keypoints[curr_blob].pt.y - SVM_IMG_SIZE/2;
			if (x < 0 || y < 0 || 
				x + SVM_IMG_SIZE >= CAM_WIDTH ||
				y + SVM_IMG_SIZE >= CAM_HEIGHT)
				continue;

			Rect candidate_region(x, y, SVM_IMG_SIZE, SVM_IMG_SIZE);
			Mat mat_candidate_img = mat_currframe_gray(candidate_region);
			gpu::GpuMat candidate_img(mat_candidate_img);

			// SVM
			Rect eye_rect = haarEyeClassify(eye_cascade,candidate_img);
			bool haar_detected = eye_rect.area() > 1;
			bool svm_detected = svmEyeClassify(svm,mat_candidate_img);

			if (svm_detected) {
				// Draw circle for SVM
				circle(
					mat_currframe, 
					Point(x+SVM_IMG_SIZE/2, y+SVM_IMG_SIZE/2), 
					5, CV_RGB(255,0,0), 3
				);
			}

			// Haar
			if (haar_detected) {
				// Draw rectangle for Haar
				rectangle(
					mat_currframe,
					Point(x+eye_rect.x, y+eye_rect.y),
					Point(x+eye_rect.width+eye_rect.x, y+eye_rect.height+eye_rect.y),
					CV_RGB(255,255,51), 2
				);
			}

			if (svm_detected || haar_detected)
				curr_blobs.push_back(keypoints[curr_blob]);
		}

		// Use optical flow to track blobs from previous frame
		for (list<KeyPoint>::iterator it = prev_blobs.begin();
			 it != prev_blobs.end(); it++) 
		{
			int x = it->pt.x;
			int y = it->pt.y;
			circle(
				mat_currframe, 
				Point(x, y), 
				5, CV_RGB(0,255,0), 3
			);
		}

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

bool pointsOverlap(Point2f &p1, Point2f &p2) {
	int x = p1.x - SVM_IMG_SIZE/2;
	int y = p1.y - SVM_IMG_SIZE/2;
	return (p2.x < (x+SVM_IMG_SIZE) && p2.x > x
		&& p2.y < (y+SVM_IMG_SIZE) && p2.y > y);
}

void timing(bool start, string what) {
	return; //DEBUG
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

Rect haarEyeClassify(gpu::CascadeClassifier_GPU &cascade, gpu::GpuMat &image) {
	gpu::GpuMat eyes, eqImage;
	const float scaleFactor = 1.2;
	const int minNeighbors = 3;
	const Size minSize = Size(20,20);

	gpu::equalizeHist(image, eqImage);
	eqImage.convertTo(eqImage,CV_8U);
	int numDetected = cascade.detectMultiScale(eqImage, eyes, scaleFactor, minNeighbors, minSize);

	// If no eyes detected, return rectangle with area 1
	if (numDetected == 0)
		return Rect(0,0,1,1);
	
	Mat tmpMat;
	eyes.colRange(0, numDetected).download(tmpMat);
	Rect *eyeRect = tmpMat.ptr<Rect>();
	return eyeRect[0];
}

bool svmEyeClassify(CvSVM &svm, Mat &image) {
	Mat eqImage;
    
	equalizeHist(image,eqImage);
	Mat imageMat1D(1,eqImage.size().area(),CV_32FC1);
	
	int index = 0;
	for (int r = 0; r<eqImage.rows; r++) {
		for (int c = 0; c < eqImage.cols; c++) {
			imageMat1D.at<float>(index++) = eqImage.at<uchar>(r,c);
		}
	}
	
	float retval = svm.predict(imageMat1D); // Non-eye: -1, Eye: +1
	return retval>0;
}

