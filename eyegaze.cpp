#include <cv.h> 
#include <highgui.h> 
#include <cstdio>  
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
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

#define MAX_BLOBS 20

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
Rect haarEyePairClassify(gpu::CascadeClassifier_GPU &cascade, gpu::GpuMat &image);
Rect getEyePairRect(KeyPoint &kp1, KeyPoint &kp2);
gpu::GpuMat getEyePairImage(Mat &mat_image, Rect &r);
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
	gpu::GpuMat frame1, frame1_gray, 
	            frame2, frame2_gray,
	            frame2_gray_prev,
	            diff_img, thresh_img,
	            hist;
	Mat mat_frame1, mat_frame1_gray,
	    mat_frame2, mat_frame2_gray,
	    mat_frame2_gray_prev,
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
	vector<KeyPoint> keypoints, prev_eyes, curr_eyes;

	// Histogram-related constants
	const int histThresholdPixels = CAM_WIDTH*CAM_HEIGHT*HIST_THRESHOLD;
	const float range[] = { 0, HIST_BINS } ;
	const float* histRange = { range };
	int bins = HIST_BINS;
	int hist_count, thresh_bin;

	// Optical flow stuff
	std::vector<Point2f> optflow_prev, optflow_curr;
	std::vector<uchar> optflow_status;
	std::vector<float> optflow_err;

	// Display windows
	cvNamedWindow(W_COLOR, CV_WINDOW_NORMAL);
	cvNamedWindow(W_DIFF, CV_WINDOW_NORMAL);
	cvNamedWindow(W_THRESH, CV_WINDOW_NORMAL);

	capture >> mat_frame2;
	frame2.upload(mat_frame2);
	gpu::cvtColor(frame2, frame2_gray_prev, CV_BGR2GRAY);
	for(;;) {
		// Grab two consecutive frames, convert to grayscale
		capture >> mat_frame1;
		frame1.upload(mat_frame1);
		gpu::cvtColor(frame1, frame1_gray, CV_BGR2GRAY);
		
		capture >> mat_frame2;
		frame2.upload(mat_frame2);
		gpu::cvtColor(frame2, frame2_gray, CV_BGR2GRAY);
		frame2_gray.download(mat_frame2_gray);

		// Obtain difference image and blur
		gpu::absdiff(frame2_gray, frame1_gray, diff_img);
		diff_img.download(mat_diff);
		blur(mat_diff, mat_diff, Size(3, 3), Point(-1,-1));
		diff_img.upload(mat_diff);
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

		// If too many blobs, skip the processing
		if (keypoints.size() > MAX_BLOBS)
			keypoints.clear();
		else
			drawKeypoints(mat_frame2,keypoints,mat_frame2,CV_RGB(0,0,255));

		curr_eyes.clear();
		optflow_curr.clear();
		optflow_prev.clear();

		// Update positions of previous blobs with optical flow
		if (prev_eyes.size() > 0) {
			vector<KeyPoint>::iterator it;
			for (it = prev_eyes.begin(); it != prev_eyes.end(); it++) 
				optflow_prev.push_back(it->pt);

			calcOpticalFlowPyrLK(
				mat_frame2_gray_prev, mat_frame2_gray, 
				optflow_prev, optflow_curr,
				optflow_status, optflow_err
			);
		}

		int keypoints_size_orig = keypoints.size();
		for (int curr_blob=0; curr_blob<keypoints.size(); curr_blob++) {
			// Check if current keypoint overlaps with previously known ones
			vector<Point2f>::iterator it;
			for (it = optflow_curr.begin(); it != optflow_curr.end();) {
				// If no overlap, add to current frame's keypoints
				if (pointsOverlap(*it, keypoints[curr_blob].pt))
					it = optflow_curr.erase(it);
				else 
					it++;
			}
		}
		for (vector<Point2f>::iterator it = optflow_curr.begin(); 
			it != optflow_curr.end(); it++) 
		{
			KeyPoint k(*it,0);
			keypoints.push_back(k);
		}

		// Iterate through keypoints and check using SVM and Haar cascades
		bool svm_detected, haar_detected;
		for (int curr_blob=0; curr_blob<keypoints.size(); curr_blob++) {
			// Top left corner of square region
			int x = keypoints[curr_blob].pt.x - SVM_IMG_SIZE/2;
			int y = keypoints[curr_blob].pt.y - SVM_IMG_SIZE/2;
			if (x < 0 || y < 0 || 
				x + SVM_IMG_SIZE >= CAM_WIDTH ||
				y + SVM_IMG_SIZE >= CAM_HEIGHT)
				continue;

			Rect candidate_region(x, y, SVM_IMG_SIZE, SVM_IMG_SIZE);
			Mat mat_candidate_img = mat_frame2_gray(candidate_region);
			gpu::GpuMat candidate_img(mat_candidate_img);

			// SVM
			Rect eye_rect = haarEyeClassify(eye_cascade,candidate_img);
			bool haar_detected = eye_rect.area() > 1;
			bool svm_detected = svmEyeClassify(svm,mat_candidate_img);

			if (svm_detected) {
				// Draw circle for SVM
				circle(
					mat_frame2, 
					Point(x+SVM_IMG_SIZE/2, y+SVM_IMG_SIZE/2), 
					5, 
					(curr_blob>=keypoints_size_orig) ? CV_RGB(0,255,0) : CV_RGB(255,0,0), 
					3
				);
			}

			// Haar
			if (haar_detected) {
				// Draw rectangle for Haar
				rectangle(
					mat_frame2,
					Point(x+eye_rect.x, y+eye_rect.y),
					Point(x+eye_rect.width+eye_rect.x, y+eye_rect.height+eye_rect.y),
					CV_RGB(255,255,51), 2
				);
			}

			if (svm_detected || haar_detected)
				curr_eyes.push_back(keypoints[curr_blob]);
		}

		// Pair eyes
		Rect eyepair_rect_in, eyepair_rect_out;
		for (int eye1 = 0; eye1 < curr_eyes.size(); eye1++) {
			for (int eye2 = 0; eye2 < curr_eyes.size(); eye2++) {
				if (eye1 == eye2)
					continue;
				eyepair_rect_in = getEyePairRect(curr_eyes[eye1], curr_eyes[eye2]);
				gpu::GpuMat eyepair = getEyePairImage(mat_frame2_gray,eyepair_rect_in);
				/*
				int x = eyepair_rect_in.x;
				int y = eyepair_rect_in.y;
				eyepair_rect_out = haarEyePairClassify(eyepair_cascade, eyepair);
				if (eyepair_rect_out.area() > 1) {
					rectangle(
						mat_frame2,
						Point(x+eyepair_rect_out.x, y+eyepair_rect_out.y),
						Point(x+eyepair_rect_out.width+eyepair_rect_out.x, y+eyepair_rect_out.height+eyepair_rect_out.y),
						CV_RGB(0,0,255), 2
					);
				}
				*/
				int x1 = eyepair_rect_in.x+SVM_IMG_SIZE/2;
				int y1 = eyepair_rect_in.y;
				int x2 = x1 + eyepair_rect_in.width - SVM_IMG_SIZE;
				int y2 = y1 + eyepair_rect_in.height;
				int dx = (x2-x1)/4;
				eyepair_rect_out = haarEyePairClassify(eyepair_cascade, eyepair);
				/*
				if (eyepair_rect_out.area() > 1) {
					rectangle(
						mat_frame2,
						Point(eyepair_rect_in.x, eyepair_rect_in.y),
						Point(eyepair_rect_in.width+eyepair_rect_in.x, eyepair_rect_in.height+eyepair_rect_in.y),
						CV_RGB(0,0,255), 2
					);
				}
				*/
				if (eyepair_rect_out.area() > 1) {
					rectangle(
						mat_frame2,
						Point(x1-dx, y1),
						Point(x2+dx, y2),
						CV_RGB(0,0,255), 2
					);
				}
			}
		}


		//swap(frame2_gray, frame2_gray_prev);
		swap(curr_eyes, prev_eyes);
		mat_frame2_gray_prev = mat_frame2_gray.clone();
		//prev_eyes = curr_eyes;

		// Show images
		imshow(W_COLOR,mat_frame2);
		imshow(W_DIFF,mat_diff);
		thresh_img.download(mat_thresh);
		imshow(W_THRESH,mat_thresh);

		// End of loop -- check for inputs
		switch(waitKey(1)) {
		case 27: // ESC
		case 'x': 
			return 0;
		}
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

Rect getEyePairRect(KeyPoint &kp1, KeyPoint &kp2) {
	Rect r(kp1.pt, kp2.pt);
	r.x -= SVM_IMG_SIZE/2;
	r.y -= SVM_IMG_SIZE/2;
	r.height += SVM_IMG_SIZE;
	r.width += SVM_IMG_SIZE;
	return r;
}

gpu::GpuMat getEyePairImage(Mat &mat_image, Rect &r) {
	Mat eye_pair = mat_image(r);
	Mat resized_pair(SVM_IMG_SIZE, SVM_IMG_SIZE*2, CV_8UC1);
	resize(eye_pair, resized_pair, resized_pair.size(), 1, 1);

	//DEBUG
	cvNamedWindow("test", CV_WINDOW_NORMAL);
	imshow("test",resized_pair);
	return gpu::GpuMat(resized_pair);
}

Rect haarEyePairClassify(gpu::CascadeClassifier_GPU &cascade, gpu::GpuMat &image) {
	gpu::GpuMat eyePairs, eqImage;
	const float scaleFactor = 1.2;
	const int minNeighbors = 3;
	const Size minSize = Size(60,20);

	gpu::equalizeHist(image, eqImage);
	eqImage.convertTo(eqImage,CV_8U);
	int numDetected = cascade.detectMultiScale(eqImage, eyePairs, scaleFactor, minNeighbors, minSize);

	// If no eyepairs detected, return rectangle with area 1
	if (numDetected == 0)
		return Rect(0,0,1,1);
	
	Mat tmpMat;
	eyePairs.colRange(0, numDetected).download(tmpMat);
	Rect *eyePairRect = tmpMat.ptr<Rect>();
	return eyePairRect[0];
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

