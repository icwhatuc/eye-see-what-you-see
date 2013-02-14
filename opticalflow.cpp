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

#define CAMWIDTH	1920
#define CAMHEIGHT	1080

using namespace cv;
using namespace std;

void timing(bool start, string what="") {
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

void mouseEvent(int event, int x, int y, int flags, void* ptr) {
	if(event == CV_EVENT_LBUTTONDOWN)
		((std::vector<Point2f>*)ptr)->push_back(Point2f(x,y));
}

int main(int argc, const char *argv[])
{
	VideoCapture capture;
	int keyVal;

	if(argc == 1)
		capture.open(0);
	else
		capture.open(argv[1]);

	if (!capture.isOpened())
	{
		fprintf(stderr, "ERROR: capture is NULL \n");
		return -1;
	}
	capture.set(CV_CAP_PROP_FRAME_WIDTH, CAMWIDTH);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, CAMHEIGHT);

	Mat prevframe_mat(CAMWIDTH, CAMHEIGHT, CV_8UC3), 
		prevframe_gray(CAMWIDTH, CAMHEIGHT, CV_8UC1), 
		currframe_mat(CAMWIDTH, CAMHEIGHT, CV_8UC3),
		currframe_gray(CAMWIDTH, CAMHEIGHT, CV_8UC1);

	namedWindow("currframe", CV_WINDOW_NORMAL);

	// Optical flow stuff
	std::vector<Point2f> features_prev, features_next;
	std::vector<uchar> status;
	std::vector<float> err;

	capture >> currframe_mat;
	cvtColor(currframe_mat.clone(), currframe_gray, CV_BGR2GRAY);
	/*
	goodFeaturesToTrack(
		currframe_gray, // the image 
		features_next,   // the output detected features
		100,  // the maximum number of features 
		0.1,     // quality level
		20     // min distance between two features
	);
	*/
	features_next.push_back(Point2f(0,0));
	
	setMouseCallback("currframe",mouseEvent,&features_next);

	for (;;) {
	    features_prev = features_next;

		//capture >> prevframe_mat;
		//cvtColor(prevframe_mat.clone(), prevframe_gray, CV_BGR2GRAY);
		prevframe_gray = currframe_gray.clone();

		capture >> currframe_mat;
		cvtColor(currframe_mat.clone(), currframe_gray, CV_BGR2GRAY);

		calcOpticalFlowPyrLK(
			prevframe_gray, currframe_gray, // 2 consecutive images
			features_prev, // input point positions in first im
			features_next, // output point positions in the 2nd
			status,    // tracking success
			err      // tracking error
		);

		for (int i=0; i<features_next.size(); i++) {
			circle(currframe_mat, features_next[i], 5, (CV_RGB(255,0,0)), 3);
		}
		imshow("currframe", currframe_mat);

		keyVal = cvWaitKey(1) & 255;
		if (keyVal == 'x')
			break; // break out of loop to stop program
		else if (keyVal == ' ') {
			features_next.clear();	
			features_next.push_back(Point2f(-100,-100));
		}
	}

	return 0;
}
