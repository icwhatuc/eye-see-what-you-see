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

#define EXPECTEDFPS	30

#define HISTTHRESHOLD 0.0001
#define COLOUR		255

#define USECSPERSEC		1000000
#define USECSPERMSEC	1000
#define EXPCTDFTM		33333		// @ 20 fps -> 50000 microsecond between each frame

#define flip(x)		((x == '0')?x='1':x='0')
#define isOn(x)		(x == '1') // currently, when off -> bright pupil image

#define SSORIG		'0'
#define SSDIFFIMG	'9'

#define DEBUGON 	0		// boolean to turn on/off debug printf statements

#define TIMGW		75		// width of training images
#define TIMGH		75		// height of training images

#define CAMWIDTH	1920
#define CAMHEIGHT	1080

#define EYECLASS	1
#define NONEYECLASS	-1

#define RED			(CV_RGB(255,0,0))
#define BLUE		(CV_RGB(0,0,255))
#define GREEN		(CV_RGB(0,255,0))

#define HISTSIZE	256

#define centerOfRect(R)	(Point(R.x + R.width/2, R.y + R.height/2))

using namespace cv;
using namespace std;

time_t t, currrunstamp;
struct tm * now, *currrunstamp_tm;

int isInRect(Point2f &pt, Rect &r)
{
	if(pt.x < (r.x + r.width) && pt.x > r.x
		&& pt.y < (r.y + r.height) && pt.y > r.y)
		return 1;

	return 0;
}

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

float predict_eye(CvSVM &svm, Mat &img_mat_orig)
{
	//struct timeval start, end;
	//long mtime, seconds, useconds;
    
    Mat img_mat;
    
	equalizeHist(img_mat_orig,img_mat);
	Mat img_mat_1d(1,img_mat.size().area(),CV_32FC1);
	
	int ii = 0;
	
	for (int i = 0; i<img_mat.rows; i++) {
		for (int j = 0; j < img_mat.cols; j++) {
			img_mat_1d.at<float>(ii++) = img_mat.at<uchar>(i,j);
		}
	}
	
	//gettimeofday(&start, NULL);
	float retval = svm.predict(img_mat_1d);
	//gettimeofday(&end, NULL);
	
	//seconds  = end.tv_sec  - start.tv_sec;
    //useconds = end.tv_usec - start.tv_usec;
    
    //mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
    
	//printf("Elapsed time: %ld milliseconds\n", mtime);
	return retval;
}

#define overlap(r1,r2) ((r1 & r2).size().area() > (TIMGW*TIMGH)/20)

int isKnown(Rect &region, std::vector<Rect> &regionlist)
{
	/* index @ which there is a region similar to parameter in regionlist */
	for(int index = 0; index < regionlist.size(); index++)
	{
		if(overlap(region, regionlist[index]))
			return index;
	}
	return -1;
}

bool predict_eye_haar(Mat &img, Mat &coloredimg, Point &offset) {
	static CascadeClassifier cascade;
	if (cascade.empty() && !cascade.load("cascades/haarcascade_eye.xml")) {
		fprintf(stderr, "ERROR: could not load haar cascade xml for eye\n");
		exit(-1);
	}
    Mat img_mat;
	std::vector<Rect> eyes;
	equalizeHist(img,img_mat);
	cascade.detectMultiScale(img_mat, eyes, 1.2, 3, CV_HAAR_DO_CANNY_PRUNING, Size(20,20) );
	
	for(int i = 0 ; i < eyes.size(); i++)
		rectangle(coloredimg, Point(eyes[i].x + offset.x,eyes[i].y + offset.y), Point(eyes[i].x + eyes[i].width + offset.x, eyes[i].y + eyes[i].height + offset.y), 
			CV_RGB(255,255,51),2);
	
	return eyes.size() > 0;
}

bool predict_eyepair(Mat &img, Mat &coloredimg, Point &offset) {
	static CascadeClassifier cascade;
	if (cascade.empty() && !cascade.load("cascades/haarcascade_eyepair.xml")) {
		fprintf(stderr, "ERROR: could not load haar cascade xml for eye pairs\n");
		exit(-1);
	}
    Mat img_mat;
	std::vector<Rect> eyepairs;
	equalizeHist(img,img_mat);
	cascade.detectMultiScale(img_mat, eyepairs, 1.2, 2, CV_HAAR_DO_CANNY_PRUNING, Size(60,20) );
	
	for(int i = 0 ; i < eyepairs.size(); i++) {
		rectangle(coloredimg, Point(eyepairs[i].x + offset.x,eyepairs[i].y + offset.y), Point(eyepairs[i].x + eyepairs[i].width + offset.x, eyepairs[i].y + eyepairs[i].height + offset.y), 
			CV_RGB(0,0,255),2);
		break;
	}
	return eyepairs.size() == 1;
}

void displayEyePair(Mat &img, Mat &coloredimg, eyepair *ep)
{
	/*
	int x = 0, y = 0, x1 = 0, y1 = 0;
	static int paircounter = 0;
	char filename[256];
	
	if(ep->firsteye.x < ep->secondeye.x)
	{
		x = ep->firsteye.x;
		x1 = ep->secondeye.x + ep->secondeye.width;
	}
	else
	{
		x = ep->secondeye.x;
		x1 = ep->firsteye.x + ep->firsteye.width;
	}
	
	if(ep->firsteye.y < ep->secondeye.y)
	{
		y = ep->firsteye.y;
		y1 = ep->secondeye.y + ep->secondeye.height;
	}
	else
	{
		y = ep->secondeye.y;
		y1 = ep->firsteye.y + ep->firsteye.height;
	}
	
	Mat trainimg = img(Rect(x, y, x1-x, y1-y));
	
	Mat resizedpair(TIMGW, TIMGW*2, CV_8UC1);
	resize(trainimg, resizedpair, resizedpair.size(), 1, 1);
	
	Point offset(x,y);
	if (predict_eyepair(trainimg,coloredimg,offset)) {
		int pxdist = (int)sqrt(pow(ep->firsteye.x-ep->secondeye.x, 2) + pow(ep->firsteye.y - ep->secondeye.y, 2));
		
		char eyepairinfo[100];
		sprintf(eyepairinfo, "@(%d,%d); @(%d,%d). %dpx apart.", 
			ep->firsteye.x, 
			ep->firsteye.y, 
			ep->secondeye.x, 
			ep->secondeye.y, 
			pxdist);
	
		//rectangle(coloredimg, Point(x,y), Point(x1,y1), CV_RGB(0,0,0),2);
		putText(coloredimg, eyepairinfo, Point(x,y-50), FONT_HERSHEY_COMPLEX_SMALL,
			1, RED, 1, CV_AA);
	}
	paircounter++;
*/}

int main(int argc, char *argv[])
{
	/* setup the capture - either file or camera based on command line args */
	CvCapture *capture;
	if(argc == 1)
		capture = cvCaptureFromCAM( CV_CAP_ANY );
	else
		capture = cvCaptureFromFile( argv[1] );
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, CAMWIDTH);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, CAMHEIGHT);
	
	double width = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
	double height = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);

	// report properties
	printf("capture: width = %f, height = %f\n", width, height);

	
	/* time when application is run for naming any files the application writes and saves */
	currrunstamp = time(0);
	currrunstamp_tm = localtime(&currrunstamp);
	
	currrunstamp_tm->tm_year += 1900;
	currrunstamp_tm->tm_mon += 1;
	
	// storing videos
	VideoWriter *record; 
	
	short recordcandidates = 0, gethistbasis = 1;
	char arduino_state = '0', filename[256], num_eyes_str[16];
	int i = 0, j = 0, delay = 100000000, debug = 0, thresh = 100, framecount = 1;
	long stc, etc;
	bool captureFlag = false;
	double min, max, adapt_thresh, opened_adapt_thresh, time_elapsed;
	
	Point minloc, maxloc;	
	IplImage *prevframe, *currframe;
    Mat prevframe_mat, prevframe_gray, currframe_mat, currframe_matcopy, 
		currframe_gray, diff_img, diff_copy, colored_diff_img, element, 
		openedimg, openeddiffimg, overlappedimg, hist, backprojection;
	
	std::ofstream arduino("/dev/ttyUSB0");

	/* svm stuff */
	CvSVM svm;
	svm.load("eye_classify_withweights.svm");
	
	vector<KeyPoint> keypoints;

	if (!capture)
	{
		fprintf(stderr, "ERROR: capture is NULL \n");
		getchar();
		return -1;
	}

	/* Create windows in which the captured images will be presented */
	cvNamedWindow("difference", CV_WINDOW_NORMAL);
	cvNamedWindow("currframe_mat", CV_WINDOW_NORMAL);
	cvNamedWindow("thresholded_diff", CV_WINDOW_NORMAL);
	//cvNamedWindow("backprojection", CV_WINDOW_NORMAL);
	
	/* parameters to detect blobs of pupils */
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
	
	/* Eye tracking */
	std::vector <Rect> knownEyeRegions;
	std::vector <bool> regionsKnown;

	// Show the image captured from the camera in the window and repeat
	currframe = cvQueryFrame(capture);
	prevframe = cvCloneImage(currframe);
	prevframe_mat = prevframe;
	cvtColor(prevframe_mat, prevframe_gray, CV_BGR2GRAY );

	printf("frame: width = %d and height = %d\n", currframe->width, currframe->height);
	cvResizeWindow("currframe_mat", 1920, 1080);

	float range[] = { 0, 256 } ;
	const float* histRange = { range };
	int bins = HISTSIZE;
	int histThreshold = (CAMWIDTH * CAMHEIGHT)*HISTTHRESHOLD;
	int prev_num_eyes = 0;

	//gpu::GpuMat gpu_diff, gpu_thresh;
	while (1)
	{
		std::vector< eyepair> knownEyePairs;
		arduino << arduino_state;
		arduino.flush();
		stc = cvGetTickCount();
		
		currframe = cvQueryFrame(capture);
		
		if(!currframe)
			break;
		
		currframe_mat = currframe;
		currframe_matcopy = currframe_mat.clone();
		timing(true, "gray conv");
		cvtColor(currframe_mat, currframe_gray, CV_BGR2GRAY );
		timing(false);
		absdiff(currframe_gray, prevframe_gray, diff_copy);
		
		diff_img = diff_copy.clone();
		//timing(true,"gpu upload");
		//gpu_diff.upload(diff_img);
		//timing(false);
		
		//GaussianBlur(diff_img, diff_img, Size(9, 9), 8, 8);
		blur(diff_img, diff_img, Size(3, 3), Point(-1,-1));
		
		// Find threshold based on histogram
		calcHist(&diff_img, 1, 0, Mat(), hist, 1, &bins, &histRange, true, false);
		int histCount = 0, bin;
		for (bin = HISTSIZE; bin > 0; bin--) {
			histCount += hist.at<float>(bin);
			if (histCount > histThreshold)
				break;
		}
		timing(true, "thresh");
		threshold(diff_copy,diff_img,bin,255,CV_THRESH_BINARY); // assumes 256 bins
		timing(false);
		/*
		//timing(true,"gpu threshold");
		//gpu::threshold(gpu_diff,gpu_thresh,bin,255,CV_THRESH_BINARY); // assumes 256 bins
		//timing(false);

		imshow("difference", diff_copy);
		imshow("thresholded_diff", diff_img);
		
		// Set blob thresholds
		params.minThreshold = bin-1;
		params.maxThreshold = bin+1;
		params.thresholdStep = 1;
	
		Ptr<FeatureDetector> blob_detector = new 
			SimpleBlobDetector(params);
		blob_detector->create("SimpleBlob");
		blob_detector->detect(diff_img, keypoints);
		drawKeypoints(currframe_mat,keypoints,currframe_mat,BLUE);
		
		
		/* clean regionsKnown */
		//printf("cleaning the regionsKnown information from last frame...\n");
		/*int j;
		for(j = 0; j < regionsKnown.size(); j++)
		{
			regionsKnown[j] = false;
		}
		
		int num_eyes = 0;
		//printf("checking out each keyPoint and verifying last known regions...\n");
		for (j = 0; j < keypoints.size(); j++)
		{
			int x, y;
			Mat candidateimg;
			
			x = (int)(keypoints[j].pt.x - TIMGW/2);
			y = (int)(keypoints[j].pt.y - TIMGH/2);
			if(x < 0 || y < 0 ||
				x + TIMGW >= CAMWIDTH ||
				y + TIMGH >= CAMHEIGHT)
				continue;
			
			Rect candidateRegion(x, y, TIMGW, TIMGH);
			
			if(arduino_state != '0')
				candidateimg = currframe_gray(candidateRegion);
			else
				candidateimg = prevframe_gray(candidateRegion);
			
			Point offset(candidateRegion.x, candidateRegion.y);
			if(predict_eye_haar(candidateimg, currframe_mat, offset) || predict_eye(svm, candidateimg) == EYECLASS)
			{
				num_eyes++;
				int index;
				
				for(index = 0; index < knownEyeRegions.size(); index++)
				{
					if(isInRect(keypoints[j].pt, knownEyeRegions[index]))
					{
						knownEyeRegions[index] = candidateRegion;
						regionsKnown[index] = true;
						break;
					}
				}

				if(index == knownEyeRegions.size())
				{
					knownEyeRegions.push_back(candidateRegion);
					regionsKnown.push_back(true);
				}
				
				circle(currframe_mat, centerOfRect(knownEyeRegions[j]), 5, (CV_RGB(255,0,0)), 3);

				// saving every circle square
				if (recordcandidates)
				{
					sprintf(filename, "candidates/candidate%d_%d_%02d%02d%02d_%d_%d_red.jpg", 
							currrunstamp_tm->tm_mon, 
							currrunstamp_tm->tm_mday,
							currrunstamp_tm->tm_hour,
							currrunstamp_tm->tm_min,
							currrunstamp_tm->tm_sec,
							framecount,
							j);
					imwrite(filename, candidateimg);
				}
			}
		}

		//printf("filling in missed information...\n");
		for(j = 0; j < regionsKnown.size(); j++)
		{
			if(!regionsKnown[j])
			{
				Mat candidateimg;
				if(arduino_state != '0')
					candidateimg = currframe_gray(knownEyeRegions[j]);
				else
					candidateimg = prevframe_gray(knownEyeRegions[j]);
				
				Point offset(knownEyeRegions[j].x, knownEyeRegions[j].y);
				if(predict_eye_haar(candidateimg, currframe_mat, offset) || predict_eye(svm, candidateimg) == EYECLASS)
				{
					circle(currframe_mat, centerOfRect(knownEyeRegions[j]), 5, (CV_RGB(0,255,0)), 3);

					num_eyes++;
					
					// saving every circle square
					if (recordcandidates)
					{
						sprintf(filename, "candidates/candidate%d_%d_%02d%02d%02d_%d_%d_green.jpg", 
								currrunstamp_tm->tm_mon, 
								currrunstamp_tm->tm_mday,
								currrunstamp_tm->tm_hour,
								currrunstamp_tm->tm_min,
								currrunstamp_tm->tm_sec,
								framecount,
								j);
						imwrite(filename, candidateimg);
					}
					
				}
				else {
					knownEyeRegions.erase(knownEyeRegions.begin()+j);
					regionsKnown.erase(regionsKnown.begin()+j);
				}
			}
		}
		
		int numberofeyes = regionsKnown.size();
		
		/* pairing eyes */
		/*for(int i1 = 0; i1 < numberofeyes-1; i1++)
		{
			for(int i2 = i1+1; i2 < numberofeyes; i2++)
			{
				struct _eyepair currpair;
				currpair.firsteye = knownEyeRegions[i1];
				currpair.secondeye = knownEyeRegions[i2];
				knownEyePairs.push_back(currpair);
			}
		}
		
		/* display eyepairs */
		/*for(int i = 0; i < knownEyePairs.size(); i++)
		{
			displayEyePair(currframe_gray, currframe_mat, &(knownEyePairs[i]));
		}
		


		prev_num_eyes = num_eyes;
		sprintf(num_eyes_str,"Eyes: %d",num_eyes);
		putText(currframe_mat, num_eyes_str, Point(30,100), FONT_HERSHEY_COMPLEX_SMALL,
			3, RED, 1, CV_AA);
		imshow("currframe_mat", currframe_mat);
		
		// record video
		if (captureFlag == true)
		{
			if (DEBUGON) printf("Recording frame\n");
			//Mat diff_img_color;
			//cvtColor(diff_img, diff_img_color, CV_GRAY2BGR);
			(*record) << currframe_mat; 
		}
		
		std::swap(prevframe_gray,currframe_gray);

		etc = cvGetTickCount();
		time_elapsed = (etc - stc)/cvGetTickFrequency();
		
		//double fps = USECSPERSEC/time_elapsed;
		//printf("time elapsed %f usecs. wait time = %f msecs\n", time_elapsed, (EXPCTDFTM-time_elapsed)/USECSPERMSEC);
		
		int wait_time = time_elapsed-EXPCTDFTM < 0?(EXPCTDFTM-time_elapsed)/USECSPERMSEC:1;
		if(wait_time == 0)
		{
			wait_time = 1;
		}
		*/
		int keyVal = cvWaitKey(1000) & 255;
		
		/*if(keypoints.size() > 20)
			keyVal = 'n';*/

		if ( (keyVal) == 27 )
		{
		}
		else if ( (keyVal) == 'f' )
		{
			for(int i=0; i<5; i++)
				currframe = cvQueryFrame(capture);
		}
		// DEBUG: temporary stuff
		else if ( (keyVal) == 'n' ) 
		{
			while(int key = cvWaitKey(1000) & 255) {
				if (key == 'n')
					break;
			}
		}
		else if ( (keyVal) == 't' )
		{
			flip(arduino_state);	
		}
		else if ( (keyVal) == 'h')
		{
			gethistbasis = 1;
		}
		else if( (keyVal) == 32) // spacebar
		{
			if (captureFlag==false)
			{
				captureFlag=true;
				if (DEBUGON) printf("video is now on\n");
				record = new VideoWriter("icwhatucvideo.avi", CV_FOURCC('M','J','P','G'), 20, diff_img.size(), true);
				if( !record->isOpened() ) {
					printf("VideoWriter failed to open!\n");
				}
				
			}
			
			else if (captureFlag==true)
			{
				captureFlag=false;
				if (DEBUGON) printf("video is now off\n");
			}
		}
		
		else if ( (keyVal) == 'x' )
		{
			break; // break out of loop to stop program
		}
		
		else if ( (keyVal) == 'c')
		{
			if (recordcandidates == 0)
				recordcandidates = 1;
			else
				recordcandidates = 0;
		}
		
		/* image snapshot options */
		
		else if ( (keyVal) >= SSORIG && (keyVal) <= SSDIFFIMG)
		{
			char *imgtype;
			Mat *ss;
			
			t = time(0);   // get time now
			now = localtime( & t );
			
			switch(keyVal)
			{
				case SSORIG: imgtype = "coloredorig"; 
					ss = &currframe_mat; break;
				case SSDIFFIMG: imgtype = "diffimg"; 
					ss = &diff_copy; break;
			}
						
			sprintf(filename, "./snapshots/%s_%d-%d-%d_%d:%d:%d.jpg", imgtype, 
				(now->tm_year + 1900), (now->tm_mon + 1), now->tm_mday, 
				now->tm_hour, now->tm_min, now->tm_sec);
			imwrite(filename, *ss);
		}
		
		framecount++;
		flip(arduino_state);	
		
	}

	// Release the capture device housekeeping
	cvReleaseCapture( &capture );
	cvDestroyWindow( "mywindow" );
	return 0;
}


