#include <cv.h> 
#include <highgui.h> 
#include <cstdio>  
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <ctime>
#include <ml.h>

#define EXPECTEDFPS	30

#define THRESHOLD	245
#define COLOUR		255

#define USECSPERSEC		1000000
#define USECSPERMSEC	1000
#define EXPCTDFTM		33333		// @ 20 fps -> 50000 microsecond between each frame

#define SHOWPREVCURR	0

#define HISTSIZE	26			// (256/10 + 1)
#define SHOWHIST	0

#define USEOPENOP	0
#define MORPHSIZE	4
#define OPENOP		2

#define flip(x)		((x == '0')?x='1':x='0')
#define isOn(x)		(x == '1') // currently, when off -> bright pupil image

#define THRESHREFRESHRATE	5		// seconds
#define ADPTTHRSCUT		(0.75)

#define SSORIG		'0'
#define SSDIFFIMG	'9'

#define RESIZEFCTR	6

#define DEBUGON 	0		// boolean to turn on/off debug printf statements

#define TIMGW		75		// width of training images
#define TIMGH		75		// height of training images

#define CAMWIDTH	1920
#define CAMHEIGHT	1080

#define EYECLASS	1
#define NONEYECLASS	-1

using namespace cv;

void calcplothist(Mat &img)
{
	float range[] = { 0, 256 } ;
	const float* histRange = { range };
	int bins = HISTSIZE;
	
	Mat hist;
	calcHist( &img, 1, 0, Mat(), hist, 1, &bins, &histRange, true, false );
	
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double) hist_w/bins );

	Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
	
	normalize(hist, hist, 0, hist.rows*10, NORM_MINMAX, -1, Mat());
	
	for( int i = 1; i < bins; i++ )
	{
	line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
		Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i-1)) ),
		Scalar( 255, 0, 0), 2, 8, 0  );
	}
	
	imshow("diffimgHist", histImage );
}

int isInRect(Point2f &pt, Rect &r)
{
	if(pt.x < (r.x + r.width) && pt.x > r.x
		&& pt.y < (r.y + r.height) && pt.y > r.y)
		return 1;

	return 0;
}

float predict_eye(CvSVM &svm, Mat &img_mat) {
	equalizeHist(img_mat,img_mat);
	Mat img_mat_1d(1,img_mat.size().area(),CV_32FC1);
	
	int ii = 0;
	
	for (int i = 0; i<img_mat.rows; i++) {
		for (int j = 0; j < img_mat.cols; j++) {
			img_mat_1d.at<float>(ii++) = img_mat.at<uchar>(i,j);
		}
	}
	
	return svm.predict(img_mat_1d);
}

int main()
{
	short recordcandidates = 0;
	char arduino_state = '0', filename[256];
	int i = 0, j = 0, delay = 100000000, debug = 0, thresh = 100, framecount = 1;
	long stc, etc;
	double time_elapsed;
	
	if (DEBUGON) printf("point%d\n", debug++);//0
	CvCapture *capture = cvCaptureFromCAM( CV_CAP_ANY );
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, CAMWIDTH);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, CAMHEIGHT);
	// the fps currently doesn't work
	//cvSetCaptureProperty(capture, CV_CAP_PROP_FPS, 30);
	if (DEBUGON) printf("point%d\n", debug++);//1	
	IplImage *prevframe, *currframe;
	
	bool captureFlag = false;
	
	Mat prevframe_mat, prevframe_gray, currframe_mat, 
		currframe_gray, diff_img, colored_diff_img, element, 
		openedimg, openeddiffimg, overlappedimg;
	std::ofstream arduino("/dev/ttyUSB0");

	VideoWriter *record; 

	double fps = cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);
	double width = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
	double height = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);

	printf("fps = %f, height = %f, width = %f\n", fps, height, width);

	if (!capture)
	{
		fprintf(stderr, "ERROR: capture is NULL \n");
		getchar();
		return -1;
	}

	//cvNamedWindow( "coloreddiff", CV_WINDOW_NORMAL );
	//cvNamedWindow( "prevframe_mat", CV_WINDOW_NORMAL );
	cvNamedWindow( "currframe_mat", CV_WINDOW_NORMAL );

	// Show the image captured from the camera in the window and repeat
	currframe = cvQueryFrame(capture);
	prevframe = cvCloneImage(currframe);
	prevframe_mat = prevframe;
	cvtColor(prevframe_mat, prevframe_gray, CV_BGR2GRAY );

	printf("width = %d and height = %d\n", currframe->width, currframe->height);
	cvResizeWindow("currframe_mat", 1920, 1080);

	while (1)
	{
		arduino << arduino_state;
		arduino.flush();
		stc = cvGetTickCount();
		
		currframe = cvQueryFrame(capture);
		currframe_mat = currframe;
		
		// record video
		if (captureFlag == true)
		{
			if (DEBUGON) printf("Recording frame\n");
			//Mat diff_img_color;
			//cvtColor(diff_img, diff_img_color, CV_GRAY2BGR);
			(*record) << currframe_mat; 
		}
		
		imshow("currframe_mat",currframe_mat);
		
		etc = cvGetTickCount();
		time_elapsed = (etc - stc)/cvGetTickFrequency();
		
		//double fps = USECSPERSEC/time_elapsed;
		//printf("time elapsed %f usecs. wait time = %f msecs\n", time_elapsed, (EXPCTDFTM-time_elapsed)/USECSPERMSEC);
		
		int wait_time = time_elapsed-EXPCTDFTM < 0?(EXPCTDFTM-time_elapsed)/USECSPERMSEC:1;
		if(wait_time == 0)
		{
			wait_time = 1;
		}

		int keyVal = cvWaitKey(wait_time) & 255;

		if ((keyVal) == 32) // spacebar
		{
			if (captureFlag==false)
			{
				captureFlag=true;
				if (DEBUGON) printf("video is now on\n");
				record = new VideoWriter("control_will2.avi", CV_FOURCC('M','J','P','G'), 30, currframe_mat.size(), true);
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
		
		
		framecount++;
		flip(arduino_state);	
		
	}
	// Release the capture device housekeeping
	cvReleaseCapture( &capture );
	cvDestroyWindow( "mywindow" );
	return 0;
}

