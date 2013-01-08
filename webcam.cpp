#include <cv.h> 
#include <highgui.h> 
#include <cstdio>  
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <ctime>

#define EXPECTEDFPS	30

#define THRESHOLD	250
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

#define DEBUGON 1 // boolean to turn on/off debug printf statements

#define TIMGW		75		// width of training images
#define TIMGH		75		// height of training images

#define CAMWIDTH	1920
#define CAMHEIGHT	1080

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

int main()
{
	short recordcandidates = 0;
	char arduino_state = '0', filename[256];
	int i = 0, j = 0, delay = 100000000, debug = 0, thresh = 100, framecount = 1;
	long stc, etc;
	
	double min, max, adapt_thresh, opened_adapt_thresh, thresh_cut = ADPTTHRSCUT, time_elapsed;
	Point minloc, maxloc;	
	
	time_t t, currrunstamp = time(0);
	struct tm * now, *currrunstamp_tm = localtime(&currrunstamp);
	
	currrunstamp_tm->tm_year += 1900;
	currrunstamp_tm->tm_mon += 1;
	
	if (DEBUGON) printf("point%d\n", debug++);//0
	CvCapture *capture = cvCaptureFromCAM( CV_CAP_ANY );
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, CAMWIDTH);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, CAMHEIGHT);
	// the fps currently doesn't work
	//cvSetCaptureProperty(capture, CV_CAP_PROP_FPS, 30);
	if (DEBUGON) printf("point%d\n", debug++);//1	
	IplImage *prevframe, *currframe;
	
	vector<Rect> faces;
	CascadeClassifier cascade;
	const static Scalar colors[] =  { 
		CV_RGB(0,0,255),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)
	};
    cascade.load("cascades/haarcascade_frontalface_default.xml");
    vector<KeyPoint> keyPoints;
    
	// parameters to detect blobs of pupils
	SimpleBlobDetector::Params params;
	params.minDistBetweenBlobs = 75.0f;
	params.filterByInertia = false;
	params.filterByConvexity = false;
	params.filterByColor = false;
	params.filterByCircularity = true;
	params.filterByArea = true;
	params.minArea = 50.0f;
	params.maxArea = 200.0f;
	params.minCircularity = 0.5f;
	params.maxCircularity = 1.0f;
	
	Ptr<FeatureDetector> blob_detector = new 
		SimpleBlobDetector(params);
	blob_detector->create("SimpleBlob");
	vector<KeyPoint> keypoints;

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

	if (DEBUGON) printf("point%d\n", debug++);//2

	/* Create windows in which the captured images will be presented */
	cvNamedWindow( "difference", CV_WINDOW_NORMAL );
	//cvNamedWindow( "coloreddiff", CV_WINDOW_NORMAL );
	//cvNamedWindow( "prevframe_mat", CV_WINDOW_NORMAL );
	cvNamedWindow( "currframe_mat", CV_WINDOW_NORMAL );
	
	if(SHOWPREVCURR)
	{
		cvNamedWindow( "prevframe_gray", CV_WINDOW_NORMAL );
		cvNamedWindow( "currframe_gray", CV_WINDOW_NORMAL );
	}
	if(SHOWHIST)
	{
		cvNamedWindow("diffimgHist", CV_WINDOW_NORMAL);
	}
	if(USEOPENOP)
	{
		cvNamedWindow("morphopendiff", CV_WINDOW_NORMAL);
		cvNamedWindow("overlappedimg", CV_WINDOW_NORMAL);
	}
	printf("point%d\n", debug++);//3

	// Show the image captured from the camera in the window and repeat
	currframe = cvQueryFrame(capture);
	prevframe = cvCloneImage(currframe);
	prevframe_mat = prevframe;
	cvtColor(prevframe_mat, prevframe_gray, CV_BGR2GRAY );

	printf("width = %d and height = %d\n", currframe->width, currframe->height);
	cvResizeWindow("currframe_mat", 1920, 1080);

	printf("point%d\n", debug++);//4
	while (1)
	{
		arduino << arduino_state;
		arduino.flush();
		stc = cvGetTickCount();
		
		currframe = cvQueryFrame(capture);
		currframe_mat = currframe;
		
		cvtColor(currframe_mat, currframe_gray, CV_BGR2GRAY );
		
		//if(i++%2) {
			if(SHOWPREVCURR)
			{
				imshow("prevframe_gray", prevframe_gray);
				imshow("currframe_gray", currframe_gray);
			}
			absdiff(prevframe_gray, currframe_gray, diff_img);
			
			/*
			 * MIHIR - do we need this?
			 *
			RNG rng(12345);
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			threshold(diff_img, diff_img, 50, COLOUR, CV_THRESH_BINARY);
			findContours( diff_img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
			//Mat drawing = Mat::zeros( diff_img.size(), CV_8UC3 );
			for( int i = 0; i< contours.size(); i++ )
			{
				Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
				drawContours( diff_img, contours, i, color, 2, 8, hierarchy, 0, Point() );
			}
			*/

			if(USEOPENOP && i++%(EXPECTEDFPS*THRESHREFRESHRATE) == 0) // recalculate threshold based on refresh rate
			{
				minMaxLoc(diff_img, &min, &max, &minloc, &maxloc);
				adapt_thresh = (max - min)*thresh_cut + min;
				//std::cout << "adjusted first thresh\n";
			}
			
			Mat resized_frame;
			Size s( diff_img.size().width / RESIZEFCTR, diff_img.size().height / RESIZEFCTR );
			resize( currframe_gray, resized_frame, s, 0, 0, CV_INTER_AREA );

			
			GaussianBlur( diff_img, diff_img, Size(9, 9), 16, 16 );
			equalizeHist(diff_img, diff_img);
			threshold(diff_img, diff_img,THRESHOLD,255,CV_THRESH_BINARY);

/*
			int dilation_size = 3;
			Mat element = getStructuringElement( MORPH_ELLIPSE,
				Size( 2*dilation_size + 1, 2*dilation_size+1 ),
				Point( dilation_size, dilation_size ) );
			erode(diff_img, diff_img, element);
*/
			blob_detector->detect(diff_img, keypoints);
			
			drawKeypoints(currframe_mat,keypoints,currframe_mat,colors[0]);
			//equalizeHist( currframe_gray, currframe_gray );
			/*cascade.detectMultiScale( resized_frame, faces,
		    1.1, 2, 0
		    //|CV_HAAR_FIND_BIGGEST_OBJECT
		    //|CV_HAAR_DO_ROUGH_SEARCH
		    |CV_HAAR_SCALE_IMAGE
		    ,
		    Size(30, 30) );
		    
		    if(recordcandidates)
		    {	    
				sprintf(filename, "candidates/candidate%d_%d_%02d%02d%02d_%d.jpg", 
									currrunstamp_tm->tm_mon, 
									currrunstamp_tm->tm_mday,
									currrunstamp_tm->tm_hour,
									currrunstamp_tm->tm_min,
									currrunstamp_tm->tm_sec,
									framecount);
				imwrite(filename, currframe_mat);
		    }
		    
			if(keypoints.size() > 0)
		    {
				for( i = 0; i < faces.size(); i++ )
				{
					Scalar color = colors[i%8];
					// cut off bottom half of the rectangle
					// faces[i].height/=2;
					
					faces[i].x*=RESIZEFCTR;
					faces[i].y*=RESIZEFCTR;
					faces[i].width*=RESIZEFCTR;
					faces[i].height*=RESIZEFCTR;
					
					faces[i].y += (height/50);
					//faces[i].height -= (height/10);
					faces[i].height = faces[i].height*1/2;
					
					int oldwidth = faces[i].width;
					
					faces[i].width = faces[i].width*5/6;
					faces[i].x += ((oldwidth - faces[i].width)/2);
					
					rectangle(currframe_mat, faces[i], color);
					//Mat roi = diff_img(faces[i]);
					
					/*Point maxLoc;
					double maxval;
				
					minMaxLoc(roi, 0, &maxval, 0, &maxLoc);
				
					if(maxval < 50)
						continue;
					maxLoc.x += faces[i].x;
					maxLoc.y += faces[i].y;
					circle(currframe_mat, maxLoc, 5, color, 3);
					
				}
			}*/
			if(recordcandidates)
			{
				for (int j = 0; j < keypoints.size(); j++)
				{
				//std::cout << keypoints[i].pt.x << ", " << keypoints[i].pt.y << std::endl;
				//if(isInRect(keypoints[j].pt, faces[i]))
					//circle(currframe_mat, keypoints[j].pt, 15, color, 3);
				
				
					int x, y;
					x = (int)(keypoints[j].pt.x - TIMGW/2);
					y = (int)(keypoints[j].pt.y - TIMGH/2);
					
					if(x < 0 || y < 0 ||
						x + TIMGW >= CAMWIDTH ||
						y + TIMGH >= CAMHEIGHT)
						continue;
					
					Rect candidateRegion(x, y, TIMGW, TIMGH);
				
					sprintf(filename, "candidates/candidate%d_%d_%02d%02d%02d_%d_%d.jpg", 
							currrunstamp_tm->tm_mon, 
							currrunstamp_tm->tm_mday,
							currrunstamp_tm->tm_hour,
							currrunstamp_tm->tm_min,
							currrunstamp_tm->tm_sec,
							framecount,
							j);
					if(arduino_state != '0')
					{
						Mat candidateimg = currframe_gray(candidateRegion);
						imwrite(filename, candidateimg);						
					}
					else
					{
						Mat candidateimg = prevframe_gray(candidateRegion);
						imwrite(filename, candidateimg);
					}
				}
				
				recordcandidates = 0;
			}
			
			
			j++;
			/* testing synchronization and checking bright and dark images
			
			if(j < 1000)
			{
				char filename[100];
				if(arduino_state == '0')
				{
					sprintf(filename, "brightimgs/brightimg%d.jpg", j/2);
					imwrite(filename, currframe_mat);
				}
				else
				{
					sprintf(filename, "darkimgs/darkimg%d.jpg", j/2);
					imwrite(filename, currframe_mat);
				}
				j++;
			}*/
			
			//equalizeHist(diff_img,diff_img);
			//threshold(diff_img, diff_img, (int)adapt_thresh, COLOUR, THRESH_BINARY);
			//smooth(diff_img, diff_img, CV_BLUR);
			imshow("difference", diff_img);
			imshow("currframe_mat", currframe_mat);
			if(SHOWHIST)
				calcplothist(diff_img);
			if(USEOPENOP && !isOn(arduino_state))
			{
				element = getStructuringElement(MORPH_CROSS, Size( 2*MORPHSIZE + 1, 2*MORPHSIZE+1 ), Point( MORPHSIZE, MORPHSIZE ) );
				morphologyEx( currframe_gray, openedimg, OPENOP, element );
				absdiff(currframe_gray, openedimg, openeddiffimg);
				if(i%(EXPECTEDFPS*THRESHREFRESHRATE) == 1) // recalculate threshold
				{
					minMaxLoc(openeddiffimg, &min, &max, &minloc, &maxloc);
					opened_adapt_thresh = (max - min)*.20 + min;
					//std::cout << "adjusted second thresh\n";
				}
				//threshold(openeddiffimg, openeddiffimg, opened_adapt_thresh, COLOUR, THRESH_BINARY);
				imshow("morphopendiff", openeddiffimg);
				
				bitwise_and(openeddiffimg, diff_img, overlappedimg);
				//threshold(overlappedimg, overlappedimg, COLOUR + 1, 255, THRESH_BINARY);
				imshow("overlappedimg", overlappedimg);
			}
		//}
		
		//imshow("coloreddiff", colored_diff_img);
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

		int keyVal = cvWaitKey(wait_time) & 255;

		if ( (keyVal) == 27 )
		{
			if (DEBUGON) printf("time elapsed %f usecs. wait time = %d msecs\n", time_elapsed, wait_time);
		}

		else if ( (keyVal) == 't' )
		{
			flip(arduino_state);	
		}

		else if ( (keyVal) == 'u' )
		{
			thresh_cut += .05;
			if (DEBUGON) printf("thresh_cut value updated to %f\n", thresh_cut);
			minMaxLoc(diff_img, &min, &max, &minloc, &maxloc);
			adapt_thresh = (max - min)*thresh_cut + min;
		}
		
		else if ( (keyVal) == 'd' )
		{
			thresh_cut -= .05;
			if (DEBUGON) printf("thresh_cut value updated to %f\n", thresh_cut);
			minMaxLoc(diff_img, &min, &max, &minloc, &maxloc);
			adapt_thresh = (max - min)*thresh_cut + min;
		}
		
		else if( (keyVal) == 's' )
		{
			imwrite("diff_img.jpg", diff_img);
		}
		
		else if( (keyVal) == 32) // spacebar
		{
			if (captureFlag==false)
			{
				captureFlag=true;
				if (DEBUGON) printf("video is now on\n");
				record = new VideoWriter("ICwhatUCVideo.avi", CV_FOURCC('M','J','P','G'), 30, diff_img.size(), true);
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
		else if ( (keyVal) == 'c' )
		{
			recordcandidates = 1;
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
					ss = &diff_img; break;
			}
						
			sprintf(filename, "./snapshots/%s_%d-%d-%d_%d:%d:%d.jpg", imgtype, 
				(now->tm_year + 1900), (now->tm_mon + 1), now->tm_mday, 
				now->tm_hour, now->tm_min, now->tm_sec);
			imwrite(filename, *ss);
		}
		
		// record video
		if (captureFlag == true)
		{
			if (DEBUGON) printf("Recording frame\n");
			Mat diff_img_color;
			cvtColor(diff_img, diff_img_color, CV_GRAY2BGR);
			(*record) << diff_img_color; 
		}
		
		framecount++;
		flip(arduino_state);	
		
	}
	// Release the capture device housekeeping
	cvReleaseCapture( &capture );
	cvDestroyWindow( "mywindow" );
	return 0;
}

