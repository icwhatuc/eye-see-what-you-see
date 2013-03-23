#include <iostream>
#include <vector>
#include <sys/time.h>

// opencv headers
#include <cv.h> 
#include <highgui.h> 
#include <ml.h>
// opencv gpu
#include "opencv2/gpu/gpu.hpp"

// eyepair header
#include "eyegaze.h"

// camera info
#define CAM_WIDTH   1920
#define CAM_HEIGHT  1080

// display screen info
#define TV_WIDTH		(40/12)				// ft
#define TV_HEIGHT		(23.5/12) 			// ft
#define TV_PX_PER_FT 	(CAM_WIDTH/TV_WIDTH)

// display windows
#define W_COLOR  	"Color frame - 2"
#define W_DIFF   	"Difference Image"
#define W_THRESH 	"Thresholded Difference"

// radius of eye
#define REYE      0.0416667 // feet

// Histogram related constants
#define HIST_BINS   		256
#define HIST_THRESHOLD 		0.0001
#define HIST_DESIREDPIXELS	(CAM_WIDTH*CAM_HEIGHT*HIST_THRESHOLD)

// SVM and Haar Classifier
#define SVM_IMG_SIZE 		75				// training and testing image size
#define SVM_FILE 			"eye_classify_withweights.svm"
#define HAAR_EYE_FILE     	"cascades/haarcascade_eye.xml"
#define HAAR_EYEPAIR_FILE 	"cascades/haarcascade_eyepair.xml"
#define haarEyeClassify(cascade,image) haarClassify(cascade,image,20,20)
#define haarEyePairClassify(cascade,image) haarClassify(cascade,image,60,20)

// colors
#define RED		(CV_RGB(255, 0, 0))
#define GREEN	(CV_RGB(0, 255, 0))
#define BLUE	(CV_RGB(0, 0, 255))
#define YELLOW	(CV_RGB(255, 255, 0))
#define PURPLE	(CV_RGB(255, 0, 255))
#define CYAN	(CV_RGB(0, 255, 255))
#define BLACK	(CV_RGB(0, 0, 0))
#define WHITE	(CV_RGB(255, 255, 255))

#define mark(img, loc, col)	circle(img, loc, 5, col, 3)

using namespace std;
using namespace cv;

void initDisplayWindows();
void initComponents(eyedetectcomponents &edcs);
void checkKnownPairs(eyedetectcomponents &edcs);
void lookForNewEyes(eyedetectcomponents &edcs);
bool svmEyeClassify(CvSVM &svm, Mat &image);
inline unsigned char calcThreshold(gpu::GpuMat &img);
Rect haarClassify(gpu::CascadeClassifier_GPU &cascade, gpu::GpuMat &image, int w, int h);
inline gpu::GpuMat getEyePairImage(Mat &mat_image, Rect &r);
inline Rect getEyePairRect(Point &kp1, Point &kp2);
vector<Point> gazePoints(float d, Point &centerLocLeft, Point &centerLocRight, Point &eyeLocLeft, Point &eyeLocRight);
float getDistFromCamera(Point &kp1, Point &kp2);
void timing(bool start, string what="");
void checkNoses(eyedetectcomponents &edcs);

int main(int argc, char *argv[])
{
	// setup video capture
	VideoCapture capture;

	if(argc == 1)
		capture.open(0); 		// default to camera if no video file provided
	else
		capture.open(argv[1]);

	if(!capture.isOpened()) {
		cerr << "ERROR: capture is NULL\n";
		return -1;
	}

	capture.set(CV_CAP_PROP_FRAME_WIDTH, CAM_WIDTH);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT);

	initDisplayWindows();
	
	eyedetectcomponents edcs;
	initComponents(edcs);

	capture >> edcs.currColorFrame;
	cvtColor(edcs.currColorFrame, edcs.prevgrayframe2_m, CV_BGR2GRAY);

	while(1)
	{
		gpu::GpuMat frame1_g, frame2_g, grayframe1_g, grayframe2_g;
		
		// capture 2 consecutive frames
		capture >> edcs.currColorFrame;
		frame1_g.upload(edcs.currColorFrame);
		gpu::cvtColor(frame1_g, grayframe1_g, CV_BGR2GRAY);
		grayframe1_g.download(edcs.grayframe1_m);
	
		capture >> edcs.currColorFrame;
		frame2_g.upload(edcs.currColorFrame);
		gpu::cvtColor(frame2_g, grayframe2_g, CV_BGR2GRAY);
		grayframe2_g.download(edcs.grayframe2_m);
	
		// update noses for eye pairs via optical flow
		checkNoses(edcs);
		// update the previous soon as optical flow is done in checkNoses fn
		// so that zeroing out the local regions in checkKnownPairs does not
		// affect optical flow.
		edcs.prevgrayframe2_m = edcs.grayframe2_m.clone();

		// confirm known eyepairs
		checkKnownPairs(edcs);
		
		// process entire frames for new eyes
		lookForNewEyes(edcs);

		// Show images
		// DEBUG: show center of screen
		circle(
			edcs.currColorFrame, 
			Point2f(CAM_WIDTH/2,CAM_HEIGHT/2), 
			2, CV_RGB(255,255,255), 2
		);
		
		imshow(W_COLOR, edcs.currColorFrame);
		
		// End of loop -- check for inputs
		switch(waitKey(1)) {
		case 27: // ESC
		case 'x': 
			return 0;
		case 'c':
			if (edcs.knownpairs.size() == 1) {
				cout << "calibrating..." << endl;
				edcs.knownpairs[0].calibrationPoints[0] = edcs.knownpairs[0].eyes[0];
				edcs.knownpairs[0].calibrationPoints[1] = edcs.knownpairs[0].eyes[1];
				edcs.knownpairs[0].calibrationPoints_orig[0] = edcs.knownpairs[0].eyes[0];
				edcs.knownpairs[0].calibrationPoints_orig[1] = edcs.knownpairs[0].eyes[1];
				cout << "calibrated" << endl;
			}
			break;
		}	
	}
}

void initDisplayWindows()
{
	// Display windows
	namedWindow(W_DIFF, CV_WINDOW_NORMAL);
	namedWindow(W_THRESH, CV_WINDOW_NORMAL);
	namedWindow(W_COLOR, CV_WINDOW_NORMAL);
	
	// Move color image window to (0,0) and make fullscreen
	moveWindow(W_COLOR,0,0);
	setWindowProperty(W_COLOR, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
}

void initComponents(eyedetectcomponents &edcs)
{
	// SVM
	edcs.svm.load(SVM_FILE);

	// Haar
	if (!edcs.eye_cascade.load(HAAR_EYE_FILE)) {
		cerr << "ERROR: could not load haar cascade xml for eyes\n";
		exit(-1);
	}
	if (!edcs.eyepair_cascade.load(HAAR_EYEPAIR_FILE)) {
		cerr << "ERROR: could not load haar cascade xml for eye pairs\n";
		exit(-1);
	}

	// Blob detection edcs.params
	edcs.params.minDistBetweenBlobs = 30.0f;
	edcs.params.filterByInertia = false;
	edcs.params.filterByConvexity = false;
	edcs.params.filterByColor = false;
	edcs.params.filterByCircularity = false;
	edcs.params.filterByArea = true;
	edcs.params.minArea = 4.0f;
	edcs.params.maxArea = 200.0f;
	edcs.params.minCircularity = 0.3f;
	edcs.params.maxCircularity = 1.0f;
	edcs.params.thresholdStep = 1;
}

// confirmKnownPairs carries out the typical method of taking frame difference
// in local regions where eyes are known to update locations and blanks out the 
// regions
// instead of blob detection - simple minMaxLoc is used
void checkKnownPairs(eyedetectcomponents &edcs)
{
	int pair, eye;
	//for each known eye
	for(pair = 0; pair < edcs.knownpairs.size(); pair++) {
		for(eye = 0; eye < 2; eye++) {
			//take local regions from gray1 and gray2
			int x = edcs.knownpairs[pair].eyes[eye].x - SVM_IMG_SIZE/2-1;
			int y = edcs.knownpairs[pair].eyes[eye].y - SVM_IMG_SIZE/2-1;
			if (x < 0 || y < 0 || x + SVM_IMG_SIZE >= CAM_WIDTH || y + SVM_IMG_SIZE >= CAM_HEIGHT) {
				edcs.knownpairs.erase(edcs.knownpairs.begin() + pair--);
				break;
			}

			Rect relevant_region(x, y, SVM_IMG_SIZE, SVM_IMG_SIZE);
			
			Mat grayregion1 = edcs.grayframe1_m(relevant_region), grayregion2 = edcs.grayframe2_m(relevant_region);
			gpu::GpuMat pr_grayregion1_g(grayregion1), pr_grayregion2_g(grayregion2), diffimg_g, diffimgblurred_g;
			
			// get the difference image
			gpu::absdiff(pr_grayregion1_g, pr_grayregion2_g, diffimg_g);
			gpu::blur(diffimg_g, diffimgblurred_g, Size(3, 3), Point(-1,-1));
	
			// take the brightest location in the new difference as the updated location of eye
			Point confirmedEye;
			
			gpu::minMaxLoc(diffimgblurred_g, 0, 0, 0, &confirmedEye);
			
			edcs.knownpairs[pair].eyes[eye].x = confirmedEye.x + x;
			edcs.knownpairs[pair].eyes[eye].y = confirmedEye.y + y;

			// DEBUG
			mark(edcs.currColorFrame,edcs.knownpairs[pair].eyes[eye],GREEN);

			x = edcs.knownpairs[pair].eyes[eye].x - SVM_IMG_SIZE/2-1;
			y = edcs.knownpairs[pair].eyes[eye].y - SVM_IMG_SIZE/2-1;
		
			// if the eye is now on the edge - stop tracking it
			if(x < 0 || y < 0 || x + SVM_IMG_SIZE >= CAM_WIDTH || y + SVM_IMG_SIZE >= CAM_HEIGHT) {
				edcs.knownpairs.erase(edcs.knownpairs.begin() + pair--);
				break;
			}

			// else confirm that it is an eye with svm
			Rect regionToZero(x, y, SVM_IMG_SIZE, SVM_IMG_SIZE);
			
			Mat r1 = edcs.grayframe1_m(regionToZero);
			Mat r2 = edcs.grayframe2_m(regionToZero);
			
			if(!svmEyeClassify(edcs.svm, r2)) {
				edcs.knownpairs.erase(edcs.knownpairs.begin() + pair--);
				cout << "SVM CHECK FAILED!!!" << endl;
				break;
			}

			// blank out the regions if classifier confirms it is an eye
			r1 = Mat::zeros(SVM_IMG_SIZE, SVM_IMG_SIZE, CV_8UC1);
			r2 = Mat::zeros(SVM_IMG_SIZE, SVM_IMG_SIZE, CV_8UC1);
		
			// mark on color image the new eye locations for display purposes
			if(eye) {
				mark(edcs.currColorFrame, edcs.knownpairs[pair].eyes[0], BLACK);
				mark(edcs.currColorFrame, edcs.knownpairs[pair].eyes[1], BLACK);
				mark(edcs.currColorFrame, edcs.knownpairs[pair].nose, PURPLE);
			}
		}
	}
}

inline unsigned char calcThreshold(gpu::GpuMat &img_g)
{
	gpu::GpuMat hist_g;
	Mat hist_m;
	
	unsigned int histcount = 0;
	unsigned char threshbin;

	gpu::calcHist(img_g, hist_g);
	hist_g.download(hist_m);
	
	for (threshbin = HIST_BINS-1; threshbin > 0; threshbin--) {
		histcount += hist_m.at<uint>(threshbin);
		if (histcount > HIST_DESIREDPIXELS)
			break;
	}

	return threshbin;
}

void lookForNewEyes(eyedetectcomponents &edcs)
{
	gpu::GpuMat grayframe1_g, grayframe2_g, diffimg_g, blurreddiffimg_g, threshimg_g;
	Mat diffimg_m;

	vector<KeyPoint> eyeCandidates;
	vector<Point> detectedEyes;

	grayframe1_g.upload(edcs.grayframe1_m);
	grayframe2_g.upload(edcs.grayframe2_m);

	// Obtain difference image and blur
	gpu::absdiff(grayframe2_g, grayframe1_g, diffimg_g);
	gpu::blur(diffimg_g, blurreddiffimg_g, Size(3, 3), Point(-1,-1));

	// Thresholding - find threshold based on histogram
	unsigned char threshold = calcThreshold(blurreddiffimg_g);
	
	// DEBUG: For display purposes only
	gpu::threshold(diffimg_g, threshimg_g, threshold, 255, CV_THRESH_BINARY);
	Mat threshimg_m;
	threshimg_g.download(threshimg_m);

	imshow(W_THRESH, threshimg_m);
	
	// detect possible eye candidates based on thresholded image (done within 
	// detection) and return as currentCandidates
	edcs.params.minThreshold = threshold-1;
	edcs.params.maxThreshold = threshold+1;
	Ptr<FeatureDetector> blob_detector = new SimpleBlobDetector(edcs.params);
	blob_detector->create("SimpleBlob");
	diffimg_g.download(diffimg_m); // Blob detect doesn't work on GpuMat objects
	blob_detector->detect(diffimg_m, eyeCandidates);
	
	imshow(W_DIFF, diffimg_m);
	// Iterate through currentCandidates and check using SVM and Haar cascades
	int c;
	for (c = 0; c < eyeCandidates.size(); c++) {
		// Top left corner of candidate region
		int x = eyeCandidates[c].pt.x - SVM_IMG_SIZE/2-1;
		int y = eyeCandidates[c].pt.y - SVM_IMG_SIZE/2-1;

		// ignore edge cases
		if(x < 0 || y < 0 || x + SVM_IMG_SIZE >= CAM_WIDTH || y + SVM_IMG_SIZE >= CAM_HEIGHT)
			continue;

		Rect candidate_region(x, y, SVM_IMG_SIZE, SVM_IMG_SIZE);
		Mat candidateimg_m = edcs.grayframe2_m(candidate_region);
		gpu::GpuMat candidateimg_g(candidateimg_m);

		// check against haar classifier
		Rect eye_rect = haarEyeClassify(edcs.eye_cascade, candidateimg_g);
		bool haar_detected = eye_rect.area() > 1;
		
		// check against svm
		bool svm_detected = svmEyeClassify(edcs.svm,candidateimg_m);
		

		if (svm_detected)
			mark(edcs.currColorFrame, eyeCandidates[c].pt, RED);

		if (haar_detected)
			mark(edcs.currColorFrame, eyeCandidates[c].pt, YELLOW);

		if (svm_detected || haar_detected){
			Point detectedEye(eyeCandidates[c].pt.x, eyeCandidates[c].pt.y);
			detectedEyes.push_back(detectedEye);
		}	
	}


	// Pair eyes
	for (int eye1 = 0; eye1 < detectedEyes.size(); eye1++) {
		for (int eye2 = eye1+1; eye2 < detectedEyes.size(); eye2++) {
			Rect eyepair_rect_in, eyepair_rect_out;
			Point left_eye = detectedEyes[eye1], right_eye = detectedEyes[eye2];
			
			if (detectedEyes[eye1].x > detectedEyes[eye2].x)
				swap(left_eye,right_eye);

			eyepair_rect_in = getEyePairRect(left_eye, right_eye);
			gpu::GpuMat eyepair_g = getEyePairImage(edcs.grayframe2_m, eyepair_rect_in);

			int x = eyepair_rect_in.x;
			int y = eyepair_rect_in.y;
			eyepair_rect_out = haarEyePairClassify(edcs.eyepair_cascade, eyepair_g);
			
			if (eyepair_rect_out.area() > 1) { // Eyepair detected
				// Draw rectangle around eyepair
				rectangle(
					edcs.currColorFrame,
					Point(eyepair_rect_in.x, eyepair_rect_in.y),
					Point(eyepair_rect_in.width+eyepair_rect_in.x, eyepair_rect_in.height+eyepair_rect_in.y),
					GREEN, 2
				);

				// Place point for nose based on distance between eyes
				Point nose(
					(left_eye.x + right_eye.x)/2,
					(left_eye.y + right_eye.y)/2 + 0.7*(right_eye.x - left_eye.x)
				);

				eyepair currenteyepair;
				
				currenteyepair.eyes[0] = detectedEyes[eye1];
				currenteyepair.eyes[1] = detectedEyes[eye2];
				currenteyepair.nose = nose;
				currenteyepair.nose_orig = nose;

				edcs.knownpairs.push_back(currenteyepair);

				detectedEyes.erase(detectedEyes.begin() + eye1--);
				detectedEyes.erase(detectedEyes.begin() + --eye2);
				
				//DEBUG
				char dist_str[256];
				float dist_from_cam = getDistFromCamera(left_eye,right_eye);
				sprintf(
					dist_str, 
					"%.1fft away",
					dist_from_cam
				);
				putText(edcs.currColorFrame, dist_str, Point(x,y), FONT_HERSHEY_COMPLEX_SMALL,
					1, CV_RGB(255,0,0), 1, CV_AA);

				//DEBUG
				char eyepos_str[256];
				sprintf(
					eyepos_str,
					"(%d, %d) and (%d, %d)",
					left_eye.x,
					left_eye.y,
					right_eye.x,
					right_eye.y
				);
				putText(edcs.currColorFrame, eyepos_str, Point(x,y-40), FONT_HERSHEY_COMPLEX_SMALL,
					1, CV_RGB(255,0,0), 1, CV_AA);
			}
		}
	}

	for(int pair = 0; pair < edcs.knownpairs.size(); pair++)
	{
		// Calibration
		vector<Point> gaze_loc;
		if (edcs.knownpairs[pair].isCalibrated)
		{
			float dist_from_cam = getDistFromCamera(edcs.knownpairs[pair].eyes[0], edcs.knownpairs[pair].eyes[1]);
			gaze_loc = gazePoints(
				dist_from_cam,
				edcs.knownpairs[pair].calibrationPoints[0],
				edcs.knownpairs[pair].calibrationPoints[1],
				edcs.knownpairs[pair].eyes[0],
				edcs.knownpairs[pair].eyes[1]
			);
		}

		mark(edcs.currColorFrame, gaze_loc[0], WHITE);
	}
}

bool svmEyeClassify(CvSVM &svm, Mat &image)
{
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

Rect haarClassify(gpu::CascadeClassifier_GPU &cascade, gpu::GpuMat &image, int w, int h) 
{
	gpu::GpuMat features, eqImage;
	const float scaleFactor = 1.2;
	const int minNeighbors = 3;
	Size minSize = Size(w,h);

	gpu::equalizeHist(image, eqImage);
	eqImage.convertTo(eqImage,CV_8U);
	int numDetected = cascade.detectMultiScale(eqImage, features, scaleFactor, minNeighbors, minSize);

	// If no features detected, return rectangle with area 1
	if (numDetected == 0)
		return Rect(0,0,1,1);
	
	Mat tmpMat;
	features.colRange(0, numDetected).download(tmpMat);
	Rect *featureRect = tmpMat.ptr<Rect>();
	return featureRect[0];
}

inline gpu::GpuMat getEyePairImage(Mat &mat_image, Rect &r)
{
	Mat eye_pair = mat_image(r);
	Mat resized_pair(SVM_IMG_SIZE, SVM_IMG_SIZE*3, CV_8UC1);
	resize(eye_pair, resized_pair, resized_pair.size(), 1, 1);
	return gpu::GpuMat(resized_pair);
}

inline Rect getEyePairRect(Point &kp1, Point &kp2)
{
	int x, y;	

	Rect r(kp1, kp2);
	r.x -= SVM_IMG_SIZE/2-1;
	r.y -= SVM_IMG_SIZE/2-1;
	r.height += SVM_IMG_SIZE;
	r.width += SVM_IMG_SIZE;
	
	if (r.x < 0)
		r.x=0;
	if (r.y < 0)
		r.y=0;
	if (r.x + r.width >= CAM_WIDTH)
		r.x = CAM_WIDTH - r.width - 1;
	if	(r.y + r.height >= CAM_HEIGHT)
		r.y = CAM_HEIGHT - r.height - 1;

	return r;
}

vector<Point> gazePoints(float d, Point &centerLocLeft, Point &centerLocRight, Point &eyeLocLeft, Point &eyeLocRight)
{
	float delxLeft, delyLeft, delxRight, delyRight;
	vector<Point> gazeLoc;

	// pixel to feet - computed from distance from camera
	float ftPerPx = d/1500;		// units ft/px

	// obtain the delta x and delta y - in feet
	delxLeft = (eyeLocLeft.x - centerLocLeft.x)*ftPerPx;
	delyLeft = (eyeLocLeft.y - centerLocLeft.y)*ftPerPx;

	delxRight = (eyeLocRight.x - centerLocRight.x)*ftPerPx;
	delyRight = (eyeLocRight.y - centerLocRight.y)*ftPerPx;

	// compute the approximate location on screen
	Point2f gazeShiftL, gazeShiftR; 

	gazeShiftL.x = delxLeft/REYE*(REYE+d);
	gazeShiftL.y = delyLeft/REYE*(REYE+d);

	gazeShiftR.x = delxRight/REYE*(REYE+d);
	gazeShiftR.y = delyRight/REYE*(REYE+d);

	// take average of both eyes
	Point gazeLocTemp;

	gazeLocTemp.x = (int)(-(gazeShiftL.x)/ftPerPx + CAM_WIDTH/2);
	gazeLocTemp.y = (int)((gazeShiftL.y)/ftPerPx + CAM_HEIGHT/2);

	gazeLoc.push_back(gazeLocTemp);

	gazeLocTemp.x = (int)(-(gazeShiftR.x)*TV_PX_PER_FT + CAM_WIDTH/2);
	gazeLocTemp.y = (int)((gazeShiftR.y)*TV_PX_PER_FT + CAM_HEIGHT/2);

	gazeLoc.push_back(gazeLocTemp);

	return gazeLoc;
}

float getDistFromCamera(Point &kp1, Point &kp2)
{
	float dx = kp2.x - kp1.x;
	float dy = kp2.y - kp1.y;
	float distBetweenEyes = sqrt(dx*dx + dy*dy); // Dist in pixels
	return 300/distBetweenEyes; // 300: constant determined experimentally
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

void checkNoses(eyedetectcomponents &edcs)
{
	// Update positions of noses
	vector<Point2f> prevNoses, currNoses;

	vector<uchar> optflow_status;
	vector<float> optflow_err;

	int nose;
	for(nose = 0; nose < edcs.knownpairs.size(); nose++)
		prevNoses.push_back(edcs.knownpairs[nose].nose);

	if (prevNoses.size() > 0) {
		calcOpticalFlowPyrLK(
			edcs.prevgrayframe2_m, edcs.grayframe2_m, 
			prevNoses, currNoses,
			optflow_status, optflow_err
		);
	}
	for(nose = 0; nose < edcs.knownpairs.size(); nose++)
	{
		edcs.knownpairs[nose].calibrationPoints[0].x = edcs.knownpairs[nose].calibrationPoints_orig[0].x + (currNoses[nose].x - edcs.knownpairs[nose].nose_orig.x);
		edcs.knownpairs[nose].calibrationPoints[0].y = edcs.knownpairs[nose].calibrationPoints_orig[0].y + (currNoses[nose].y - edcs.knownpairs[nose].nose_orig.y);
		edcs.knownpairs[nose].calibrationPoints[1].x = edcs.knownpairs[nose].calibrationPoints_orig[1].x + (currNoses[nose].x - edcs.knownpairs[nose].nose_orig.x);
		edcs.knownpairs[nose].calibrationPoints[1].y = edcs.knownpairs[nose].calibrationPoints_orig[1].y + (currNoses[nose].y - edcs.knownpairs[nose].nose_orig.y);
	
		edcs.knownpairs[nose].nose = currNoses[nose];
		mark(edcs.currColorFrame, edcs.knownpairs[nose].calibrationPoints[0], GREEN);
		mark(edcs.currColorFrame, edcs.knownpairs[nose].calibrationPoints[1], GREEN);
		
		cout << edcs.knownpairs[nose].calibrationPoints[1] << endl;
		cout << edcs.knownpairs[nose].calibrationPoints[0] << endl; 
	}
}
