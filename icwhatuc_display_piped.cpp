#include <cstdio>  
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <cmath>
#include <vector>
#include <dirent.h>
#include <ctime>
#include <string>

#include <ml.h>
#include <cv.h> 
#include <highgui.h>
#include "opencv2/gpu/gpu.hpp"

#define HEATMAP_WIN		"heatmap"
#define DISPLAY_WIN		"display"
#define WINDOW_PROPS	CV_WINDOW_NORMAL

#define DISPLAYHEIGHT	1080
#define DISPLAYWIDTH	1920
#define DISPLAYCOLS		5
#define DISPLAYROWS		2

#define HEATMAPHEIGHT	1080
#define HEATMAPWIDTH	1920

#define RADIUSOFGAZE	70

#define UPDATETHRESHOLD	50

using namespace std;

void initWindows()
{
	cv::namedWindow(HEATMAP_WIN, WINDOW_PROPS);
	cv::namedWindow(DISPLAY_WIN, WINDOW_PROPS);
	cv::moveWindow(DISPLAY_WIN,0,0);
	cv::setWindowProperty(DISPLAY_WIN, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
}

void loadImages(char *path, std::vector<cv::gpu::GpuMat> &imagelist)
{
	DIR *imagedp;
	imagedp = opendir(path);
	
	if (imagedp == NULL)
	{
		fprintf(stderr, "ERROR: Couldn't open specified directory to load posters <%s>\n.", path);
		exit(-1);
	}
	
	struct dirent *ep;
	
	while (ep = readdir(imagedp)) {
		std::string imgname = ep->d_name;
		if (imgname.compare(".") == 0 || imgname.compare("..") == 0)
			continue;
		
		cv::Mat img = cv::imread(path + imgname);
		cv::resize(img, img, cv::Size(DISPLAYWIDTH/DISPLAYCOLS, DISPLAYHEIGHT/DISPLAYROWS));
		cv::gpu::GpuMat img_g;
		img_g.upload(img);
		
		imagelist.push_back(img_g);
	}
}

void cycleImageDisplay(std::vector<cv::gpu::GpuMat> &imagelist)
{
	int i;
	
	for(i = 0; i < imagelist.size(); i++)
	{
		cv::Mat currimg;
		imagelist[i].download(currimg);
		cv::imshow(DISPLAY_WIN, currimg);
		
		cvWaitKey(2000);
	}
}

cv::gpu::GpuMat getImageCollage(std::vector<cv::gpu::GpuMat> &imagelist)
{
	cv::Mat collage(DISPLAYHEIGHT, DISPLAYWIDTH, CV_8UC3, cv::Scalar::all(0));
	cv::gpu::GpuMat collage_g;
	
	for (int r=0; r<DISPLAYROWS; r++) {
		for (int c=0; c<DISPLAYCOLS; c++) {
			cv::Rect roi(
				c*DISPLAYWIDTH/DISPLAYCOLS,
				r*DISPLAYHEIGHT/DISPLAYROWS,
				(DISPLAYWIDTH/DISPLAYCOLS),
				(DISPLAYHEIGHT/DISPLAYROWS)
			);
			
			cv::Mat collageroi = collage(roi);
			
			imagelist[r*DISPLAYCOLS+c].download(collageroi);
		}
	}

	collage_g.upload(collage);
	return collage_g;
}

cv::gpu::GpuMat imageOverlay(cv::gpu::GpuMat &original, cv::gpu::GpuMat &overlayimg)
{
	cv::gpu::GpuMat result;
	cv::gpu::addWeighted(original, 0.25, overlayimg, 0.75, 0.0, result);
	
	return result;
}


// generate the heatmap
cv::gpu::GpuMat heatmap(cv::gpu::GpuMat &stateMat, std::vector<cv::Point> &centerPoints)
{
	for (int ii=0; ii<centerPoints.size(); ii++)
	{
		// draw a circle in cpu
		cv::Mat circ(DISPLAYHEIGHT, DISPLAYWIDTH, CV_8UC3, cv::Scalar::all(0));
		// we are drawing red circles
		circle(circ, centerPoints[ii],RADIUSOFGAZE,CV_RGB(25,0,0),-1);


		// convert it to gpu
	
		cv::gpu::GpuMat orig;
		orig.upload(circ);
	
		// add in gpu
		cv::gpu::add(orig,stateMat,stateMat);
	}

	return stateMat;
}

// create window resize height
cv::gpu::GpuMat bigPoster(cv::gpu::GpuMat poster)
{
	// resize GpuMat
	
	int newx = DISPLAYHEIGHT/poster.cols*poster.rows;
	int newy = DISPLAYHEIGHT;
	cv::gpu::GpuMat enlarged(newx,newy, CV_8UC3, cv::Scalar::all(0));
	
	cv::gpu::resize(poster,enlarged,cv::Size(newx,newy),0,0,cv::INTER_CUBIC);

	// Show it on the window
	return enlarged;
}

int pointInput = 0, switchPoster = 0, desiredPosterRow, desiredPosterCol;

int posterhits[DISPLAYROWS][DISPLAYCOLS];
void mouseEvent(int event, int x, int y, int flags, void* ptr) {
	if(event == CV_EVENT_LBUTTONDOWN) {
		((std::vector<cv::Point>*)ptr)->push_back(cv::Point(x,y));

		pointInput = 1;

		if(++posterhits[y*DISPLAYROWS/DISPLAYHEIGHT][x*DISPLAYCOLS/DISPLAYWIDTH] > UPDATETHRESHOLD)
			switchPoster = 1;
		
		desiredPosterRow = y*DISPLAYROWS/DISPLAYHEIGHT;
		desiredPosterCol = x*DISPLAYCOLS/DISPLAYWIDTH;
	}
}

bool wasUsed(int index, std::vector<int> &usedUpIndices)
{
	int i;
	for(i = 0; i < usedUpIndices.size(); i++)
	{
		if(usedUpIndices[i] == index)
			return true;
	}
	
	return false;
}

void randomImageSelection(std::vector<cv::gpu::GpuMat> &bag, std::vector<cv::gpu::GpuMat> &selection)
{
	int i;
	std::vector<int> usedIndices;
	
	/* initialize random seed: */
	srand (time(NULL));
	
	selection.clear();
	
	for(i = 0; i < DISPLAYCOLS*DISPLAYROWS; i++)
	{
		int randindex;
		do
		{
			randindex = rand()%bag.size();
		}while(wasUsed(randindex, usedIndices));
		
		
		usedIndices.push_back(randindex);
		
		selection.push_back(bag[randindex]);
	}
}

void emptyPosterHits()
{
	memset(posterhits,0,sizeof(posterhits));

	/*
	int row, col;
	for(row = 0; row < DISPLAYROWS; row++)
	{
		for(col = 0; col < DISPLAYCOLS; col++)
		{
			posterhits[row][col] = 0;
		}
	}
	*/
}

int main(int argc, char *argv[])
{
	if(argc != 2)
	{
		fprintf(stderr, "ERROR: Unexpected number of commandline arguments.  Please provide only the directory path with the posters as an argument.\n");
		return -1;
	}
	
	initWindows();
	
	std::vector<cv::gpu::GpuMat> bagofimages, currimagesdisplayed;
	std::vector<cv::Point> points;
	
	setMouseCallback(DISPLAY_WIN, mouseEvent, &points);
	
	loadImages(argv[1], bagofimages);
	//cycleImageDisplay(bagofimages);
	cv::Mat emptyheatmap(DISPLAYHEIGHT, DISPLAYWIDTH, CV_8UC3, cv::Scalar::all(0));
	cv::gpu::GpuMat heatmap_g(DISPLAYHEIGHT, DISPLAYWIDTH, CV_8UC3, cv::Scalar::all(0));
	cv::Mat mycollage, myheatmap_collage;
	
	//select random images
	randomImageSelection(bagofimages, currimagesdisplayed);
	
	//create collage
	cv::gpu::GpuMat mycollage_g = getImageCollage(currimagesdisplayed);
	mycollage_g.download(mycollage);
		
	//store string from input
	std::string line;
	while(!std::cin.eof())
	{			
		cv::imshow(DISPLAY_WIN, mycollage);

		if(switchPoster)
		{
			cv::Mat imageOfInterest;
			currimagesdisplayed[desiredPosterRow*DISPLAYCOLS + desiredPosterCol].download(imageOfInterest);
			
			cv::imshow(DISPLAY_WIN, imageOfInterest);
			cv::imshow(HEATMAP_WIN, imageOfInterest);
			switchPoster = 0;
			pointInput = 0;
			
			cvWaitKey(5000);
			
			// next set of posters
			randomImageSelection(bagofimages, currimagesdisplayed);
			mycollage_g = getImageCollage(currimagesdisplayed);
			mycollage_g.download(mycollage);
			
			// clear heat map
			emptyPosterHits();
			heatmap_g.upload(emptyheatmap);
			cv::gpu::GpuMat heatmap_collage_g = imageOverlay(mycollage_g, heatmap_g);
			heatmap_collage_g.download(myheatmap_collage);
			cv::imshow(HEATMAP_WIN, myheatmap_collage);
		}
		else if(pointInput)
		{
			heatmap_g = heatmap(heatmap_g, points);
	
			cv::gpu::GpuMat heatmap_collage_g = imageOverlay(mycollage_g, heatmap_g);
			heatmap_collage_g.download(myheatmap_collage);
			cv::imshow(HEATMAP_WIN, myheatmap_collage);
			
			pointInput = 0;
			points.clear();
		}
		
		switch(cv::waitKey(1))
		{
		case 'x': case 'q':
			return 0;
		case ' ':
			// next set of posters
			randomImageSelection(bagofimages, currimagesdisplayed);
			mycollage_g = getImageCollage(currimagesdisplayed);
			mycollage_g.download(mycollage);
			
			// clear heat map
			emptyPosterHits();
			heatmap_g.upload(emptyheatmap);
			cv::gpu::GpuMat heatmap_collage_g = imageOverlay(mycollage_g, heatmap_g);
			heatmap_collage_g.download(myheatmap_collage);
			cv::imshow(HEATMAP_WIN, myheatmap_collage);
			break;
		}

		std::getline(std::cin, line);
		std::istringstream iss(line);
		int x, y;
		while (iss >> x, iss >> y) {
			points.push_back(cv::Point(x,y));
			pointInput = 1;
			if(++posterhits[y*DISPLAYROWS/DISPLAYHEIGHT][x*DISPLAYCOLS/DISPLAYWIDTH] > UPDATETHRESHOLD)
				switchPoster = 1;
		
			desiredPosterRow = y*DISPLAYROWS/DISPLAYHEIGHT;
			desiredPosterCol = x*DISPLAYCOLS/DISPLAYWIDTH;
		}
	}
}
