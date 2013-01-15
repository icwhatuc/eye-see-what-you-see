#include <cv.h> 
#include <highgui.h> 
#include <ml.h> 
#include <sys/types.h>
#include <dirent.h>
#include <cstdio>  
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <ctime>

#define EYECLASS	1
#define NONEYECLASS	-1

int count_files(DIR *dp);
float predict_eye(CvSVM &svm, char *img_path);

using namespace cv;
int main()
{
	DIR *dp1, *dp2, *testdp, *currdp;
	struct dirent *ep;
	std::string filename;

	char *dirname1 = "./candidates/pos/";		// positive
	char *dirname2 = "./candidates/neg_old/", *currdirname;	// negative
	char *testdir  = "./candidates/test/";		
	dp1 = opendir(dirname1);
	dp2 = opendir(dirname2);
	testdp = opendir(testdir);
	
	if (dp1 == NULL) {
		perror("Couldn't open directory");
		exit(-1);
	}

	// Get dimensions of first image
	std::string imgname;
	while (ep = readdir(dp1)) {
		imgname = ep->d_name;
		if (imgname.compare(".") && imgname.compare(".."))
			break;
	}
	Mat img = imread(dirname1 + imgname,0);
	int imgsize = img.size().area();

	// Count files to get number of rows in SVM input
	int negcount = count_files(dp2);
	int filecount = count_files(dp1)*2 /* normal eyes & flipped eyes */ + negcount;
	Mat svm_mat(filecount,imgsize,CV_32FC1);

	// Generate the SVM input matrix by reading files
	Mat img_mat, img_mat_flipped;
	Mat labels(filecount,1,CV_32FC1);
	int ii, filenum = 0;
	char *dirname = dirname1;
	std::vector<std::string> filenames;
	
	int negcounter = -1;
	
	fprintf(stderr, "creating svm_mat...\n");
	
	while ((ep = readdir(dp1)) || ((dirname = dirname2) && (ep = readdir(dp2))))
	{
		imgname = ep->d_name;
		if (imgname.compare(".") == 0 || imgname.compare("..") == 0)
			continue;
		ii = 0;
		filenames.push_back(imgname);
		img_mat = imread(dirname + imgname,0);
		equalizeHist(img_mat,img_mat);

		// Copy the pixels from the original image matrix to svm input matrix
		for (int i = 0; i<img_mat.rows; i++) {
			for (int j = 0; j < img_mat.cols; j++) {
				svm_mat.at<float>(filenum,ii++) = img_mat.at<uchar>(i,j);
			}
		}

		if(strcmp(dirname,dirname1) == 0)
		{
			labels.at<float>(filenum) = strcmp(dirname,dirname1) ? NONEYECLASS : EYECLASS;
			flip(img_mat, img_mat_flipped, 1);
			
			filenum++;
			ii = 0;
		
			// Copy the pixels from the original image matrix to svm input matrix
			for (int i = 0; i<img_mat_flipped.rows; i++) {
				for (int j = 0; j < img_mat_flipped.cols; j++) {
					svm_mat.at<float>(filenum,ii++) = img_mat_flipped.at<uchar>(i,j);
				}
			}
		}
		else if(negcounter == -1)
		{
			negcounter = 0;
			fprintf(stderr, "dealing with the negative set...\n");
		}
		else
		{
			negcounter++;
			//fprintf(stderr, "negcount = %d\n", negcounter);
		}
		
		labels.at<float>(filenum) = strcmp(dirname,dirname1) ? NONEYECLASS : EYECLASS;
		filenum++;
		if(negcounter == negcount-1)
			break;
	}
	
	fprintf(stderr, "finished creating the negative set...\n");
	//std::cout << labels << std::endl;

	// Initialize the SVM with parameters
	CvSVMParams params;
	params.svm_type    = CvSVM::C_SVC;
	params.kernel_type = CvSVM::POLY;
	params.gamma = 3;
	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 15000, 1e-6);
	params.degree = 2;
	
	fprintf(stderr, "training svm...\n");
	

	
	// Train the SVM
	CvSVM svm;
	
	// loading doesn't work
	//svm.load("eye_classify_good.svm");
	
	clock_t init = clock();
	svm.train(svm_mat, labels, Mat(), Mat(), params);
	clock_t final = clock() - init;
	fprintf(stderr, "finished training svm...\n");
	std::cout << (double)final/(double)CLOCKS_PER_SEC << " secs\n";
	
	svm.save("eye_classify_oldworking.svm");

	// Print out the filenames that correspond to the support vectors
	int num_supports = svm.get_support_vector_count();
	for (int i = 0; i < num_supports; i++) {
		//std::cout << filenames[*svm.get_support_vector(i)] << std::endl;
	}

#define DESIREDTEST		1
	
	int desiredresult, total, correct;
	if(DESIREDTEST)
	{
		currdp = dp1;
		currdirname = dirname1;
		desiredresult = EYECLASS; total = 0; correct = 0;
	}
	else
	{
		currdp = dp2;
		currdirname = dirname2;
		desiredresult = NONEYECLASS; total = 0; correct = 0;
	}
	
	rewinddir(currdp);
	while (ep = readdir(currdp)) {
		//if (!imgname.compare(".") && !imgname.compare("..")) {
		imgname = ep->d_name;
		//std::cout << imgname << std::endl;
		if (!imgname.compare(".") || !imgname.compare("..")) 
			continue;

		std::string tmp = currdirname+imgname;
		const char *param = tmp.c_str();
		
		int result = (int)predict_eye(svm, (char *)param);
		
		if(result == desiredresult)
			correct++;
		
		total++;
		//std::cout << predict_eye(svm, (char *)param) << std::endl;
		
		//std::cout << predict_eye(svm, dirname1+imgname) << std::endl;
	}
	
	std::cout << "Total: " << total << " & Correct: " << correct << std::endl;
	
	
	while (ep = readdir(testdp)) {
		//if (!imgname.compare(".") && !imgname.compare("..")) {
		imgname = ep->d_name;
		if (!imgname.compare(".") || !imgname.compare("..")) 
			continue;

		std::string tmp = testdir+imgname;
		const char *param = tmp.c_str();
		
		int result = (int)predict_eye(svm, (char *)param);
		std::cout << imgname << "\t" << predict_eye(svm, (char *)param) << std::endl;
		//if (result == 1)
			//std::cout << "mv " << tmp << " ./candidates/neg_pos_pos/" << imgname << std::endl;
		
		//std::cout << predict_eye(svm, dirname1+imgname) << std::endl;
	}
	
}

// Count number of files in directory, not including "." or ".."
int count_files(DIR *dp) {
	struct dirent *ep;
	int filecount = 0;
	std::string imgname;
	rewinddir(dp);
	while (ep = readdir(dp)) {
		imgname = ep->d_name;
		if (imgname.compare(".") && imgname.compare(".."))
			filecount++;
	}
	rewinddir(dp);
	return filecount;
}

float predict_eye(CvSVM &svm, char *img_path) {
	Mat img_mat = imread(img_path,0);
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


