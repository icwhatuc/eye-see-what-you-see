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

int count_files(DIR *dp);

using namespace cv;
int main() {
	DIR *dp1, *dp2;
	struct dirent *ep;
	std::string filename;

	char *dirname1 = "./candidates/pos/";
	char *dirname2 = "./candidates/neg/";
	dp1 = opendir(dirname1);
	dp2 = opendir(dirname2);

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
	int filecount = count_files(dp1) + count_files(dp2);
	Mat svm_mat(filecount,imgsize,CV_32FC1);

	// Generate the SVM input matrix by reading files
	Mat img_mat;
	Mat labels(filecount,1,CV_32FC1);
	int ii, filenum = 0;
	char *dirname = dirname1;
	std::vector<std::string> filenames;
	while ((ep = readdir(dp1)) || ((dirname = dirname2) && (ep = readdir(dp2)))) {
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
		labels.at<float>(filenum) = (dirname == dirname1) ? 1 : -1;
		filenum++;
	}

	// Initialize the SVM with parameters
	CvSVMParams params;
	params.svm_type    = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
	params.gamma = 3;
	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	// Train the SVM
	CvSVM svm;
	svm.train(svm_mat, labels, Mat(), Mat(), params);

	// Print out the filenames that correspond to the support vectors
	int num_supports = svm.get_support_vector_count();
	for (int i = 0; i < num_supports; i++) {
		std::cout << filenames[*svm.get_support_vector(i)] << std::endl;
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

