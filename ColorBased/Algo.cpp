#include <iostream>
#include "Algo.h"

using namespace std;
using namespace cv;

void DetectBlob (IplImage *image){
	Mat ImageIn = cvarrToMat(image);
	Mat ImageOut = ImageIn.clone();

	// set up the parameters (check the defaults in opencv's code in blobdetector.cpp)
	SimpleBlobDetector::Params params;
	params.minDistBetweenBlobs = 50.0f;
	params.filterByInertia = false;
	params.filterByConvexity = false;
	params.filterByColor = false;
	params.filterByCircularity = false;
	params.filterByArea = false;
	params.minArea = 20.0f;
	params.maxArea = 500.0f;
	// ... any other params you don't want default value
	Ptr<SimpleBlobDetector> d = SimpleBlobDetector::create(params);

	// detect!
	vector<cv::KeyPoint> keypoints;
	d.get()->detect(ImageIn, keypoints);

	drawKeypoints (ImageIn, keypoints, ImageOut);
	// extract the x y coordinates of the keypoints: 

	if(keypoints.size() == 0) cout<<"no points find"<<endl;

	for (int i=0; i<keypoints.size(); i++){
		cout<<"points: "<<keypoints[i].pt<<";"<<endl;
	}

	if(keypoints.size() != 0){
		namedWindow("123", WINDOW_NORMAL);
		imshow("123", ImageOut);
		waitKey(0);
	}
}
