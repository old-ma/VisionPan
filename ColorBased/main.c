#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "Algo.h"

int Vision_Histogram2D(const IplImage * img_in, int hist_size, int *hist_value);
int Vision_StretchContrast(const IplImage *img_in, IplImage *img_out, int methrod);

int main()
{
	IplImage* Frame;
	IplImage* FrameCopy;
	IplImage* FrameOut;

	CvMat* vector;

	CvMat** vects;
	CvMat* cov_mat;
	CvMat* avg_vec;

	char key;
	int nPixels, nChannels;
	int SizePixel;

	// Get Camera
	CvCapture* pCapture = cvCaptureFromCAM(1);
	if (pCapture == NULL){
		printf("Error to get camera\n");
		return 0;
	}
	else{
		printf("Capture Created\n");
	}

	Frame = cvQueryFrame(pCapture);

	if(Frame != NULL){
		printf("Width: %d\n", Frame->width);
		printf("height: %d\n", Frame->height);
		printf("WidthStep: %d\n", Frame->widthStep);
		printf("Channels: %d\n", Frame->nChannels);
		printf("Depth: %d\n", Frame->depth);
	}


	nPixels = Frame->width * Frame->height;
	nChannels = Frame->nChannels;
	SizePixel = nChannels*1;
 
	/*********start *********/

	// Init point to set of vector
	vects = (CvMat**)malloc( nPixels * sizeof(CvMat *));

	// Init  set of vector;
	for(int i = 0; i < nPixels; i++) vects[i] = cvCreateMat(nChannels, 1,  CV_8UC1);

	// init output
	cov_mat = cvCreateMat(nChannels, nChannels,  CV_64FC1);
	avg_vec = cvCreateMat(nChannels, 1,  CV_64FC1);
	vector = cvCreateMat(nChannels, 1,  CV_64FC1);
	FrameOut = cvCreateImage(cvSize(Frame->width, Frame->height), IPL_DEPTH_8U, 1);

	// init window
	/*cvNamedWindow("11", 0);*/
	cvNamedWindow("out", 0);

	printf("Initialization completed!\n");

	// start loop
	double max, min;
	int count;
	while ( Frame = cvQueryFrame(pCapture) ){
		FrameCopy = cvCloneImage(Frame);
		cvShowImage("11", FrameCopy);

		// interaction
		
		key = cvWaitKey(20);
		if(key=='q')break;


		// get background 
		if(key=='b'){
			// generate the set of vector
			for(int i = 0; i < nPixels; i++) vects[i]->data.ptr = (unsigned char *)&FrameCopy->imageData[i*SizePixel];
			
			cvCalcCovarMatrix((const void **)vects, nPixels, cov_mat, avg_vec, CV_COVAR_NORMAL | CV_COVAR_SCALE);
			cvInvert(cov_mat, cov_mat, CV_SVD);
			printf("covariance matrix computed !\n");
			printf("-----Covariance Matrix-----\n");
			printf("%.3f %.3f %.3f\n", cov_mat->data.db[0], cov_mat->data.db[1], cov_mat->data.db[2]);
			printf("%.3f %.3f %.3f\n", cov_mat->data.db[3], cov_mat->data.db[4], cov_mat->data.db[5]);
			printf("%.3f %.3f %.3f\n", cov_mat->data.db[6], cov_mat->data.db[7], cov_mat->data.db[8]);
			printf("-----Mean Vector-----\n");
			printf("%.3f %.3f %.3f\n", avg_vec->data.db[0], avg_vec->data.db[1], avg_vec->data.db[2]);
		}

		if(key=='c'){
			count = 0;
			max = 0;
			min = 255;
			
			/*cov_mat->data.db[0] = 1; cov_mat->data.db[1] = 0; cov_mat->data.db[2] = 0;*/
			/*cov_mat->data.db[3] = 0; cov_mat->data.db[4] = 1; cov_mat->data.db[5] = 0;*/
			/*cov_mat->data.db[6] = 0; cov_mat->data.db[7] = 0; cov_mat->data.db[8] = 1;*/

			// generate the output image
			/*avg_vec->data.db[0] = 255;*/
			/*avg_vec->data.db[1] = 255;*/
			/*avg_vec->data.db[2] = 255;*/

			for(int i = 0; i < nPixels; i++){
				vector->data.db[0] = (unsigned char)FrameCopy->imageData[i*SizePixel + 0];
				vector->data.db[1] = (unsigned char)FrameCopy->imageData[i*SizePixel + 1];
				vector->data.db[2] = (unsigned char)FrameCopy->imageData[i*SizePixel + 2];
				double v = cvMahalanobis(vector, avg_vec, cov_mat);

				/*if(i%2000 == 0) printf("%d %d %d\n", (unsigned char)FrameCopy->imageData[i*SizePixel + 0], (unsigned char)FrameCopy->imageData[i*SizePixel + 1], (unsigned char)FrameCopy->imageData[i*SizePixel + 2]);*/
				/*if(i%2000 == 0) printf("%.3f %.3f %.3f\n", vector->data.db[0], vector->data.db[1], vector->data.db[2]);*/

				// find max and min
				if (v > max) max = v;
				if (v < min) min = v;

				FrameOut->imageData[i] = (unsigned char)v ;
				if (FrameOut->imageData[i] < 10)count++;
			}

			//Vision_StretchContrast (FrameOut, FrameOut, 0);
			cvThreshold(FrameOut, FrameOut, 0, 255, CV_THRESH_BINARY+CV_THRESH_OTSU);
			
			DetectBlob (FrameCopy); 

			printf ( "mean: %.3f\n", cvAvg(FrameOut, NULL ).val[0]);

			printf ( "Count: %d\n", count);
			printf ( "Max: %.3f\n", max);
			printf ( "Min: %.3f\n", min);
			
			/*cvShowImage("out", FrameOut);*/
		}

		cvReleaseImage(&FrameCopy);
		
	}

	// release memory
	/*cvDestroyWindow("11");*/
	cvDestroyWindow("out");
	cvReleaseImage(&Frame);
	cvReleaseImage(&FrameOut);
	cvReleaseImage(&FrameCopy);
	cvReleaseMat(&cov_mat);
	cvReleaseMat(&avg_vec);

	for(int i = 0; i < nPixels; i++){
	//	cvReleaseMat(&&vects[i]);
	}

	free(vects);

	return 1;
}


int Vision_StretchContrast(const IplImage *img_in, IplImage *img_out, int methrod){
	unsigned char r_max = 0;
	unsigned char r_min = 0;
	int dep = img_in->depth;
	int nPixels = img_in->width * img_in->height;
	int hist_size = pow(2,dep);
	int *hist = (int *)malloc(hist_size * sizeof(int));

	Vision_Histogram2D(img_in, hist_size, hist);

	/* Looking for r_max and r_min */
	int n_counter = 0;
	int min_thr = nPixels * 0.15;
	int max_thr = nPixels - min_thr;
	for(int i = 1; i < hist_size; i++){
		n_counter += hist[i];	
		if(n_counter < min_thr) r_min = (unsigned char)i; 
		if(n_counter < max_thr) r_max = (unsigned char)i; 
	}

	
	/* Stretching */
	float hist_size_f = (float)hist_size;
	float r_size = (unsigned char)(r_max - r_min);
	float r_k = hist_size_f/r_size;
	int img_h = img_in->height;
	int img_w = img_in->width;
	int img_ws = img_in->widthStep;
	unsigned char pixelvalue;
	float pixelvalueout;
	for(int i = 0; i < img_h; i += 1){
		for ( int j = 0; j < img_w; j += 1 ) {
			pixelvalue = (unsigned char)img_in->imageData[i*img_ws + j];
			pixelvalueout = r_k *((float)pixelvalue - (float)r_min);
			if (pixelvalueout >= hist_size) img_out->imageData[i*img_ws + j] = (unsigned char)(hist_size - 1); 
			else if(pixelvalueout < 0) img_out->imageData[i*img_ws + j] = 0; 
			else img_out->imageData[i*img_ws + j] = (unsigned char)pixelvalueout; 
		}
	}
	/* End function */
	free(hist);
	return 1;
}

int Vision_Histogram2D(const IplImage * img_in, int hist_size, int *hist_value){
	int img_h = img_in->height;
	int img_w = img_in->width;
	int img_ws = img_in->widthStep;

	/* Initial hist_value */
	//memset(hist_value, 0, hist_size);

	/* Initialize Histogram */
	for ( int i = 0; i < hist_size; i += 1 ) {
		hist_value[i] = 0;
	}
	/* Find hist value */
	char pixelvalue;
	for(int i = 0; i < img_h; i += 1){
		for ( int j = 0; j < img_w; j += 1 ) {
			pixelvalue = img_in->imageData[i*img_ws + j];
			hist_value[(unsigned char)pixelvalue]++;
		}
	}
	return 1;
}

