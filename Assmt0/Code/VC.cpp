#include "opencv2/opencv.hpp"
#include <string>
#include <vector>

using namespace cv;

int main(int, char**)
{
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    Mat Webcam;
    namedWindow("Webcam Display",1);
    for(int i=0;;i++)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera
        //cvtColor(frame, Webcam Display, COLOR_BGR2GRAY);
        //GaussianBlur(Webcam Display, Webcam Display, Size(7,7), 1.5, 1.5);
        //Canny(Webcam Display, Webcam Display, 0, 30, 3);
        imshow("Webcam Display", frame);
    	std::string frameName = "./Captured/";
    	frameName+=std::to_string(i);
    	frameName+=".jpg";
    	std::cout<<frameName;
    	std::vector<int> compression_params;
    	compression_params.push_back(IMWRITE_JPEG_QUALITY);
    	compression_params.push_back(100);
	    imwrite(frameName, frame, compression_params);
        if(waitKey(30) >= 0) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
