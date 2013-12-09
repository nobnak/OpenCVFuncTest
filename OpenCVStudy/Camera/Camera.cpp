
#include "stdafx.h"


#ifdef WIN32
int _tmain(int argc, _TCHAR* argv[])
#else
int main(int argc, char * argv[])
#endif
{
	cv::ocl::DevicesInfo devices;
	cv::ocl::getOpenCLDevices(devices, cv::ocl::CVCL_DEVICE_TYPE_CPU);

	cv::ocl::FarnebackOpticalFlow flow;
    flow.numLevels = 4;
    flow.pyrScale = 0.5;
    flow.fastPyramids = false;
    flow.winSize = 11;
    flow.numIters = 10;
    flow.polyN = 5;
    flow.polySigma = 1.1;

	cv::VideoCapture capture(0);
	cv::Mat src0, src1;
    cv::Mat flowxCpu, flowyCpu;
	cv::ocl::oclMat frame0, frame1;
	cv::ocl::oclMat small0, small1;
	cv::ocl::oclMat gray0, gray1;
	cv::ocl::oclMat flowx, flowy;
	cv::namedWindow("Capture");
	
	capture >> src0;
	cv::Size computeSize = src0.size();
	computeSize.width /= 2;
	computeSize.height /= 2;
    cv::Mat dst(computeSize, CV_8UC3);

	frame0.upload(src0);
	cv::ocl::resize(frame0, small0, computeSize);
	cv::ocl::cvtColor(small0, gray0, CV_RGB2GRAY);

	int counter = 0;
	while (cv::waitKey(1) != 'q') {
		bool even = (counter++ % 2) == 0;
		if (even) {
			capture >> src1;
			frame1.upload(src1);
			cv::ocl::resize(frame1, small1, computeSize);
			cv::ocl::cvtColor(small1, gray1, CV_RGB2GRAY);
			flow(gray0, gray1, flowx, flowy);
			flowx.download(flowxCpu);
            flowy.download(flowyCpu);
		} else {
			capture >> src0;
			frame0.upload(src0);
			cv::ocl::resize(frame0, small0, computeSize);
			cv::ocl::cvtColor(small0, gray0, CV_RGB2GRAY);
			flow(gray1, gray0, flowx, flowy);
			flowx.download(flowxCpu);
            flowy.download(flowyCpu);
		}
        
        drawColorField(flowxCpu, flowyCpu, dst);

		cv::imshow("Capture", dst);
	}

	flow.releaseMemory();
	cv::destroyAllWindows();

	return 0;
}

