#pragma once

#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  

#include <iostream>  
#include <string>
#include <fstream>
#include <thread>

#include "readPath.h"


using namespace std;
using namespace cv;

void processImageThreads(const string &folder_in,
	const string &folder_out, const int num_thread);
