#include "imageProcess_threads.h"

#include <iostream>

using namespace std;

int main()
{
	string folder_in = "tt_in";
	processImageThreads(folder_in, "tt", 2);
	return 0;
}