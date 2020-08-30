#ifndef _READPATH_H
#define _READPATH_H

#include <iostream>  
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>

#ifdef linux  
#include <unistd.h>  
#include <dirent.h>  
#include <sys/types.h>  
#include <sys/stat.h> 

#endif  
#ifdef WIN32  
#include <direct.h>  
#include <io.h>  
#endif  

using namespace std;

vector<string> getFiles(const string &cate_dir, bool append);
void split_train_test(const string &folder, double split_rate);
void shuffle(const string &fileList_in, const string &out);

int readImageFile(const string &imageFile, vector<string> &pathOfImage);
string getFileName(const string &str);
void shuffle_2(const string &fileList_in, const string &out);

#endif