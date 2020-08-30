#include "readPath.h"

/**
* @function: 获取cate_dir目录下的所有文件名
* @param: cate_dir - string类型
* @result：vector<string>类型
*/

vector<string> getFiles(const string &cate_dir,bool append=false)
{
	vector<string> files;//存放文件名  
	string tmp_cate_dir=cate_dir + "\\*";
#ifdef WIN32  
	struct __finddata64_t file;
	__int64 lf;
	//输入文件夹路径  
	if ((lf = _findfirst64(tmp_cate_dir.c_str(), &file)) == -1) {
		cout << cate_dir << " not found!!!" << endl;
	}
	else {
		while (_findnext64(lf, &file) == 0) {
			//输出文件名  
			//cout<<file.name<<endl;  
			if (strcmp(file.name, ".") == 0 || strcmp(file.name, "..") == 0)
				continue;
			if(append) files.push_back(cate_dir+"/"+file.name);
			else files.push_back(file.name);
		}
	}
	_findclose(lf);
#endif  

#ifdef linux  
	DIR *dir;
	struct dirent *ptr;
	char base[1000];

	if ((dir = opendir(cate_dir.c_str())) == NULL)
	{
		perror("Open dir error...");
		exit(1);
	}

	while ((ptr = readdir(dir)) != NULL)
	{
		if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)    ///current dir OR parrent dir  
			continue;
		else if (ptr->d_type == 8)    ///file  
			//printf("d_name:%s/%s\n",basePath,ptr->d_name);  
			files.push_back(ptr->d_name);
		else if (ptr->d_type == 10)    ///link file  
			//printf("d_name:%s/%s\n",basePath,ptr->d_name);  
			continue;
		else if (ptr->d_type == 4)    ///dir  
		{
			files.push_back(ptr->d_name);
			/*
			memset(base,'\0',sizeof(base));
			strcpy(base,basePath);
			strcat(base,"/");
			strcat(base,ptr->d_nSame);
			readFileList(base);
			*/
		}
	}
	closedir(dir);
#endif  

	//排序，按从小到大排序  
	sort(files.begin(), files.end());
	return files;
}


void split_train_test(const string &folder, double split_rate = 0.3)
{
	ofstream train("video_train.txt");
	ofstream test("video_test.txt");

	vector<string> filePaths = getFiles(folder, true);

	for (int i = 0; i < filePaths.size(); ++i)
	{
		string subFolder = filePaths[i];
		vector<string> subFilePaths = getFiles(subFolder, true);
		for (auto &s : subFilePaths)
		{
			s += " " + to_string(i);
		}

		int max = subFilePaths.size();
		vector<bool> visited(max, false);
		int num_test = max*split_rate;
		while (num_test--)
		{
			int id = rand() % max;
			visited[id] = true;
		}
		for (int i = 0; i < visited.size(); ++i)
		{
			if (visited[i]) test << subFilePaths[i] << endl;
			else train << subFilePaths[i] << endl;
		}
	}
	train.close();
	test.close();
}

void shuffle(const string &fileList_in, const string &out)
{
	ifstream in(fileList_in);
	vector<string> path_in;
	int count = 0;
	while (in)
	{
		string buf;
		if (getline(in, buf))
		{
			path_in.push_back(buf);
			count++;
		}
	}
	in.close();

	vector<int> index(count, 0);
	vector<int> seed_tmp(count, 0);
	vector<string> path_out(count);
	for (int i = 0; i != count; ++i) index[i] = i;
	for(int i = 0; i != count; ++i)
	{
		int seed = rand() % (count - i);
		cout << seed << endl;
		seed_tmp[i] = index[seed];
		path_out[i] = path_in[index[seed]];
		index[seed] = index[count - i - 1];
	}
	sort(seed_tmp.begin(), seed_tmp.end());
	ofstream out_shuffled(out);
	for(auto p:path_out)
	{
		out_shuffled << p << endl;
	}
	out_shuffled.close();
}

int readImageFile(const string &imageFile, vector<string> &pathOfImage)
{
	string buf;
	int imageCount = 0;

	ifstream img_list(imageFile);

	if (img_list)
	{
		cout << "file is : " << imageFile << endl;

	}
	else
	{
		cout << "cannot open the imagelist file." << endl;
	}

	while (img_list)
	{
		if (getline(img_list, buf))
		{
			pathOfImage.push_back(buf);
			imageCount++;
		}
	}
	img_list.close();
	cout << imageCount << " things have been read" << endl;

	return imageCount;
}

string getFileName(const string &str)
{
	int m = str.find_last_of('/');
	int n = str.find_last_of('.');

	return str.substr(m + 1, n - m - 1);
}

void shuffle_2(const string &fileList_in, const string &out)
{
	ifstream in(fileList_in);
	vector<string> path_in;
	int count = 0;
	while (in)
	{
		string buf;
		if (getline(in, buf))
		{
			path_in.push_back(buf);
			count++;
		}
	}
	in.close();

	random_shuffle(path_in.begin(), path_in.end());

	ofstream out_shuffled(out);
	for (auto p : path_in)
	{
		out_shuffled << p << endl;
	}
	out_shuffled.close();
}