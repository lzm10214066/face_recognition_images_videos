
#include "imageProcess_threads.h"
#include "preProcessImage.h"

void processImage_thread(const vector<string> &image_paths, const int s, const int t, const string &dst_classFolder)
{
	ImagePreProcess imagePre;
	for (int i = s; i < s + t; ++i)
	{
		string image_path = image_paths[i];
		cout << "i: " << i << "   " << image_path << endl;

		Mat img = imread(image_path);
		vector<Mat> imgs_add;
		try
		{
			Mat img_color;
			int r = rand() % 180;
			imagePre.colorOverlay(img, img_color, r, 150, 0);
			imgs_add.push_back(img_color);

			Mat img_noise = img.clone();
			imagePre.addGaussianNoise(img_noise, 0, 8);
			imgs_add.push_back(img_noise);

			Mat img_blur = img.clone();
			medianBlur(img_blur, img_blur, 3);
			imgs_add.push_back(img_blur);

			Mat img_resize = img.clone();
			Mat t1,t2;
			//imagePre.resizeBlur(img_resize, t1, 1.5);
			//imgs_add.push_back(t1);
			imagePre.resizeBlur(img_resize, t2, 2);
			imgs_add.push_back(t2);

			Mat s1, s2, s3, s4;
			imagePre.addShadow(img, s1, 0);
			imgs_add.push_back(s1);
			imagePre.addShadow(img, s2, 1);
			imgs_add.push_back(s2);
			imagePre.addShadow(img, s3, 2);
			imgs_add.push_back(s3);
			imagePre.addShadow(img, s4, 3);
			imgs_add.push_back(s4);

		}
		catch (exception &e)
		{
			cout << img.size() << endl;
			cout << "exception: "<<image_path << endl;
			waitKey(0);
		}
		

		int a = image_path.find_last_of('.');
		int b = image_path.find_last_of('/', a - 1);
		int c = image_path.find_last_of('/', b - 1);

		string className = image_path.substr(c + 1, b - c - 1);
		for (int i = 0; i < imgs_add.size(); ++i)
		{
			string dst_name = dst_classFolder + "/" + className + "/" + image_path.substr(b + 1, a - b - 1) + "_"+to_string(i)+".png";
			cv::imwrite(dst_name, imgs_add[i]);
		}
	
	}
}

void processImageThreads(const string &folder_in, const string &folder_out, const int num_thread)
{
	vector<string> filePaths = getFiles(folder_in, true);
	vector<string> classNames = getFiles(folder_in, false);

	vector<string> image_paths;
	for (int i = 0; i<filePaths.size(); ++i)
	{
		string dst_classFolder = folder_out + "/" + classNames[i];
		if (_access(dst_classFolder.c_str(), 0) == -1)
		{
			cout << dst_classFolder << " is not existing" << endl;
			int flag = _mkdir(dst_classFolder.c_str());
			if (flag == 0) cout << "make successfully" << endl;
			else cout << "make errorly" << endl;
			CV_Assert(flag == 0);
		}

		string class_path = filePaths[i];
		vector<string> tmp = getFiles(class_path, true);
		image_paths.insert(image_paths.end(), tmp.begin(), tmp.end());
	}
	int total_images = image_paths.size();
	int images_per_thread = (int)ceil((double)total_images / num_thread);
	vector<thread> threads;

	int s = 0;
	for (int i = 0; i < num_thread; ++i)
	{

		int t = min(images_per_thread, total_images - s);
		threads.push_back(thread(processImage_thread, image_paths, s, t, folder_out));
		s += t;
	}

	for (int i = 0; i<threads.size(); i++)
		threads[i].join();

}

