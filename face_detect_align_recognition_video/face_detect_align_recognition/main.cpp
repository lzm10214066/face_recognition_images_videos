#include <iostream>
#include "recognition/CenterFace.h"
#include "detection/MTCNN.h"
#include "SimilarityTransform.h"
#include "video_face.h"
#include "videoProcess.h"

using namespace std;
using namespace cv;

int main()
{
	VideoFace video_facer;

	cout << "\n\nface detection......" << endl;
	double t = (double)getTickCount();

	vector<Mat> faces_aligned;
	string video_name = "t7";
	unordered_map<string, double> res=video_facer.video_face_detection_recognition("test/"+video_name+".mp4", faces_aligned);
	unordered_map<string, double> res = video_facer.video_face_detection_classification_realtime("test/" + video_name + ".mp4", faces_aligned,"video_face_out");

	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "Times passed in ms: " << t*1000 << endl;

	for (auto it=res.begin(); it != res.end();++it)
	{
	    cout << it->first << "  :  " << it->second << endl;
	}

///////////////////////////*result*/////////////////////////////////////////////////////////////////
	string faces_out = "video_face_out/" + video_name;
	if (_access(faces_out.c_str(), 0) == -1)
	{
		int flag = _mkdir(faces_out.c_str());
		if (flag == 0) cout << "make successfully" << endl;
		else cout << "make errorly" << endl;
		CV_Assert(flag == 0);
	}

	for (int i = 0; i < faces_aligned.size(); ++i)
	{
		string folder_out = faces_out+"/";
		string img_name = folder_out + to_string(i) + ".png";
		imwrite(img_name, faces_aligned[i]);
	}

	return 0;
}


//int main(int argc, char** argv)
//{
//	string video_folder = "G:/数据集/video_classify/data/电影-电影剪辑";
//	string faces_folder = "tt_out";
//	processVideos2FacesThreads(video_folder, faces_folder, 2);
//}


//int main()
//{
//	string img_folder = "faces";
//	ImageFace image_facer;
//	image_facer.compareImages(img_folder);
//
//	return 0;
//}
