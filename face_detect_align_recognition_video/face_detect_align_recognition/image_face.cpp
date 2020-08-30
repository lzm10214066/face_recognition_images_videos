#include "image_face.h"
#include <unordered_map>


namespace ImageFace {

	ImageFace::ImageFace() : dist_threshold(0.8),th_person(0.5)
	{
		vector<string> mtcnn_model_file = {
			"model/det1.prototxt",
			"model/det2.prototxt",
			"model/det3.prototxt"
		};

		vector<string> mtcnn_trained_file = {
			"model/det1.caffemodel",
			"model/det2.caffemodel",
			"model/det3.caffemodel"
		};

		string centerFace_model_file = "model/face_deploy.prototxt";
		//string centerFace_trained_file = "model/face_model.caffemodel";
		string centerFace_trained_file = "model/face_iter_30000.caffemodel";
		//string centerFace_trained_file = "center_loss_ms/center_loss_ms.caffemodel";

		centerFace_model_file = "SphereFace/sphereface_deploy.prototxt";
		centerFace_trained_file = "SphereFace/sphereface_model_iter_38000.caffemodel";

		mtcnn = new MTCNN(mtcnn_model_file, mtcnn_trained_file);
		center_face = new CenterFace(centerFace_model_file, centerFace_trained_file);

		aligner = new AlignFace();

		string tem_file = "face_templates_10000_added.xml";
		readFaceTemplates(tem_file, faces_temps);

	}

	void ImageFace::rect_extend(Rect &rec, double r, int width, int height)
	{

		Rect pr = rec;
		int dx = cvRound(pr.width*(r - 1) / 2.0);
		int dy = cvRound(pr.height*(r - 1) / 2.0);

		int w = cvRound(pr.width*r);
		int h = cvRound(pr.height*r);

		int x = pr.x - dx;
		int y = pr.y - dy;

		if (x < 0) x = 0;
		if (y < 0) y = 0;
		if (x + w > width - 1) w = width - 1 - x;
		if (y + h > height - 1) h = height - 1 - y;

		pr.x = x;
		pr.y = y;
		pr.width = w;
		pr.height = h;

		rec = pr;
	}


	bool ImageFace::is_frontalFace(const Rect &box, const vector<Point2f> &landermarks)
	{
		Point2f left_eye = landermarks[0];
		Point2f right_eye = landermarks[1];
		Point2f nose = landermarks[2];
		Point2f mouth_left = landermarks[3];
		Point2f mouth_right = landermarks[4];

		/*double face_width = box.width;
		double r = abs(right_eye.x - left_eye.x)/face_width;
		if(r < 0.3) return false;*/
		double dist_l = nose.x - left_eye.x;
		double dist_r = right_eye.x - nose.x;
		double r_d = dist_l / dist_r;
		if (dist_l < 0 || dist_r < 0 || r_d>3 || r_d < 1 / 3) return false;

		dist_l = nose.x - mouth_left.x;
		dist_r = mouth_right.x - nose.x;
		r_d = dist_l / dist_r;
		if (dist_l < 0 || dist_r < 0 || r_d>3 || r_d < 1 / 3) return false;

		return true;
	}

	void ImageFace::calculateDistanceFace(const vector<Mat> &faces, vector<Pair_Dist> &dists)
	{
		int n_faces = faces.size();
		vector<vector<float>> feature_faces(n_faces);
		center_face->extractFeature(faces, feature_faces);
		/*for (int i = 0; i < faces.size(); ++i)
		{
			vector<float> res;
			center_face->extractFeature(faces[i], res);
			feature_faces[i] = res;
		}*/
		feature_faces_total = feature_faces;   //

		for (int i = 0; i != n_faces; ++i)
		{
			for (int j = i + 1; j != n_faces; ++j)
			{
				double norm_i = norm(feature_faces[i]);
				double norm_j = norm(feature_faces[j]);
				double cos_dist = Mat(feature_faces[i]).dot(Mat(feature_faces[j])) / (norm_i*norm_j);
				/*	Mat vi(feature_faces[i]);
					Mat vj(feature_faces[j]);
					Mat v_dist = vi - vj;
					double dist = norm(v_dist);*/
				if (cos_dist < dist_threshold) continue;
				dists.push_back(Pair_Dist(i, j, cos_dist));
			}
		}

		sort(dists.begin(), dists.end(), [](Pair_Dist &a, Pair_Dist &b) {return a.dist > b.dist; });
	}

	void ImageFace::face_align(const vector<Mat> &img_in, const vector<vector<Point2f>> &alignments, vector<Mat> &img_out)
	{
		for (int i = 0; i < img_in.size(); ++i)
		{
			Mat img = img_in[i];
			Mat img_aligned;
			//cout << "i: " << i << endl;
			aligner->similarity_transform(img, alignments[i], img_aligned);
			img_out.push_back(img_aligned);
		}
	}


	vector<vector<int>> ImageFace::findFaceClusters(const vector<Mat> &faces)
	{
		vector<Pair_Dist> dists;
		calculateDistanceFace(faces, dists);
		int num = faces.size();

		uset.resize(num);
		set_size.resize(num);
		for (int i = 0; i < num; ++i) { uset[i] = i; set_size[i] = 1; }
		for (int k = 0; k < dists.size(); ++k)
		{
			int i = dists[k].i;
			int j = dists[k].j;
			unionSet(i, j);
		}

		vector<int> counter(num, 0);
		vector<int> index_sorted(num, 0);
		for (int i = 0; i != num; ++i)
		{
			counter[uset[i]]++;
			index_sorted[i] = i;
		}

		vector<vector<int>> tmp(num);
		for (int i = 0; i < uset.size(); ++i)
		{
			tmp[uset[i]].push_back(i);
		}
		vector<vector<int>> res;

		for (int i = 0; i < num; ++i)
		{
			if (!tmp[i].empty())
				res.push_back(tmp[i]);
		}
		return res;
	}



	void ImageFace::compareImages(const string &folder_in)
	{
		vector<string> filePaths = getFiles(folder_in, true);
		vector<Mat> face_imgs;
		for (auto path : filePaths)
		{
			Mat image = imread(path);
			face_imgs.push_back(image);
		}

		int n_faces = face_imgs.size();
		vector<vector<float>> feature_faces(n_faces);
		center_face->extractFeature(face_imgs, feature_faces);

		for (int i = 0; i < filePaths.size(); ++i) cout << i << " : " << filePaths[i] << endl;
		printf("Distance matrix\n");
		printf("    ");
		for (int i = 0; i < filePaths.size(); ++i)
			printf("    %1d     ", i);
		cout << "\n";
		for (int i = 0; i != n_faces; ++i)
		{
			//printf("%1d  ", i);
			for (int j = i + 1; j != n_faces; ++j)
			{
				double norm_i = norm(feature_faces[i]);
				double norm_j = norm(feature_faces[j]);
				double cos_dist = Mat(feature_faces[i]).dot(Mat(feature_faces[j])) / (norm_i*norm_j);
				//printf("  %1.4f  ", cos_dist);
				/*	Mat vi(feature_faces[i]);
				Mat vj(feature_faces[j]);
				Mat v_dist = vi - vj;
				double dist = norm(v_dist);*/
				if (cos_dist > 0.5)
					cout << filePaths[i] << "  " << filePaths[j] << "  " << cos_dist << endl;
			}
			cout << "\n";
		}
	}

	void ImageFace::compareImages(const string &folder_1, const string &folder_2)
	{
		vector<string> filePaths_1 = getFiles(folder_1, true);
		vector<Mat> face_imgs_1;
		for (auto path : filePaths_1)
		{
			Mat image = imread(path);
			face_imgs_1.push_back(image);
		}

		int n_faces_1 = face_imgs_1.size();
		vector<vector<float>> feature_faces_1(n_faces_1);
		center_face->extractFeature(face_imgs_1, feature_faces_1);

		vector<string> filePaths_2 = getFiles(folder_2, true);
		vector<Mat> face_imgs_2;
		for (auto path : filePaths_2)
		{
			Mat image = imread(path);
			face_imgs_2.push_back(image);
		}

		int n_faces_2 = face_imgs_2.size();
		vector<vector<float>> feature_faces_2(n_faces_2);
		center_face->extractFeature(face_imgs_2, feature_faces_2);


		for (int i = 0; i != n_faces_1; ++i)
		{
			for (int j = 0; j != n_faces_2; ++j)
			{
				double norm_i = norm(feature_faces_1[i]);
				double norm_j = norm(feature_faces_2[j]);
				double cos_dist = Mat(feature_faces_1[i]).dot(Mat(feature_faces_2[j])) / (norm_i*norm_j);
				//printf("  %1.4f  ", cos_dist);
				/*	Mat vi(feature_faces[i]);
				Mat vj(feature_faces[j]);
				Mat v_dist = vi - vj;
				double dist = norm(v_dist);*/
				if (cos_dist > 0.8)
					cout << filePaths_1[i] << "  " << filePaths_2[j] << "  " << cos_dist << endl;
			}
			cout << "\n";
		}
	}

	void ImageFace::makeFaceTemplates(const string &face_folder, const string &file_out)
	{
		vector<string> filePaths = getFiles(face_folder, true);
		vector<FaceTemp> faces;

		for (int i = 0; i < filePaths.size(); ++i)
		{
			string img_path = filePaths[i];
			cout << i << "  :  " << img_path << endl;
			Mat image = imread(img_path);
			if (!image.data) continue;

			Mat face = mtcnn->image_face_detection_align(image);
			if (!face.data) continue;

			FaceTemp face_temp;
			face_temp.name = getFileName(img_path);
			face_temp.face = face;
			center_face->extractFeature(face, face_temp.feature);

			faces.push_back(face_temp);
		}

		int n_faces = faces.size();
		cout << n_faces << " faces detected" << endl;

		FileStorage fs(file_out, FileStorage::WRITE);
		fs << "n_faces" << n_faces;
		fs << "name_features" << "[";
		for (int i = 0; i < n_faces; ++i)
		{
			if ((i + 1) % 100 == 0) { cout << i + 1 << " faces >" << endl; }
			fs << faces[i].name << faces[i].feature;
		}
		fs << "]";
		cout << "\nmake face template done" << endl;
	}

	void ImageFace::readFaceTemplates(const string &temp_file, vector<FaceTemp> &face_temps)
	{
		FileStorage fs;
		fs.open(temp_file, FileStorage::READ);
		if (!fs.isOpened())
		{
			cerr << "Failed to open the feature file " << temp_file << endl;
			return;
		}
		else
		{
			cout << "load the feature file successfully!" << endl;
		}

		int n_faces = 0;
		fs["n_faces"] >> n_faces;
		FileNode name_features = fs["name_features"];
		face_temps.resize(n_faces);
		for (int i = 0; i < n_faces; ++i)
		{
			name_features[2 * i] >> face_temps[i].name;
			name_features[2 * i + 1] >> face_temps[i].feature;
		}
	}

	vector<FaceResult> ImageFace::faceRecognition(const string &img_path, const vector<FaceTemp> &face_temps, int k)
	{
		vector<FaceResult> res;
		Mat image = imread(img_path);
		if (!image.data) {
			cout << "can not read the image.." << endl;
			return res;
		}
		Mat face = mtcnn->image_face_detection_align(image);
		if (!face.data) {
			cout << "can not detect any face" << endl;
			return res;
		}

		vector<float> img_feature;
		center_face->extractFeature(face, img_feature);

		res.resize(face_temps.size());

		for (int i = 0; i != face_temps.size(); ++i)
		{
			vector<float> temp_feature = face_temps[i].feature;
			double norm_i = norm(temp_feature);
			double norm_img = norm(img_feature);
			double cos_dist = Mat(temp_feature).dot(Mat(img_feature)) / (norm_i*norm_img);
			/*	Mat vi(feature_faces[i]);
			Mat vj(feature_faces[j]);
			Mat v_dist = vi - vj;
			double dist = norm(v_dist);*/
			res[i].similarity = cos_dist;
			res[i].name = face_temps[i].name;
		}

		sort(res.begin(), res.end(), [](FaceResult &a, FaceResult &b) {return a.similarity > b.similarity; });
		vector<FaceResult> res_th;
		for (int i = 0; i < k; ++i)
		{
			if (res[i].similarity < th_person)
				break;
			res_th.push_back(res[i]);
		}
		return res_th;
	}

	vector<FaceResult> ImageFace::faceRecognition(const Mat &face, const vector<FaceTemp> &face_temps, int k)
	{
		vector<FaceResult> res;
		if (!face.data) {
			cout << "can not detect any face" << endl;
			return res;
		}

		vector<float> img_feature;
		center_face->extractFeature(face, img_feature);

		res.resize(face_temps.size());

		for (int i = 0; i != face_temps.size(); ++i)
		{
			vector<float> temp_feature = face_temps[i].feature;
			double norm_i = norm(temp_feature);
			double norm_img = norm(img_feature);
			double cos_dist = Mat(temp_feature).dot(Mat(img_feature)) / (norm_i*norm_img);
			/*	Mat vi(feature_faces[i]);
			Mat vj(feature_faces[j]);
			Mat v_dist = vi - vj;
			double dist = norm(v_dist);*/
			res[i].similarity = cos_dist;
			res[i].name = face_temps[i].name;
		}

		sort(res.begin(), res.end(), [](FaceResult &a, FaceResult &b) {return a.similarity > b.similarity; });
		vector<FaceResult> res_th;
		for (int i = 0; i < k; ++i)
		{
			if (res[i].similarity < th_person)
				break;
			res_th.push_back(res[i]);
		}
		return res_th;
	}


	void ImageFace::addPersonsToTemplate(const string &temp_file, const string &person_folder)
	{
		vector<string> img_path;
		if (string::npos != person_folder.find_last_of('.'))
		{
			img_path.push_back(person_folder);
		}
		else
		{
			img_path = getFiles(person_folder, true);
		}
		vector<FaceTemp> faces_temps;
		readFaceTemplates(temp_file, faces_temps);

		for (auto p : img_path)
		{
			Mat image = imread(p);
			if (!image.data) continue;

			imshow("add", image);
			waitKey(20);

			Mat face = mtcnn->image_face_detection_align(image);
			if (!face.data) continue;

			FaceTemp face_temp;
			face_temp.name = getFileName(p);
			face_temp.face = face;
			center_face->extractFeature(face, face_temp.feature);

			faces_temps.push_back(face_temp);
		}

		string file_out = getFileName(temp_file) + "_added.xml";
		FileStorage fs(file_out, FileStorage::WRITE);
		int n_faces = faces_temps.size();
		fs << "n_faces" << n_faces;
		fs << "name_features" << "[";
		for (int i = 0; i < n_faces; ++i)
		{
			if ((i + 1) % 100 == 0) { cout << i + 1 << " faces >" << endl; }
			fs << faces_temps[i].name << faces_temps[i].feature;
		}
		fs << "]";
		cout << "\nmake new face template done" << endl;
	}

	void ImageFace::replacePersonsToTemplate(const string &temp_file, const string &person_folder)
	{
		vector<string> img_path;
		if (string::npos != person_folder.find_last_of('.'))
		{
			img_path.push_back(person_folder);
		}
		else
		{
			img_path = getFiles(person_folder, true);
		}
		vector<FaceTemp> faces_temps;
		readFaceTemplates(temp_file, faces_temps);
		if (faces_temps.empty()) return;

		unordered_map<string, vector<float>> name_feature;
		for (auto i : faces_temps)
		{
			name_feature[i.name] = i.feature;
		}

		//vector<FaceTemp> faces_temps_add;
		for (auto p : img_path)
		{
			Mat image = imread(p);
			if (!image.data) continue;

			imshow("add", image);
			waitKey(20);

			Mat face = mtcnn->image_face_detection_align(image);
			if (!face.data) continue;

			FaceTemp face_temp;
			face_temp.name = getFileName(p);
			face_temp.face = face;
			center_face->extractFeature(face, face_temp.feature);

			//faces_temps_add.push_back(face_temp);
			name_feature[face_temp.name] = face_temp.feature;
		}

		string file_out = getFileName(temp_file) + "_added.xml";
		FileStorage fs(file_out, FileStorage::WRITE);
		int n_faces = name_feature.size();
		fs << "n_faces" << n_faces;
		fs << "name_features" << "[";
		for (auto it = name_feature.begin(); it != name_feature.end(); ++it)
		{
			fs << it->first << it->second;
		}
		fs << "]";
		cout << "\nmake new face template done" << endl;
	}

	vector<vector<FaceResult>> ImageFace::faceRecognition_folder(const string &persons_folder, int k)
	{
		vector<string> img_paths;
		vector<vector<FaceResult>> res;
		img_paths = getFiles(persons_folder, true);
		for (auto p : img_paths)
		{
			cout << p << endl;
			vector<FaceResult> temp = faceRecognition(p, faces_temps, k);
			res.push_back(temp);
		}
		return res;
	}

	vector<vector<FaceResult>> ImageFace::faceRecognition_faces(const vector<Mat> &faces, int k)
	{
		vector<vector<FaceResult>> res;
		for (auto &f : faces)
		{
			vector<FaceResult> temp = faceRecognition(f, faces_temps, k);
			res.push_back(temp);
		}
		return res;
	}
}