#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>
#include <random>
#include <experimental/filesystem>

#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"


using std::cout;
using std::endl;

//using namespace cv;
namespace fs = std::experimental::filesystem;


cv::Mat computeHomography( const cv::Mat& R_1to2, const cv::Mat& tvec_1to2, const double d_inv, const cv::Mat& normal)
{
    cv::Mat homography = R_1to2 + d_inv * tvec_1to2*normal.t();
    return homography;
}

cv::Mat computeHomography( const cv::Mat& R1, const cv::Mat& tvec1, cv::Mat& R2, cv::Mat& tvec2,
					  const double d_inv, const cv::Mat& normal )
{
    cv::Mat homography = R2 * R1.t() + d_inv * (-R2 * R1.t() * tvec1 + tvec2) * normal.t();
    return homography;
}

void computeC2MC1( const cv::Mat& R1, const cv::Mat& tvec1, const cv::Mat& R2, const cv::Mat& tvec2,
                   cv::Mat& R_1to2, cv::Mat& tvec_1to2 )
{
	//c2Mc1 = c2Mo * oMc1 = c2Mo * c1Mo.inv()
	R_1to2 = R2 * R1.t();
	tvec_1to2 = R2 * (-R1.t()*tvec1) + tvec2;
}

void extractRotation( const cv::Mat& pose, cv::Mat& r )
{
	for( int y=0; y < 3; y++ )
		for( int x=0; x < 3; x++ )
			r.at<double>(y,x) = pose.at<double>(y,x);
}

void extractTranslation( const cv::Mat& pose, cv::Mat& t )
{
	for( int y=0; y < 3; y++ )
		t.at<double>(y,0) = pose.at<double>(y,3);
}

void readme()
{
	cout << "Usage: ./make_homography_data_kitti <dataset_directory> <sequence> <int_patch_size> <int_max_jitter> <n_samples> <bool_show_plots>" << endl;
}


int main( int argc, char **argv )
{
	cout << "OpenCV Version: " << CV_MAJOR_VERSION << ".";
	cout << CV_MINOR_VERSION << endl;

	if( argc < 3 )
	{
		readme();
		return -1;
	}

	/*
	 * parse command line arguments
	 */
	const std::string dataset_path(argv[1]);
	const std::string sequence(argv[2]);
	const std::string pose_path(dataset_path + "/poses/" + sequence + ".txt");

	cout << "data path:  " << dataset_path << endl;
	cout << "pose path:  " << pose_path << endl << endl;
	

	/*
	 * camera instrinsic calibration
	 * note:  sequences 0-2 of the dataset have the same calibration
	 *        add support for reading this from the calib.txt files
	 */
	cv::Mat instrinsic_mat = (cv::Mat_<double>(3,3) << 
			7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02,
			0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 
			0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00);

	cout << "camera intrinsic calibration = " << endl << " " << instrinsic_mat << endl << endl;


	/*
	 * open the camera pose file
	 */
	std::ifstream pose_file(pose_path, std::ifstream::in);
	
	if( !pose_file.is_open() )
	{
		cout << "failed to open pose file:  " << pose_path << endl;
		return -1;
	}


	/*
	 * parse the camera poses
	 */
	std::vector<cv::Mat> poses;

	for( std::string line; std::getline(pose_file, line); )
	{
		//cout << line << endl;

		std::stringstream parser;
		parser << line;

		cv::Mat cam_pose = cv::Mat::zeros(3, 4, CV_64FC1);

		for( int y=0; y < 3; y++ )
		{
			for( int x=0; x < 4; x++ )
			{
				double value = 0.0;
				parser >> value;

				if( (parser.rdstate() & std::ifstream::failbit) != 0 )
				{
					cout << "parser error" << endl;
					continue;
				}

				cam_pose.at<double>(y,x) = value;
			}
		}

		//cout << endl << "pose[" << poses.size() << "] = " << endl << " " << cam_pose << endl << endl;
		poses.push_back(cam_pose);

		//if( poses.size() >= 3 )
		//	break;	
	}

	pose_file.close();

	const uint32_t numPoses = poses.size();	
	cout << "parsed " << numPoses << " camera poses" << endl << endl;
	
	if( numPoses == 0 )
	{
		cout << "error:  zero camera poses were parsed" << endl;
		return -1;
	}

	
	/*
	 * process each frame
	 */	
	const uint32_t framesAhead = 2;	// the number of frames ahead to compute the homography

	std::vector<uint32_t> outlierFrames;
	const double outlierThreshold = 15.0;

	for( uint32_t n=0; n < numPoses-framesAhead; n++ )
	{
		cout << "-----------------------------------------" << endl;
		cout << "-- FRAME " << n << endl;
		cout << "-----------------------------------------" << endl;

		// extract rotation/translation from frame #1
		cv::Mat R1(3, 3, CV_64FC1);
		cv::Mat T1(3, 1, CV_64FC1);

		extractRotation(poses[n], R1);
		extractTranslation(poses[n], T1);

		cout << "R1 = " << endl << " " << R1 << endl << endl;
		cout << "T1 = " << endl << " " << T1 << endl << endl;

		// extract rotation/translation from frame #2
		cv::Mat R2(3, 3, CV_64FC1);
		cv::Mat T2(3, 1, CV_64FC1);

		extractRotation(poses[n+framesAhead], R2);
		extractTranslation(poses[n+framesAhead], T2);

		cout << "R2 = " << endl << " " << R2 << endl << endl;
		cout << "T2 = " << endl << " " << T2 << endl << endl;

		// compute camera displacement
		cv::Mat R_1to2;
		cv::Mat T_1to2;

		computeC2MC1(R1, T1, R2, T2, R_1to2, T_1to2);

		//cout << "R_1to2 rows: " << R_1to2.rows << " cols: " << R_1to2.cols << endl;
		//cout << "T_1to2 rows: " << T_1to2.rows << " cols: " << T_1to2.cols << endl << endl;

		cout << "R_1to2 = " << endl << " " << R_1to2 << endl << endl;
		cout << "T_1to2 = " << endl << " " << T_1to2 << endl << endl;

		// compute plane normal @ camera #1
		cv::Mat normal  = (cv::Mat_<double>(3,1) << 0, 0, 1);
    		cv::Mat normal1 = R1*normal;

		cout << "normal1 = " << endl << " " << normal1 << endl << endl;

		// compute plane distance to camera frame #1
		cv::Mat origin(3, 1, CV_64F, cv::Scalar(0));
		cv::Mat origin1 = R1 * origin + T1;

		cout << "origin1 = " << endl << " " << origin1 << endl << endl;

		const double d_inv1 = 1.0 / normal1.dot(origin1);
		cout << "d_inv1 = " << d_inv1 << endl << endl;

		if( std::fabs(d_inv1) >= outlierThreshold )
			outlierFrames.push_back(n);

		// compute homography from camera displacement
		cv::Mat homography_euclidean = computeHomography(R_1to2, T_1to2, d_inv1, normal1);
		cv::Mat homography = instrinsic_mat * homography_euclidean * instrinsic_mat.inv();

		//cout << "homography_euclidean = " << endl << " " << homography_euclidean << endl << endl;
		//cout << "homography = " << endl << " " << homography << endl << endl;

#if 1		
		homography_euclidean /= homography_euclidean.at<double>(2,2);
		homography /= homography.at<double>(2,2);
#endif
		//cout << "homography_euclidean = " << endl << " " << homography_euclidean << endl << endl;
		cout << "homography = " << endl << " " << homography << endl << endl;
		
		// same, but using absolute camera poses instead of camera displacement, just for check
#if 0
		cv::Mat homography_euclidean2 = computeHomography(R1, T1, R2, T2, d_inv1, normal1);
		cv::Mat homography2 = instrinsic_mat * homography_euclidean2 * instrinsic_mat.inv();

		cout << "homography_euclidean2 = " << endl << " " << homography_euclidean2 << endl << endl;
		cout << "homography2 = " << endl << " " << homography2 << endl << endl;
		
		homography_euclidean2 /= homography_euclidean2.at<double>(2,2);
		homography2 /= homography2.at<double>(2,2);
		
		cout << "homography_euclidean2 = " << endl << " " << homography_euclidean2 << endl << endl;
		cout << "homography2 = " << endl << " " << homography2 << endl << endl;
#endif

		// test the homography transform
		const int img_width  = 1241;
		const int img_height = 376;

		std::vector<cv::Point2f> pts1;
		std::vector<cv::Point2f> pts2;

		pts1.resize(4);
		pts2.resize(4);

		pts1[0].x = 0.0f;       pts1[0].y = 0.0f;
		pts1[1].x = img_width;  pts1[1].y = 0.0f;
		pts1[2].x = img_width;  pts1[2].y = img_height;
		pts1[3].x = 0.0f;       pts1[3].y = img_height;

		cv::perspectiveTransform(pts1, pts2, homography);

		cout << "Corner points transformed by homography = " << endl;

		for( uint32_t i=0; i < pts2.size(); i++ )
			cout << "  " << pts1[i] << " -> " << pts2[i] << endl;

		cout << endl;


		// load the images for visualization
		char img1_path[512];
		char img2_path[512];

		sprintf(img1_path, "%s/sequences/%s/image_0/%06u.png", dataset_path.c_str(), sequence.c_str(), n);
		sprintf(img2_path, "%s/sequences/%s/image_0/%06u.png", dataset_path.c_str(), sequence.c_str(), n+framesAhead);
		
		cv::Mat img1 = cv::imread(img1_path);
		cv::Mat img2 = cv::imread(img2_path);

		if( !img1.data )
		{
			cout << "failed to load image " << img1_path << endl;
			continue;
		}

		if( !img2.data )
		{
			cout << "failed to load image " << img1_path << endl;
			continue;
		}

		// warp the input image by the homography
		cv::Mat img1_warp;
    		cv::warpPerspective(img1, img1_warp, homography, img1.size());
	
		// apply some overlay text to the images
		char img_label[128];

		sprintf(img_label, "Frame #%u", n);

		cv::putText(img1, img_label, cvPoint(10,40), 
    				  cv::FONT_HERSHEY_SIMPLEX, 1.5, cvScalar(0,175,255), 3/*, CV_AA*/);

		sprintf(img_label, "Frame #%u", n+framesAhead);

		cv::putText(img2, img_label, cvPoint(10,40), 
    				  cv::FONT_HERSHEY_SIMPLEX, 1.5, cvScalar(0,175,255), 3/*, CV_AA*/);

		cv::putText(img1_warp, "Warped", cvPoint(10,40), 
    				  cv::FONT_HERSHEY_SIMPLEX, 1.5, cvScalar(0,175,255), 3/*, CV_AA*/);

		// display the warped images
		cv::Mat img_compare;

		cv::hconcat(img1, img2, img_compare);
		cv::hconcat(img_compare, img1_warp, img_compare);
		cv::imshow("Comparision of camera image warped by homography", img_compare);
		cv::waitKey(/*1*/);
	}


	// print out the outlier frames
	cout << "number of outlier frames:   " << outlierFrames.size() << endl;
	cout << "indices of outlier frames:  ";

	for( uint32_t n=0; n < outlierFrames.size(); n++ )
		cout << outlierFrames[n] << " ";

	cout << endl;

	return 0;
}

