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

#include "opencv2/features2d.hpp"

using std::cout;
using std::endl;


//using namespace cv;
namespace fs = std::experimental::filesystem;


void readme()
{
	cout << "Usage: ./make_homography_sequence <dataset_directory> <sequence> <int_patch_size> <int_max_jitter> <n_samples> <bool_show_plots>" << endl;
}


int main( int argc, char **argv )
{
	cout << "OpenCV Version: " << CV_MAJOR_VERSION << ".";
	cout << CV_MINOR_VERSION << endl;

	if( argc < 2 )
	{
		readme();
		return -1;
	}

	/*
	 * parse command line arguments
	 */
	const std::string dataset_path(argv[1]);

	cout << "data path:  " << dataset_path << endl;


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
	 * create ORB feature detector
	 */
	//cv::Ptr<cv::ORB> feature_detector = cv::ORB::create(4096);
	cv::Ptr<cv::AKAZE> feature_detector = cv::AKAZE::create( cv::AKAZE::DESCRIPTOR_MLDB, 0, 3,
												  0.0001f, 4, 4 );

	if( !feature_detector )
	{
		cout << "failed to initialize feature detector" << endl;
		return 0;
	}


	/*
	 * create feature matching engine
	 */
	cv::Ptr<cv::DescriptorMatcher> feature_matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

	if( !feature_matcher )
	{
		cout << "failed to initialize feature matcher" << endl;
		return 0;
	}


	/*
	 * generate homography for all images in directory
	 */
	uint32_t img_index = 0;	// current index of image being processed
	uint32_t img_ahead = 1;	// number of images ahead to compare against

	while(true)
	{
		/*
		 * attempt to load this pair of images
		 */
		const uint32_t img1_index = img_index;
		const uint32_t img2_index = img_index + img_ahead;

		char img1_path[512];
		char img2_path[512];

		sprintf(img1_path, dataset_path.c_str(), img1_index);
		sprintf(img2_path, dataset_path.c_str(), img2_index);

		cv::Mat img1 = cv::imread(img1_path/*, cv::IMREAD_GRAYSCALE*/);
		cv::Mat img2 = cv::imread(img2_path/*, cv::IMREAD_GRAYSCALE*/);

		if( !img1.data )
		{
			cout << "failed to load image " << img1_path << endl;
			break;
		}

		if( !img2.data )
		{
			cout << "failed to load image " << img1_path << endl;
			break;
		}

	
		/*
		 * detect keypoints
		 */
		std::vector<cv::KeyPoint> img1_keypoints;
		std::vector<cv::KeyPoint> img2_keypoints;

		feature_detector->detect(img1, img1_keypoints);
		feature_detector->detect(img2, img2_keypoints);

		cout << "frame # " << img1_index << "  keypoints detected = " << img1_keypoints.size() << endl;
		cout << "frame # " << img2_index << "  keypoints detected = " << img2_keypoints.size() << endl;


		/*
		 * extract descriptors
		 */
		cv::Mat img1_descriptors;
		cv::Mat img2_descriptors;

		feature_detector->compute(img1, img1_keypoints, img1_descriptors);
		feature_detector->compute(img2, img2_keypoints, img2_descriptors);

		//cout << "frame # " << img1_index << "  descriptors dimension = " << img1_descriptors.rows << "x" << img1_descriptors.cols << endl;
		//cout << "frame # " << img2_index << "  descriptors dimension = " << img2_descriptors.rows << "x" << img2_descriptors.cols << endl;


		/*
		 * find and filter matches
		 */
		std::vector< std::vector<cv::DMatch> > matches;

		feature_matcher->knnMatch(img1_descriptors, img2_descriptors, matches, 2);
		feature_matcher->clear();

		cout << "frame # " << img1_index << "  matches = " << matches.size() << endl;

#if 0
		// calculate the min/max distance between matched keypoints
		double max_distance = 0.0;
		double min_distance = 1000.0;

		for( uint32_t n=0; n < matches.size(); n++ )
		{
			const double distance = matches[n].distance;

			if( distance < min_distance )
				min_distance = distance;

			if( distance > max_distance )
				max_distance = distance;
		}

		// only accept matches whose distance is less than 3 * min_distance
		const double distance_threshold = min_distance * 3.0;

		for( uint32_t n=0; n < matches.size(); n++ )
		{
			if( matches[n].distance < distance_threshold )
				good_matches.push_back(matches[n]);
		}
#endif

		// filter by nearest-neighbour matching ratio
		std::vector<cv::KeyPoint> img1_matched_keypoints;
		std::vector<cv::KeyPoint> img2_matched_keypoints;
		
		for( uint32_t n=0; n < matches.size(); n++ )
		{
			//if( matches[n][0].distance < matches[n][1].distance * 0.8 )
			//{
				img1_matched_keypoints.push_back( img1_keypoints[matches[n][0].queryIdx] );
				img2_matched_keypoints.push_back( img2_keypoints[matches[n][0].trainIdx] );
			//}
		}

		cout << "frame # " << img1_index << "  filtered matches = " << img1_matched_keypoints.size() << endl;


		// convert to Point2f for cv::findHomography()
		std::vector<cv::Point2f> img1_matched_points;
		std::vector<cv::Point2f> img2_matched_points;

		for( uint32_t n=0; n < img1_matched_keypoints.size(); n++ )
		{
			img1_matched_points.push_back( img1_matched_keypoints[n].pt );
			img2_matched_points.push_back( img2_matched_keypoints[n].pt );
		}
		

		/*
		 * homography estimation
		 */
		cv::Mat inlier_mask;
		cv::Mat H;

		if( img1_matched_points.size() < 4 )
		{
			cout << "frame # " << "  not enough matches to compute homography" << endl;
			continue;
		}

		// TODO:  use RANSAC in a first step with a large reprojection error (to get a rough estimate) 
		//        in order to detect the spurious matchings then use LMEDS on the correct ones.
		H = cv::findHomography( img1_matched_points, img2_matched_points,
						    /*cv::RANSAC*/ cv::LMEDS, 3.0, inlier_mask, 50000, 0.99999 );


		// unmask the inliers
		std::vector<cv::KeyPoint> img1_inlier_keypoints;
		std::vector<cv::KeyPoint> img2_inlier_keypoints;

		for( uint32_t n=0; n < img1_matched_keypoints.size(); n++ )
		{
			if( inlier_mask.at<uchar>(n) )
			{
				img1_inlier_keypoints.push_back( img1_matched_keypoints[n] );
				img2_inlier_keypoints.push_back( img2_matched_keypoints[n] );
			}
		}		

		cout << "frame # " << img1_index << "  inlier matches = " << img1_inlier_keypoints.size() << endl;


		// check for valid homography
		if( H.empty() )
		{
			cout << "frame # " << "  failed to compute valid homography" << endl;
			continue;
		}

		cout << endl << "H = " << endl << " " << H << endl << endl;

		
		// test the homography transform
		std::vector<cv::Point2f> pts1;
		std::vector<cv::Point2f> pts2;

		pts1.resize(4);
		pts2.resize(4);

		pts1[0].x = 0.0f;       pts1[0].y = 0.0f;
		pts1[1].x = img1.cols;  pts1[1].y = 0.0f;
		pts1[2].x = img1.cols;  pts1[2].y = img1.rows;
		pts1[3].x = 0.0f;       pts1[3].y = img1.rows;

		cv::perspectiveTransform(pts1, pts2, H);

		cout << "Corner points transformed by H = " << endl;

		for( uint32_t i=0; i < pts2.size(); i++ )
			cout << "  " << pts1[i] << " -> " << pts2[i] << endl;

		cout << endl;


		// warp the input image by the homography
		cv::Mat img1_warped;

    		cv::warpPerspective(img1, img1_warped, H, img1.size());

		//cout << "input image  = " << img1.rows << "x" << img1.cols << endl;
		//cout << "warped image = " << img1_warped.rows << "x" << img1_warped.cols << endl;


		/*
		 * visualize results
		 */
		cv::Mat img1_overlay;
		cv::Mat img2_overlay;

		// draw the keypoints
		cv::drawKeypoints(img1, img1_inlier_keypoints, img1_overlay, cvScalar(0,255,0));
		cv::drawKeypoints(img2, img2_inlier_keypoints, img2_overlay, cvScalar(0,255,0));

		// apply some overlay text to the images
		char img_label[128];

		sprintf(img_label, "Frame #%u", img1_index);

		cv::putText(img1_overlay, img_label, cvPoint(10,40), 
    				  cv::FONT_HERSHEY_SIMPLEX, 1.5, cvScalar(0,175,255), 3/*, CV_AA*/);

		sprintf(img_label, "Frame #%u", img2_index);

		cv::putText(img2_overlay, img_label, cvPoint(10,40), 
    				  cv::FONT_HERSHEY_SIMPLEX, 1.5, cvScalar(0,175,255), 3/*, CV_AA*/);

		cv::putText(img1_warped, "Warped", cvPoint(10,40), 
    				  cv::FONT_HERSHEY_SIMPLEX, 1.5, cvScalar(0,175,255), 3/*, CV_AA*/);

		// compost overlays into one image
		cv::Mat img_overlay_composted;

		cv::hconcat(img1_overlay, img2_overlay, img_overlay_composted);
		cv::hconcat(img_overlay_composted, img1_warped, img_overlay_composted);
		
		// display the composted image
		cv::imshow("Keypoints", img_overlay_composted);
		cv::waitKey(/*1*/);

		/*
		 * extract descriptors
		 */
		//feature_detector.
		img_index++;
	}

#if 0
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
    		cv::warpPerspective(img1, img1_warp, homography.inv(), img1.size());
	
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
#endif

	return 0;
}

