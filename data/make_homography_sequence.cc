#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>
#include <random>
#include <dirent.h> 

#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

#include "opencv2/features2d.hpp"


using std::cout;
using std::endl;

enum HomographyMethod
{
	HOMOGRAPHY_FEATURE = 0,	// feature-based matching using ORB/AKAZE and cv::findHomography()
	HOMOGRAPHY_PIXEL,		// direct pixel-based alignment using cv::findTransformECC()
	HOMOGRAPHY_HYBRID		// first perform feature matching, then refine it with pixel method
};

HomographyMethod homography_method = HOMOGRAPHY_PIXEL;

uint32_t dir_count = 0;	// total number of subdirs with files processed
uint32_t img_count = 0;	// total number of frames been processed
uint32_t img_ahead = 1;	// number of images ahead to compare against

std::string   dataset_out_path;	  // path to save the rescaled 8-bit mono images to
std::ofstream labels_file;		  // path to save the homography displacements to

bool display_visualization = false;  // when true, display a window visualizing the results


// find homography using feature matching
bool find_homography_features( cv::Mat& img1, cv::Mat& img2, cv::Mat& H, 
						 std::vector<cv::KeyPoint>& img1_keypoints, 
						 std::vector<cv::KeyPoint>& img2_keypoints )
{
	/*
	 * create feature detector
	 */
#ifdef USE_ORB
	cv::Ptr<cv::ORB> feature_detector = cv::ORB::create(4096);
#else
	const float akaze_threshold = 0.0001;
	const int   akaze_octaves   = 4;

	cv::Ptr<cv::AKAZE> feature_detector = cv::AKAZE::create( cv::AKAZE::DESCRIPTOR_MLDB, 0, 3,
												  akaze_threshold, akaze_octaves, akaze_octaves );
#endif

	if( !feature_detector )
	{
		cout << "failed to initialize feature detector" << endl;
		return 0;
	}


	/*
	 * create feature matcher
	 */
	cv::Ptr<cv::DescriptorMatcher> feature_matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

	if( !feature_matcher )
	{
		cout << "failed to initialize feature matcher" << endl;
		return 0;
	}


	/*
	 * detect keypoints
	 */
	std::vector<cv::KeyPoint> img1_detected_keypoints;
	std::vector<cv::KeyPoint> img2_detected_keypoints;

	feature_detector->detect(img1, img1_detected_keypoints);
	feature_detector->detect(img2, img2_detected_keypoints);

	cout << "  keypoints detected = " << img1_detected_keypoints.size() << endl;
	cout << "  keypoints detected = " << img2_detected_keypoints.size() << endl;


	/*
	 * extract descriptors
	 */
	cv::Mat img1_descriptors;
	cv::Mat img2_descriptors;

	feature_detector->compute(img1, img1_detected_keypoints, img1_descriptors);
	feature_detector->compute(img2, img2_detected_keypoints, img2_descriptors);

	//cout << "frame # " << img1_index << "  descriptors dimension = " << img1_descriptors.rows << "x" << img1_descriptors.cols << endl;
	//cout << "frame # " << img2_index << "  descriptors dimension = " << img2_descriptors.rows << "x" << img2_descriptors.cols << endl;


	/*
	 * find and filter matches
	 */
	std::vector< std::vector<cv::DMatch> > matches;

	feature_matcher->knnMatch(img1_descriptors, img2_descriptors, matches, 2);
	feature_matcher->clear();

	cout << "  matches = " << matches.size() << endl;

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
			img1_matched_keypoints.push_back( img1_detected_keypoints[matches[n][0].queryIdx] );
			img2_matched_keypoints.push_back( img2_detected_keypoints[matches[n][0].trainIdx] );
		//}
	}

	cout << "  filtered matches = " << img1_matched_keypoints.size() << endl;


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

	if( img1_matched_points.size() < 4 )
	{
		cout << "  not enough matches to compute homography" << endl;
		return false;
	}

	// TODO:  use RANSAC in a first step with a large reprojection error (to get a rough estimate) 
	//        in order to detect the spurious matchings then use LMEDS on the correct ones.
	// TODO:  try using findEssentialMatrix() or findFundamentalMatrix()
	//          https://github.com/vivekseth/blog-posts/tree/master/Difference-between-perspective-transform-homography-essential-matrix-fundamental-matrix#essential-matrix
	H = cv::findHomography( img1_matched_points, img2_matched_points,
					    /*cv::RANSAC*/ cv::LMEDS, 3.0, inlier_mask, 50000, 0.99999 );


	// unmask the inliers
	std::vector<cv::KeyPoint> img1_inlier_keypoints;
	std::vector<cv::KeyPoint> img2_inlier_keypoints;

	for( uint32_t n=0; n < img1_matched_keypoints.size(); n++ )
	{
		if( inlier_mask.at<uchar>(n) )
		{
			img1_keypoints.push_back( img1_matched_keypoints[n] );
			img2_keypoints.push_back( img2_matched_keypoints[n] );
		}
	}		

	cout << "  inlier matches = " << img1_inlier_keypoints.size() << endl;


	// check for valid homography
	if( H.empty() )
	{
		cout << "  failed to compute valid homography" << endl;
		return false;
	}

	return true;
}


// find homography using direct pixel-based alignment
bool find_homography_pixel( cv::Mat& img1, cv::Mat& img2, cv::Mat& H )
{
	const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 50, 0.001);
	
	try
	{
		const double cc = cv::findTransformECC(img1, img2, H, cv::MOTION_HOMOGRAPHY, criteria);

		if( cc == -1 )
		{
			cout << "  error, pixel correlation coefficient is:  " << cc << endl;
			cout << "  try the HOMOGRAPHY_HYBRID method to improve the initialization" << endl;
			
			return false;
		}
	}
	catch( cv::Exception e ) 
	{
		cout << "cv::findTransformECC exception: " << e.err << endl;
		return false;
	}

	return true;
}


// estimate homographies for the provided image files
int process_files( const std::vector<std::string>& img_files )
{
	/*
	 * make sure there are enough files to process
	 */
	if( img_files.size() < img_ahead + 1 )
		return 0;

	
	/*
	 * generate homography for all images in directory
	 */
	const uint32_t num_images = img_files.size();

	for( uint32_t img_index=0; img_index < num_images - img_ahead; img_index++ )
	{
		/*
		 * attempt to load this pair of images
		 */
		const uint32_t img1_index = img_index;
		const uint32_t img2_index = img_index + img_ahead;

		cv::Mat img1_org = cv::imread(img_files[img1_index], cv::IMREAD_GRAYSCALE);
		cv::Mat img2_org = cv::imread(img_files[img2_index], cv::IMREAD_GRAYSCALE);

		if( !img1_org.data )
		{
			cout << "failed to load image " << img_files[img1_index] << endl;
			break;
		}

		if( !img2_org.data )
		{
			cout << "failed to load image " << img_files[img2_index] << endl;
			break;
		}

		cout << "/////////////////////////////////////////////////////" << endl;
		cout << "// DIR #" << dir_count << ", FRAME #" << img1_index << endl;
		cout << "/////////////////////////////////////////////////////" << endl;


		/*
		 * rescale images
		 */
		const double scale_factor  = 0.3;
		const double inverse_scale = 1.0 / scale_factor;

		const cv::Size img_size_org(img1_org.cols, img1_org.rows);
		const cv::Size img_size_scaled(128, 128);

		cv::Mat img1;
		cv::Mat img2;

		if( scale_factor != 1.0 )
		{
			const int interpolation_method = scale_factor < 1.0 ? cv::INTER_AREA : cv::INTER_CUBIC;

			cv::resize(img1_org, img1, img_size_scaled/*cv::Size()*/, scale_factor, scale_factor, interpolation_method);
			cv::resize(img2_org, img2, img_size_scaled/*cv::Size()*/, scale_factor, scale_factor, interpolation_method);
		}
		else
		{
			img1 = img1_org.clone();
			img2 = img2_org.clone();
		}

		// save the rescaled image if desired
		char img1_file_out[512];
		char img2_file_out[512];

		sprintf(img1_file_out, "%09u.png", dir_count * 1000000 + img_count);
		sprintf(img2_file_out, "%09u.png", dir_count * 1000000 + img_count + 1);

		if( !dataset_out_path.empty() )
		{
			char img1_path_out[512];
			char img2_path_out[512];

			sprintf(img1_path_out, "%s/%s", dataset_out_path.c_str(), img1_file_out);
			sprintf(img2_path_out, "%s/%s", dataset_out_path.c_str(), img2_file_out);

			if( cv::imwrite(img1_path_out, img1) )
				cout << "dir #" << dir_count << ", frame #" << img1_index << " saved to " << img1_path_out << endl << endl;
			else
				cout << "dir #" << dir_count << ", frame #" << img1_index << " failed to save to " << img1_path_out << endl << endl;

			if( cv::imwrite(img2_path_out, img2) )
				cout << "dir #" << dir_count << ", frame #" << img2_index << " saved to " << img2_path_out << endl << endl;
			else
				cout << "dir #" << dir_count << ", frame #" << img2_index << " failed to save to " << img2_path_out << endl << endl;
		}
	

		/*
		 * feature-based alignment
		 */
		std::vector<cv::KeyPoint> img1_keypoints; 
		std::vector<cv::KeyPoint> img2_keypoints;

		cv::Mat H = cv::Mat::eye(3, 3, CV_32F);

		if( homography_method == HOMOGRAPHY_FEATURE || homography_method == HOMOGRAPHY_HYBRID )
		{
			if( !find_homography_features(img1, img2, H, img1_keypoints, img2_keypoints) )
			{
				cout << "dir #" << dir_count << ", frame #" << img1_index << "  failed to estimate feature-based homography" << endl;
				continue;
			}
		}


		/*
		 * direct pixel alignment
		 */
		if( homography_method == HOMOGRAPHY_PIXEL || homography_method == HOMOGRAPHY_HYBRID )
		{
			if( !find_homography_pixel(img1, img2, H) )
			{
				cout << "dir #" << dir_count << ", frame #" << img1_index << "  failed to estimate direct pixel homography" << endl;
				continue;
			}
		}


		cout << "H = " << endl << " " << H << endl << endl;


		/*
		 * calculate the displacement of the homography
		 */
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

		labels_file << img1_file_out << ";" << img2_file_out << ";";

		for( uint32_t i=0; i < pts2.size(); i++ )
		{
			const double max_displacement = 32.0;

			const double dx = pts2[i].x - pts1[i].x;
			const double dy = pts2[i].y - pts1[i].y;

			labels_file << dx << ",";
			labels_file << dy << ",";

			cout << "  " << pts1[i] << " -> " << pts2[i] << "   (dx = " << dx << ") (dy = " << dy << ")";

			if( std::fabs(dx) > max_displacement || std::fabs(dy) > max_displacement )
				cout << "   (max displacement of " << max_displacement << "px exceeded)";

			cout << endl;
		}

		labels_file << endl;
		cout << endl;


		/*
		 * render the GUI if requested
		 */
		if( display_visualization )
		{
			/*
			 * warp input image by the homography
			 */
			cv::Mat img1_warped;

	    		cv::warpPerspective(img1, img1_warped, H, img1.size());
			cv::cvtColor(img1_warped, img1_warped, cv::COLOR_GRAY2BGR);


			/*
			 * visualize results
			 */
			cv::Mat img1_overlay;
			cv::Mat img2_overlay;

			// draw the keypoints
			cv::drawKeypoints(img1, img1_keypoints, img1_overlay, cvScalar(0,255,0));
			cv::drawKeypoints(img2, img2_keypoints, img2_overlay, cvScalar(0,255,0));

			// rescale images to original size
			if( scale_factor != 1.0 )
			{
				const int interpolation_method = inverse_scale < 1.0 ? cv::INTER_AREA : cv::INTER_CUBIC;

				cv::resize(img1_overlay, img1_overlay, img_size_org/*cv::Size()*/, inverse_scale, inverse_scale, interpolation_method);
				cv::resize(img2_overlay, img2_overlay, img_size_org/*cv::Size()*/, inverse_scale, inverse_scale, interpolation_method);

				cv::resize(img1_warped, img1_warped, img_size_org/*cv::Size()*/, inverse_scale, inverse_scale, interpolation_method);
			}

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
			cv::waitKey(1);
		}

		// increment total frame count
		img_count++;
	}

	dir_count++;
}

// returns true if the directory should be filtered out
bool filter_dir( const char* name, const std::vector<std::string>& filters )
{
	if( !name )
		return true;

	if( strcmp(name, ".") == 0 || strcmp(name, "..") == 0 )
		return true;

	const uint32_t num_filters = filters.size();

	for( uint32_t n=0; n < num_filters; n++ )
	{
		if( strcmp(name, filters[n].c_str()) == 0 )
			return true;
	}

	return false;
}

// recursively transverse subdirectories except those filtered out
void process_dir( const std::string& path, const std::vector<std::string>& filters )
{
	cout << path << endl;

	// open this directory
	DIR* dir = NULL;
	struct dirent* dir_entry;
	dir = opendir(path.c_str());

	if( !dir )
		return;

	// keep track of subdirectories and files
	std::vector<std::string> subdirs;
	std::vector<std::string> files;

	// enumerate all entries
	while((dir_entry = readdir(dir)) != NULL) 
	{
		if( filter_dir(dir_entry->d_name, filters) )
			continue;

		const std::string qualified_path = path + "/" + std::string(dir_entry->d_name);

		if( dir_entry->d_type == DT_DIR )
			subdirs.push_back(qualified_path);
		else if( dir_entry->d_type == DT_REG )
			files.push_back(qualified_path);
		
			//process_dir(path + "/" + std::string(dir_entry->d_name), filters);
	}

	// close the directory
	closedir(dir);

	// sort subdirectories and filenames
	std::sort(subdirs.begin(), subdirs.end());
	std::sort(files.begin(), files.end());

	// process all files
	//for( uint32_t n=0; n < files.size(); n++ )
	//	cout << "   " << files[n] << endl;

	process_files(files);

	// process all subdirectories
	for( uint32_t n=0; n < subdirs.size(); n++ )
		process_dir(subdirs[n], filters);
}


// recursively transverse subdirectories until we find image files
void process_dir( const std::string& path )
{
	std::vector<std::string> filters;
	process_dir(path, filters);
}



void readme()
{
	cout << "Usage: ./make_homography_sequence <dataset_in_path> <scaled_dataset_out_path> <labels_out_path>" << endl;
}


int main( int argc, char **argv )
{
	cout << "OpenCV Version: " << CV_MAJOR_VERSION << ".";
	cout << CV_MINOR_VERSION << endl;

	if( argc < 4 )
	{
		readme();
		return -1;
	}

	/*
	 * parse command line arguments
	 */
	const std::string  dataset_path(argv[1]);
   /*const std::string*/dataset_out_path = argv[2];
	const std::string  labels_path(argv[3]);

	cout << "dataset input path:  " << dataset_path << endl;
	cout << "scaled dataset output path:  " << dataset_out_path << endl;
	cout << "training labels output path:  " << labels_path << endl;

	labels_file.open(labels_path);


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
	 * setup filters
	 */
	std::vector<std::string> filters;
	
	filters.push_back("image_1");		// only process the left camera (image_0)
	filters.push_back("calib.txt");
	filters.push_back("times.txt");


	/*
	 * process all directories
	 */
	process_dir(dataset_path, filters);
	
	cout << "Done processing" << endl;
	cout << "   " << dir_count << " directories" << endl;
	cout << "   " << img_count << " images" << endl;


	/*
	 * cleanup
	 */
	labels_file.close();

	return 0;
}

