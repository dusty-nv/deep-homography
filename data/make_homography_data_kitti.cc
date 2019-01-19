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

#include "image_tools.h"


using std::cout;
using std::endl;

//using namespace cv;
namespace fs = std::experimental::filesystem;


void readme()
{
	cout << "Usage: ./make_homography_data_kitti <dataset_directory> <int_patch_size> <int_max_jitter> <n_samples> <bool_show_plots>" << endl;
}


int main(int argc, char **argv )
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
	const std::string pose_path(dataset_path + "/poses/00.txt");

	cout << "data path:  " << dataset_path << endl;
	cout << "pose path:  " << pose_path << endl;
	

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
		cout << line << endl;

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

				cout << "pose(" << y << "," << x << ") = " << cam_pose.at<double>(y,x) << endl;
			}
		}

		poses.push_back(cam_pose);	
	}

	cout << "parsed " << poses.size() << " camera poses" << endl;
	pose_file.close();

	

	

#if 0
  std::string dir_name(argv[1]);
  int patch_size = atoi(argv[2]);
  int max_jitter = atoi(argv[3]);
  int n_samples = atoi(argv[4]);
  
  cout << "OpenCV Version: " << CV_MAJOR_VERSION << ".";
  cout << CV_MINOR_VERSION << endl;
  
  bool show_plots = false;
  if (argc == 6){
    show_plots = (bool)atoi(argv[5]);
  } else if (argc > 6 || argc < 5){
    readme(); return -1;
  } 
 
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 gen(seed);
 
  int cnt = 0; 
  char f_roi_orig[50];
  char f_roi_warp[50];
  char f_number[9];
  std::ofstream f_labels("../label_file.txt");
  
  bool had_enough = false;
  while (!had_enough){
    for (auto const& f_it: fs::directory_iterator(dir_name)){
      cout << cnt << endl;
       
      // reading the file
      std::string img_file = f_it.path().string();
      //cout << img_file << endl;
      Mat img_init = imread( img_file, CV_LOAD_IMAGE_COLOR);
      Mat img;
      resize(img_init, img, Size(320, 240));   
      //print_dim(img);

      if (img.rows > 220 && img.cols > 300){
        // the new files
        sprintf(f_number, "%09d", cnt);
        sprintf(f_roi_orig, "../synth_data/%09d_orig.jpg", cnt);
        sprintf(f_roi_warp, "../synth_data/%09d_warp.jpg", cnt);
        //cout << f_roi_orig << endl;
        f_labels << f_number << ";";

        // patch and jittered patch
        Patch patch(img, patch_size, max_jitter);
        patch.random_shift(gen);
        vector<Point2f> pts1 = patch.get_corners();
        patch.random_skew(gen);
        vector<Point2f> pts2 = patch.get_corners();

        Mat h = findHomography(pts1, pts2).inv();
         
        // save the label data
        for (int i_pts = 0; i_pts < pts1.size(); ++i_pts){
          f_labels << pts2[i_pts].x - pts1[i_pts].x << ",";
          f_labels << pts2[i_pts].y - pts1[i_pts].y << ",";
        }
        f_labels << endl;

        // apply the transformation
        Mat img_new;
        warpPerspective(img, img_new, h, img.size());

        if (show_plots){
          plot_pts(img, pts1);
          plot_pts(img, pts2);
          draw_poly(img, pts1, RED);
          draw_poly(img, pts2, BLUE);
        
          imshow("Source image", img);
          imshow("Warped source image", img_new);
        }

        int width = pts1[1].x - pts1[0].x;
        int height = pts1[2].y - pts1[1].y;
 
        // convert the original roi to grayscale
        Mat roi = Mat(img, Rect(pts1[0].x, pts1[0].y, width, height)).clone();
        Mat roi_gray(roi);
        cvtColor(roi, roi_gray, CV_RGB2GRAY);

        if (show_plots){
          imshow("Source rect", roi_gray);
        }
        imwrite(f_roi_orig, roi_gray);

        // convert the warped roi to grayscale
        Mat roi_new = Mat(img_new, Rect(pts1[0].x, pts1[0].y, width, height)).clone();
        Mat roi_new_gray(roi_new);
        cvtColor(roi_new, roi_new_gray, CV_RGB2GRAY);
        if (show_plots){
          imshow("Warped rect", roi_new_gray);
          waitKey(0);
        }
        imwrite(f_roi_warp, roi_new_gray);
        cnt++;
        if (cnt >= n_samples){
          had_enough = true;
          break;
        }
      } // end if img big enough
    } // end for file in dir
  } // end while

  f_labels.close();
#endif

	return 0;
}
