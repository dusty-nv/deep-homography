cmake_minimum_required(VERSION 3.5)
project( ImagePrep )
set (CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )

MESSAGE(STATUS "Include dirs ${SOURCES} ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "LINK LIBRARIES ${OpenCV_LIBS}")

add_executable( make_homography_data_coco make_homography_data_coco.cc image_tools.cc) 
target_link_libraries( make_homography_data_coco ${OpenCV_LIBS} stdc++fs)

add_executable( make_homography_data_kitti make_homography_data_kitti.cc image_tools.cc) 
target_link_libraries( make_homography_data_kitti ${OpenCV_LIBS} stdc++fs)
