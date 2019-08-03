/*
 * Software License Agreement (BSD License)
 *
 * Point Cloud Library (PCL) - www.pointclouds.org
 * Copyright (c) 2013-, Open Perception, Inc.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following
 * disclaimer in the documentation and/or other materials provided
 * with the distribution.
 * * Neither the name of the copyright holder(s) nor the names of its
 * contributors may be used to endorse or promote products derived
 * from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * main_ground_based_people_detection_app.cpp
 * Created on: Nov 30, 2012
 * Author: Matteo Munaro
 *
 * Example file for performing people detection on a Kinect live stream.
 * As a first step, the ground is manually initialized, then people detection is performed with the GroundBasedPeopleDetectionApp class,
 * which implements the people detection algorithm described here:
 * M. Munaro, F. Basso and E. Menegatti,
 * Tracking people within groups with RGB-D data,
 * In Proceedings of the International Conference on Intelligent Robots and Systems (IROS) 2012, Vilamoura (Portugal), 2012.
 */

#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/openni2_grabber.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/common/time.h>
#include "detectclass.hpp"


#include <mutex>
#include <thread>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

// PCL viewer //
pcl::visualization::PCLVisualizer viewer("PCL Viewer");

// Mutex: //
//boost::mutex cloud_mutex;

std::mutex cloud_mutex;
int frames=6;
bool background_subtraction=true;
double background_octree_resolution=1.0;
//using namespace std::chrono_literals;


enum { COLS = 1080, ROWS = 720 };

int print_help()
{
  cout << "*******************************************************" << std::endl;
  cout << "Ground based people detection app options:" << std::endl;
  cout << "   --help    <show_this_help>" << std::endl;
  cout << "   --svm     <path_to_svm_file>" << std::endl;
  cout << "   --conf    <minimum_HOG_confidence (default = -1.5)>" << std::endl;
  cout << "   --min_h   <minimum_person_height (default = 1.3)>" << std::endl;
  cout << "   --max_h   <maximum_person_height (default = 2.3)>" << std::endl;
  cout << "*******************************************************" << std::endl;
  return 0;
}

void cloud_cb_ (const PointCloudT::ConstPtr &callback_cloud, PointCloudT::Ptr& cloud,
    bool* new_cloud_available_flag)
{
  cloud_mutex.lock ();    // for not overwriting the point cloud from another thread
  *cloud = *callback_cloud;
  *new_cloud_available_flag = true;
  cloud_mutex.unlock ();
}

struct callback_args{
  // structure used to pass arguments to the callback function
  PointCloudT::Ptr clicked_points_3d;
  pcl::visualization::PCLVisualizer::Ptr viewerPtr;
};

void
pp_callback (const pcl::visualization::PointPickingEvent& event, void* args)
{
  struct callback_args* data = (struct callback_args *)args;
  if (event.getPointIndex () == -1)
    return;
  PointT current_point;
  event.getPoint(current_point.x, current_point.y, current_point.z);
  data->clicked_points_3d->points.push_back(current_point);
  // Draw clicked points in red:
  pcl::visualization::PointCloudColorHandlerCustom<PointT> red (data->clicked_points_3d, 255, 0, 0);
  data->viewerPtr->removePointCloud("clicked_points");
  data->viewerPtr->addPointCloud(data->clicked_points_3d, red, "clicked_points");
  data->viewerPtr->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "clicked_points");
  std::cout << current_point.x << " " << current_point.y << " " << current_point.z << std::endl;
}

/*computeBackgroundCloud (float voxel_size, std::string frame_id,PointCloudT::Ptr& background_cloud)
{
  std::cout << "Background acquisition..." << std::flush;

  // Create background cloud:
  background_cloud->header = cloud->header;
  background_cloud->points.clear();
  for (unsigned int i = 0; i < frames; i++)
  {
    // Point cloud pre-processing (downsampling and filtering):
    PointCloudT::Ptr cloud_filtered(new PointCloudT);
    cloud_filtered = people_detector.preprocessCloud (cloud);

    *background_cloud += *cloud_filtered;
    ros::spinOnce();
    rate.sleep();
  }

  // Voxel grid filtering:
  PointCloudT::Ptr cloud_filtered(new PointCloudT);
  pcl::VoxelGrid<PointT> voxel_grid_filter_object;
  voxel_grid_filter_object.setInputCloud(background_cloud);
  voxel_grid_filter_object.setLeafSize (voxel_size, voxel_size, voxel_size);
  voxel_grid_filter_object.filter (*cloud_filtered);

  background_cloud = cloud_filtered;

  // Background saving:
  pcl::io::savePCDFileASCII ("/tmp/background_" + frame_id.substr(1, frame_id.length()-1) + ".pcd", *background_cloud);

  std::cout << "done." << std::endl << std::endl;
}*/
int main (int argc, char** argv)
{
  if(pcl::console::find_switch (argc, argv, "--help") || pcl::console::find_switch (argc, argv, "-h"))
        return print_help();

  // Algorithm parameters:
      std::string svm_filename = "/data/trainedLinearSVMForPeopleDetectionWithHOG.yaml";

  float min_confidence = -1.5;
  float min_height = 1.3;
  float max_height = 2.3;
  float voxel_size = 0.04;
  Eigen::Matrix3f rgb_intrinsics_matrix;
  rgb_intrinsics_matrix << 1067.7197989035835, 0.0, 1035.1100266322298, 0.0, 1057.2361156884658, 576.2813813933719, 0.0, 0.0, 1.0; // Kinect RGB camera intrinsics

  // Read if some parameters are passed from command line:
  pcl::console::parse_argument (argc, argv, "--svm", svm_filename);
  pcl::console::parse_argument (argc, argv, "--conf", min_confidence);
  pcl::console::parse_argument (argc, argv, "--min_h", min_height);
  pcl::console::parse_argument (argc, argv, "--max_h", max_height);

  // Read Kinect live stream:
  PointCloudT::Ptr cloud (new PointCloudT);
//  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
 if (pcl::io::loadPCDFile("/libfreenect2pclgrabber/build1/cloud7.pcd", *cloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read test file \n");
    return (-1);
  }


  // Wait for the first
  //frame:   // for not overwriting the point cloud

  // Display pointcloud:
  pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
  viewer.addPointCloud<PointT> (cloud, rgb, "input_cloud");
  viewer.setCameraPosition(0,0,-2,0,-1,0,0);
   GroundBasedPeopleDetectionApp_M<PointT> people_detector;    // people detection object

  // Add point picking callback to viewer:
  struct callback_args cb_args;
  PointCloudT::Ptr clicked_points_3d (new PointCloudT);
  cb_args.clicked_points_3d = clicked_points_3d;
  cb_args.viewerPtr = pcl::visualization::PCLVisualizer::Ptr(&viewer);
  viewer.registerPointPickingCallback (pp_callback, (void*)&cb_args);
  std::cout << "Shift+click on three floor points, then press 'Q'..." << std::endl;

  // Spin until 'Q' is pressed:
  viewer.spin();
  std::cout << "done." << std::endl;

  cloud_mutex.unlock ();

  // Ground plane estimation:
  Eigen::VectorXf ground_coeffs;
  ground_coeffs.resize(4);
  std::vector<int> clicked_points_indices;
  for (unsigned int i = 0; i < clicked_points_3d->points.size(); i++)
    clicked_points_indices.push_back(i);
  pcl::SampleConsensusModelPlane<PointT> model_plane(clicked_points_3d);
  model_plane.computeModelCoefficients(clicked_points_indices,ground_coeffs);
  std::cout << "Ground plane: " << ground_coeffs(0) << " " << ground_coeffs(1) << " " << ground_coeffs(2) << " " << ground_coeffs(3) << std::endl;

   // Initialization for background subtraction:
  PointCloudT::Ptr background_cloud (new PointCloudT);
  //std::string frame_id = cloud->header.frame_id;
  //max_background_frames = int(background_seconds * rate_value);
  if (background_subtraction)
  {
    std::cout << "Background subtraction enabled." << std::endl;
    pcl::io::loadPCDFile("/home/dyan/libfreenect2pclgrabber/build1/cloudbg.pcd", *background_cloud);
    // Try to load the background from file:
    /*if (pcl::io::loadPCDFile<PointT> ("/tmp/background_" + frame_id.substr(1, frame_id.length()-1) + ".pcd", *background_cloud) == -1)
    {
      // File not found, then background acquisition:
      computeBackgroundCloud (max_background_frames, voxel_size, frame_id, rate, background_cloud);
    }
    elsehttps://www.google.com/search?client=ubuntu&channel=fs&q=octree&ie=utf-8&oe=utf-8
    {
      std::cout << "Background read from file." << std::endl << std::endl;
    }*/

    people_detector.setBackground(background_subtraction, background_octree_resolution, background_cloud);
  }

  // Initialize new viewer:
  pcl::visualization::PCLVisualizer viewer("PCL Viewer");          // viewer initialization
  viewer.setCameraPosition(0,0,-2,0,-1,0,0);

  // Create classifier for people detection:
  pcl::people::PersonClassifier<pcl::RGB> person_classifier;
  person_classifier.loadSVMFromFile(svm_filename);   // load trained SVM

  // People detection app initializationhttps://www.google.com/search?client=ubuntu&channel=fs&q=octree&ie=utf-8&oe=utf-8
  people_detector.setVoxelSize(voxel_size);                        // set the voxel size
  people_detector.setIntrinsics(rgb_intrinsics_matrix);            // set RGB camera intrinsic parameters
  people_detector.setClassifier(person_classifier);
  people_detector.setPersonClusterLimits(min_height, max_height, 0.1, 8.0);           // set person classifier
  //people_detector.setHeightLimits(min_height, max_height);         // set person classifier
//  people_detector.setSensorPortraitOrientation(true);             // set sensor orientation to vertical

  // For timing:
  static unsigned count = 0;
  static double last ;//pcl::StopWatch::getTimeSeconds();


  // no_ground_cloud_ = PointCloudPtr (new PointCloud);	
     
     PointCloudT::Ptr no_ground_cloud (new PointCloudT);

  // Main loop:
std::string cloudname = "/home/dyan/libfreenect2pclgrabber/build1/cloud6.pcd";
  int i=4;
  while(!viewer.wasStopped()){
    int a=i+1;
    char b=a+'0';
    cloudname[46]=b;
    std::cout<<"Frame number"<<"******"<<i+1<<endl;
    i++;
    i=i%9;
    if(i<=4)
      i+=4;
    pcl::io::loadPCDFile(cloudname, *cloud);
           // Perform people detection on the new cloud:
      std::vector<pcl::people::PersonCluster<PointT> > clusters;   // vector containing persons clusters
      people_detector.setInputCloud(cloud);
      people_detector.setGround(ground_coeffs);
      people_detector.compute(clusters); 
            no_ground_cloud=  people_detector.getNoGroundCloud ();


      ground_coeffs = people_detector.getGround();                 // get updated floor coefficients

      // Draw cloud and people bounding boxes in the viewer:
      viewer.removeAllPointClouds();
      viewer.removeAllShapes();

      pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(no_ground_cloud);
      //viewer.addPointCloud<PointT> (cloud, rgb, "input_cloud");
      viewer.addPointCloud<PointT> (no_ground_cloud, rgb, "input_cloud");
        viewer.spinOnce();

}

  return 0;
}
//https://www.google.com/search?client=ubuntu&channel=fs&q=octree&ie=utf-8&oe=utf-8
//cmake -H. -Bbuild
//cmake --build build -- -j16