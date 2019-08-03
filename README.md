# Ground_based_people_detection
Worked on ground based people detection using RGB-depth images and Point Cloud Library. Live RGB-D data was taken from a Microsoft Kinect. It was used to detect people standing/walking on a planar ground plane in real time with standard CPU computation.
A. Delete all the files in build directory
B. Terminal commands:
 1. cmake -H. -Bbuild
 2. cmake --build build -- -j16
 3. cd build
 4. ./ground_based_rgbd_people_detector
