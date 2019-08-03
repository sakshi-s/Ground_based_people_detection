# Ground_based_people_detection
Worked on ground based people detection using RGB-depth images and Point Cloud Library. Live RGB-D data was taken from a Microsoft Kinect. It was used to detect people standing/walking on a planar ground plane in real time with standard CPU computation.
1. Delete all the files in build directory
2. Terminal commands:
cmake -H. -Bbuild
cmake --build build -- -j16
cd build
./ground_based_rgbd_people_detector
