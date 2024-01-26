cmake_minimum_required(VERSION 3.0.2)
project(hba)

set(CMAKE_BUILD_TYPE "Release")
ADD_COMPILE_OPTIONS(-std=c++17)
set(CMAKE_CXX_FLAGS "-std=c++17 -O3" )
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -std=c++0x -std=c++17 -fexceptions")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  rosbag
  std_msgs
  geometry_msgs
  visualization_msgs
  tf
)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED QUIET)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs visualization_msgs tf
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS
)

# Hierarchical Bundle Adjustment
add_executable(hba source/hba.cpp)
target_link_libraries(hba ${catkin_LIBRARIES} ${PCL_LIBRARIES} gtsam)

add_executable(visualize_map source/visualize.cpp)
target_link_libraries(visualize_map ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(calculate_MME source/calculate_MME.cpp)
target_link_libraries(calculate_MME ${catkin_LIBRARIES} ${PCL_LIBRARIES} gtsam)