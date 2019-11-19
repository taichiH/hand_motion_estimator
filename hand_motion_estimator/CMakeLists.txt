cmake_minimum_required(VERSION 2.8.3)
project(hand_motion_estimator)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  cv_bridge
  image_transport
  geometry_msgs
  sensor_msgs
  jsk_recognition_msgs
  neatness_estimator_msgs
  hand_motion_estimator_msgs
  tf
  )

find_package(OpenCV REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})

add_definitions(-std=c++11)

catkin_package(
  CATKIN_DEPENDS
  LIBRARIES ${PROJECT_NAME}
)
