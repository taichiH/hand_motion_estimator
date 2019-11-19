# hand_motion_estimator

## dependencies
jsk_perception
jsk_recognition_msgs

## usage

- input
  '~input/hand_pose' (jsk_recognition_msgs/PeoplePoseArray)
  '~input/rgb_image' (sensor_msgs/Image)
  '~input/depth_image' (sensor_msgs/Image)

- output
  '~output/motion' (hand_motion_estimator_msgs/Motion)
  