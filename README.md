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


## sample
download rosbag data
```
roscd hand_motion_estimator/sample/data
jsk_data get translation.bag
```

### simple sample launch
```
roslaunch hand_motion_estimator sample_hand_motion_estimator.launch
```

### combination of object detection and hand motion estimation
you must install `neatness_estimator` (https://github.com/taichiH/neatness_estimator).
```
git clone https://github.com/taichiH/neatness_estimator
catkin build neatness_estimator
```

sample launch
```
roslaunch hand_motion_estimator sample_hand_motion_estimator.launch interruption_check:=true
```

![translation_motion_estimation](https://user-images.githubusercontent.com/22497720/70332308-a0c16500-1884-11ea-87b5-544bde2f385e.gif)
