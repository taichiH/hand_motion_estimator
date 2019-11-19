#!/usr/bin/env python

import numpy as np

import rospy
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox

class HandMotionEstimator():

    def __init__(self):
        rospy.Subscriber(
            "~input/hand_pose_box", BoundingBoxArray, self.callback)

    def callback(self, msg):
        # TODO
        # implement hand motion estimation code
        pass

if __name__=='__main__':
    rospy.init_node('hand_motion_estimator')
    motion_estimator = HandMotionEstimator()
    rospy.spin()
