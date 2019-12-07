#!/usr/bin/env python

import sys
import numpy as np

import rospy

from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from jsk_recognition_msgs.msg import RectArray
from jsk_recognition_msgs.msg import ClassificationResult

from hand_motion_estimator_msgs.srv import VisionServer, VisionServerResponse
from hand_motion_estimator_msgs.msg import Interruption
from hand_motion_estimator_msgs.msg import Motion

from neatness_estimator_msgs.srv import CorrectData, CorrectDataResponse

class NeatnessEstimatorVisionServer():

    def __init__(self):
        self.face_class = ClassificationResult()
        self.is_interrupt = False
        self.hand_motion = ''

        rospy.Subscriber(
            "~input_face_class", ClassificationResult, self.face_box_callback)
        rospy.Subscriber(
            "~input_hand_interruption", Interruption, self.hand_interruption_callback)
        rospy.Subscriber(
            "~input_hand_motion", Motion, self.hand_motion_callback)

        rospy.Service(
            '/hri_vision_server', VisionServer, self.vision_server)

    def hand_interruption_callback(self, interruption_msg):
        self.is_interrupt = interruption_msg.is_interrupt

    def hand_motion_callback(self, motion_msg):
        self.hand_motion = motion_msg.motion

    def face_box_callback(self, classes_msg):
        if 'toward' in classes_msg.label_names:
            self.face_class = 'toward'
        else:
            self.face_class = 'away'

    def get_aware_human(self, req):
        res = VisionServerResponse()

        sleep_interval = 0.1 # sec
        time_count = 0.0
        while self.face_class == 'away':
            rospy.sleep(sleep_interval)
            time_count += sleep_interval

            if time_count > (req.observation_time / sleep_interval):
                rospy.loginfo('time up (observation time: %f sec)' %(req.observation_time))
                res.face_state = self.face_class
                res.success = False
                return res

        res.face_state = self.face_class
        res.success = True
        return res

    def get_hand_motion(self, req):
        res = VisionServerResponse()
        data_corrector = CorrectData()

        ## interrupt の前に物体の状態をdataにして、
        ## その後行った操作をmotion_labelにする必要がある

        # example
        # data: (list color geometry group)
        # item: 13 (mixjuice)

        # get object appearance state
        # data_corerctor.data = data

        hand_motion = ''
        while self.is_interrupt:
            hand_motion = self.hand_motion

        ## call data corrector
        # data_corerctor.motion_label = hand_motion

        res.hand_motion = hand_motion
        res.success = True
        return res

    def vision_server(self, req):
        for label in classes_msg.label_names:
            pass

        if req.task == 'get_aware_human':
            return self.get_aware_human(req)

        if req.task == 'get_hand_motion':
            return self.get_hand_motion(req)

if __name__ == "__main__":
    rospy.init_node("hri_vision_server")
    vision_server = HriVisionServer()
    rospy.spin()
