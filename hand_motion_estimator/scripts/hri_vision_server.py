#!/usr/bin/env python

import sys
import numpy as np

import rospy

from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from jsk_recognition_msgs.msg import RectArray
from jsk_recognition_msgs.msg import ClassificationResult
from jsk_rviz_plugins.msg import OverlayText

from hand_motion_estimator_msgs.srv import VisionServer, VisionServerResponse
from hand_motion_estimator_msgs.msg import Interruption
from hand_motion_estimator_msgs.msg import Motion

from neatness_estimator_msgs.srv import CorrectData, CorrectDataResponse

class HriVisionServer():

    def __init__(self):
        self.face_class = ClassificationResult()
        self.is_interrupt = False
        self.hand_motion = ''

        self.pub_overlay_text = rospy.Publisher(
            "~output/overlay_text", OverlayText, queue_size=1)
        self.pub_interrupt_overlay_text = rospy.Publisher(
            "~output/interrupt_overlay_text", OverlayText, queue_size=1)

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
        text_msg = OverlayText()
        text_msg.text_size = 15
        text_msg.font = "DejaVu Sans Mono"
        text_msg.line_width = 1

        res = VisionServerResponse()

        sleep_interval = 0.1 # sec
        timer = 0.0
        while self.face_class is not 'toward':
            rospy.sleep(sleep_interval)
            timer += sleep_interval
            text_msg.text = 'searching human {} sec [{}]'.format(
                req.observation_time, timer)
            self.pub_overlay_text.publish(text_msg)

            if timer >= req.observation_time:
                rospy.loginfo(
                    'time up (observation time: %f sec)'
                    %(req.observation_time))
                res.face_state = self.face_class
                res.success = False
                text_msg.text = 'time up'
                self.pub_overlay_text.publish(text_msg)
                return res

        text_msg.text = 'detect aware human {}'.format(self.face_class)
        self.pub_overlay_text.publish(text_msg)

        res.face_state = self.face_class
        res.success = True
        return res

    def get_hand_motion(self, req):
        text_msg = OverlayText()
        text_msg.text_size = 15
        text_msg.font = "DejaVu Sans Mono"
        text_msg.line_width = 1

        res = VisionServerResponse()
        data_corrector = CorrectData()

        hand_motion = ''
        sleep_interval = 0.1 # sec
        timer = 0.0
        while not self.is_interrupt:
            rospy.sleep(sleep_interval)
            timer += sleep_interval
            text_msg.text = 'waiting hand interruption {} sec {} ]'.format(
                req.observation_time, timer)
            self.pub_interrupt_overlay_text.publish(text_msg)

            if timer >= req.observation_time:
                rospy.loginfo(
                    'abort hand interrupt (observation time: %f sec)'
                    %(req.observation_time))
                text_msg.text = 'time up'
                self.pub_interrupt_overlay_text.publish(text_msg)

                res.hand_motion = 'unknown'
                res.success = False
                return res

        while self.is_interrupt:
            text_msg.text = 'interruption detected motion: {}'.format(
                self.hand_motion)
            self.pub_interrupt_overlay_text.publish(text_msg)
            rospy.loginfo(text_msg.text)
            hand_motion = self.hand_motion

        text_msg.text = 'interruption ended, final motion {}'.format(
            self.hand_motion)
        self.pub_interrupt_overlay_text.publish(text_msg)
        rospy.loginfo(text_msg.text)

        res.hand_motion = hand_motion
        res.success = True
        return res

    def vision_server(self, req):

        if req.task == 'get_aware_human':
            rospy.loginfo(req.task)
            return self.get_aware_human(req)

        if req.task == 'get_hand_motion':
            rospy.loginfo(req.task)
            return self.get_hand_motion(req)

if __name__ == "__main__":
    rospy.init_node("hri_vision_server")
    vision_server = HriVisionServer()
    rospy.spin()
