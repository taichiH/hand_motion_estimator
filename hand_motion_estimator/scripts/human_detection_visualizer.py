#!/usr/bin/env python

import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

import rospy
import message_filters
from jsk_recognition_msgs.msg import RectArray
from jsk_recognition_msgs.msg import ClassificationResult

class HumanDetectionVisualizer():

    def __init__(self):
        self.cv_bridge = CvBridge()
        self.approximate_sync = rospy.get_param('~approximate_sync', True)

        self.human_img_pub = rospy.Publisher(
            '~output/viz', Image, queue_size=1)

        queue_size = rospy.get_param('~queue_size', 100)

        sub_hand_image = message_filters.Subscriber(
            '~input/hand_image', Image, queue_size=queue_size)
        sub_face_rect = message_filters.Subscriber(
            '~input/face_rect', RectArray, queue_size=queue_size)
        sub_face_class = message_filters.Subscriber(
            '~input/face_class', ClassificationResult, queue_size=queue_size)

        self.subs = [sub_hand_image, sub_face_rect, sub_face_class]
        if self.approximate_sync:
            slop = rospy.get_param('~slop', 0.1)
            sync = message_filters.ApproximateTimeSynchronizer(
                fs=self.subs, queue_size=queue_size, slop=slop)
        else:
            sync = message_filters.TimeSynchronizer(
                fs=self.subs, queue_size=queue_size)

        sync.registerCallback(self.callback)

    def callback(self, hand_img_msg, face_rect_msg, face_class_msg):
        hand_image = self.cv_bridge.imgmsg_to_cv2(hand_img_msg, 'bgr8')
        for rect, cls in zip(face_rect_msg.rects, face_class_msg.label_names):
            color = (0,0,0)
            if cls == 'toward':
                color = (0,255,0)
            else:
                color = (0,0,255)

            hand_image = cv2.rectangle(
                hand_image,
                (int(rect.x), int(rect.y)),
                (int(rect.x + rect.width), int(rect.y + rect.height)),
                color, 3)
            cv2.putText(
                hand_image,
                cls,
                (rect.x, rect.y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        human_img_msg = self.cv_bridge.cv2_to_imgmsg(hand_image, 'bgr8')
        human_img_msg.header = human_img_msg.header
        self.human_img_pub.publish(human_img_msg)


if __name__=='__main__':
    rospy.init_node('human_detection_visualizer')
    vis = HumanDetectionVisualizer()
    rospy.spin()
