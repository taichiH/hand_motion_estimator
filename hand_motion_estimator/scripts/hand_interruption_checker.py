#!/usr/bin/env python
import numpy as np
import rospy
import message_filters

from jsk_rviz_plugins.msg import OverlayText
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from hand_motion_estimator_msgs.msg import Interruption

class HandInterruptionChecker():

    def __init__(self):
        self.interruption_pub = rospy.Publisher(
            '~output/interruption', Interruption, queue_size=1)
        self.overlay_text_pub = rospy.Publisher(
            "~output/overlay_text", OverlayText, queue_size=1)

        self.expansion = rospy.get_param('~expansion', 0.03)
        queue_size = rospy.get_param('~queue_size', 100)
        sub_hand_pose_boxes = message_filters.Subscriber(
            '~input/hand_pose_boxes', BoundingBoxArray, queue_size=queue_size)
        sub_object_boxes = message_filters.Subscriber(
            '~input/object_boxes', BoundingBoxArray, queue_size=queue_size)
        self.subs = [sub_hand_pose_boxes, sub_object_boxes]

        self.approximate_sync = rospy.get_param('~approximate_sync', True)
        if self.approximate_sync:
            slop = rospy.get_param('~slop', 0.1)
            sync = message_filters.ApproximateTimeSynchronizer(
                fs=self.subs, queue_size=queue_size, slop=slop)
        else:
            sync = message_filters.TimeSynchronizer(
                fs=self.subs, queue_size=queue_size)

        sync.registerCallback(self.callback)

    def is_interrupt(self, box, hand_pos):
        if hand_pos[0] > box.pose.position.x - (box.dimensions.x * 0.5 + self.expansion) and\
           hand_pos[0] < box.pose.position.x + (box.dimensions.x * 0.5 + self.expansion):
            return True
        if hand_pos[1] > box.pose.position.y - (box.dimensions.y * 0.5 + self.expansion) and\
           hand_pos[1] < box.pose.position.y + (box.dimensions.y * 0.5 + self.expansion):
            return True
        if hand_pos[2] > box.pose.position.z - (box.dimensions.z * 0.5 + self.expansion) and\
           hand_pos[2] < box.pose.position.z + (box.dimensions.z * 0.5 + self.expansion):
            return True

        return False

    def callback(self, hand_pose_boxes, object_boxes):

        if len(hand_pose_boxes.boxes) > 1:
            rospy.logwarn('this node requre input boxes size is 1')
            return

        finger_box = hand_pose_boxes.boxes[0]
        pos = np.array([finger_box.pose.position.x,
                        finger_box.pose.position.y,
                        finger_box.pose.position.z])

        min_distance = 24 ** 24
        nearest_box_index = 0
        for i, box in enumerate(object_boxes.boxes):
            box_pos = np.array([box.pose.position.x,
                                box.pose.position.y,
                                box.pose.position.z])
            distance = np.linalg.norm(pos - box_pos)
            if distance < min_distance:
                min_distance = distance
                nearest_box_indedx = i

        result = self.is_interrupt(object_boxes.boxes[i], pos)

        text_msg = OverlayText()
        text_msg.text_size = 15
        text_msg.font = "DejaVu Sans Mono"
        text_msg.line_width = 1
        if result:
            text_msg.text = "interrupt"
        else:
            text_msg.text = "not interrupt"
        self.overlay_text_pub.publish(text_msg)

        interruption_msg = Interruption()
        interruption_msg.header = hand_pose_boxes.header
        interruption_msg.is_interrupt = result
        interruption_msg.box = object_boxes.boxes[i]
        self.interruption_pub.publish(interruption_msg)


if __name__=='__main__':
    rospy.init_node('hand_interruption_checker')
    hand_interruption_checker = HandInterruptionChecker()
    rospy.spin()
