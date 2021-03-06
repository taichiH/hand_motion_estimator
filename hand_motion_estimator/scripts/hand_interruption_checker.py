#!/usr/bin/env python
import numpy as np
import rospy
import tf
import message_filters

from jsk_rviz_plugins.msg import OverlayText
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from hand_motion_estimator_msgs.msg import Interruption
from geometry_msgs.msg import Pose, Point, Quaternion

import time

class HandInterruptionChecker():

    def __init__(self):
        self.broadcaster = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()

        self.interrupt_dist_thresh = rospy.get_param('~interrupt_dist_thresh', 0.1)
        self.expansion = rospy.get_param('~expansion', 0.05)

        self.object_boxes = BoundingBoxArray()
        self.object_callback_called = False

        self.prev_box = BoundingBox()
        self.prev_interruption = False

        self.interruption_pub = rospy.Publisher(
            '~output/interruption', Interruption, queue_size=1)
        self.overlay_text_pub = rospy.Publisher(
            "~output/overlay_text", OverlayText, queue_size=1)
        self.transformed_boxes_pub = rospy.Publisher(
            '~output/transformed_boxes', BoundingBoxArray, queue_size=1)

        rospy.Subscriber(
            "~input/hand_pose_boxes", BoundingBoxArray, self.hand_callback)
        rospy.Subscriber(
            "~input/object_boxes", BoundingBoxArray, self.object_callback)


    def listen_transform(self, parent_frame, child_frame):
        box = BoundingBox()
        try:
            self.listener.waitForTransform(
                parent_frame, child_frame, rospy.Time(0), rospy.Duration(3.0))
            (trans, rot) = self.listener.lookupTransform(
                parent_frame, child_frame, rospy.Time(0))

            box.pose.position = Point(trans[0], trans[1], trans[2])
            box.pose.orientation = Quaternion(rot[0], rot[1], rot[2], rot[3])
            return box
        except:
            rospy.logwarn('cannot lookup transform')
            return box

    def transform_box(self, object_box, finger_box):
        object_frame = 'nearest_to_hand_object'
        self.broadcaster.sendTransform(
            (object_box.pose.position.x,
             object_box.pose.position.y,
             object_box.pose.position.z),
            (object_box.pose.orientation.x,
             object_box.pose.orientation.y,
             object_box.pose.orientation.z,
             object_box.pose.orientation.w),
            rospy.Time.now(),
            object_frame,
            object_box.header.frame_id)

        finger_frame = 'finger'
        self.broadcaster.sendTransform(
            (finger_box.pose.position.x,
             finger_box.pose.position.y,
             finger_box.pose.position.z),
            (1, 0, 0, 0),
            rospy.Time.now(),
            finger_frame,
            finger_box.header.frame_id)

        box = self.listen_transform(object_frame, finger_frame)
        return box

    def object_callback(self, object_boxes):
        self.object_callback_called = True
        self.object_boxes = object_boxes

    def hand_callback(self, hand_pose_boxes):
        if len(hand_pose_boxes.boxes) > 1:
            rospy.logwarn('this node requre input boxes size is 1')
            return

        if not self.object_callback_called:
            rospy.loginfo('wait for object_callback')
            return

        finger_box = hand_pose_boxes.boxes[0]

        min_distance = 24 ** 24
        nearest_box_index = 0
        for i, box in enumerate(self.object_boxes.boxes):
            box_pos = np.array([box.pose.position.x,
                                box.pose.position.y,
                                box.pose.position.z])
            ref_pos = np.array([finger_box.pose.position.x,
                                finger_box.pose.position.y,
                                finger_box.pose.position.z])
            distance = np.linalg.norm(ref_pos - box_pos)
            if distance < min_distance:
                min_distance = distance
                nearest_box_index = i

        interruption = False
        if min_distance < self.interrupt_dist_thresh:
            interruption = True

        cur_box = None
        if interruption and self.prev_interruption:
            cur_box = self.prev_box
        else:
            cur_box = self.object_boxes.boxes[nearest_box_index]

        transformed_box = self.transform_box(cur_box, finger_box)

        ### publishers

        ###
        text_msg = OverlayText()
        text_msg.text_size = 15
        text_msg.font = "DejaVu Sans Mono"
        text_msg.line_width = 1
        if interruption:
            text_msg.text = "interrupt\ndist: {}".format(min_distance)
        else:
            text_msg.text = "not interrupt\ndist: {}".format(min_distance)
        self.overlay_text_pub.publish(text_msg)

        ###
        interruption_msg = Interruption()
        interruption_msg.header = hand_pose_boxes.header
        interruption_msg.is_interrupt = interruption
        interruption_msg.box = transformed_box
        self.interruption_pub.publish(interruption_msg)

        ###
        boxes = BoundingBoxArray()
        boxes.header = transformed_box.header
        boxes.boxes.append(transformed_box)
        self.transformed_boxes_pub.publish(boxes)

        self.prev_interruption = interruption
        self.prev_box = cur_box

if __name__=='__main__':
    rospy.init_node('hand_interruption_checker')
    hand_interruption_checker = HandInterruptionChecker()
    rospy.spin()
