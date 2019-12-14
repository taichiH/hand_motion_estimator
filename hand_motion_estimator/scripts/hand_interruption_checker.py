#!/usr/bin/env python
import numpy as np
import rospy
import tf
import message_filters

from jsk_rviz_plugins.msg import OverlayText
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from hand_motion_estimator_msgs.msg import Interruption
from geometry_msgs.msg import Pose, Point, Quaternion

class HandInterruptionChecker():

    def __init__(self):
        self.broadcaster = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()

        self.interruption_pub = rospy.Publisher(
            '~output/interruption', Interruption, queue_size=1)
        self.overlay_text_pub = rospy.Publisher(
            "~output/overlay_text", OverlayText, queue_size=1)
        self.expanded_box_pub = rospy.Publisher(
            '~output/expanded_box', BoundingBox, queue_size=1)

        self.interrupt_dist_thresh = rospy.get_param('~interrupt_dist_thresh', 0.1)
        self.expansion = rospy.get_param('~expansion', 0.05)
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

    def transform_poses(self, pose, label, frame_id, parent):
        self.broadcaster.sendTransform(
            (pose.position.x, pose.position.y, pose.position.z),
            (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
            rospy.Time.now(), label, frame_id)
        box =  self.listen_transform(parent, label)
        return box

    def is_interrupt(self, box, finger_box):
        frame_name = 'nearest_to_hand_object'
        self.broadcaster.sendTransform(
            (box.pose.position.x,
             box.pose.position.y,
             box.pose.position.z),
            (box.pose.orientation.x,
             box.pose.orientation.y,
             box.pose.orientation.z,
             box.pose.orientation.w),
            rospy.Time.now(), frame_name, box.header.frame_id)
        object_pos = np.array([box.pose.position.x,
                               box.pose.position.y,
                               box.pose.position.z])

        transformed_finger_box = self.transform_poses(
            Pose(Point(finger_box.pose.position.x,
                       finger_box.pose.position.y,
                       finger_box.pose.position.z),
                 Quaternion(0,0,0,1)),
            'finger_pos',
            finger_box.header.frame_id,
            frame_name)
        transformed_finger_pos = [transformed_finger_box.pose.position.x,
                                  transformed_finger_box.pose.position.y,
                                  transformed_finger_box.pose.position.z]
        print(transformed_finger_pos)

        object_to_hand_dist = np.linalg.norm(object_pos - transformed_finger_pos)
        print('object_to_hand_dist: ', object_to_hand_dist)
        print('interrupt_dist_thresh: ', self.interrupt_dist_thresh)
        if object_to_hand_dist > self.interrupt_dist_thresh:
            return True, object_to_hand_dist
        else:
            return False, object_to_hand_dist

    def callback(self, hand_pose_boxes, object_boxes):
        if len(hand_pose_boxes.boxes) > 1:
            rospy.logwarn('this node requre input boxes size is 1')
            return

        finger_box = hand_pose_boxes.boxes[0]

        min_distance = 24 ** 24
        nearest_box_index = 0
        for i, box in enumerate(object_boxes.boxes):
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

        # print(object_boxes.boxes[nearest_box_index])
        result, dist = self.is_interrupt(object_boxes.boxes[nearest_box_index], finger_box)

        text_msg = OverlayText()
        text_msg.text_size = 15
        text_msg.font = "DejaVu Sans Mono"
        text_msg.line_width = 1
        if result:
            text_msg.text = "interrupt\ndist: {}".format(dist)
        else:
            text_msg.text = "not interrupt\ndist: {}".format(dist)
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
