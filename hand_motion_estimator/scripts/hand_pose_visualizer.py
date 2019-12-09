#!/usr/bin/env python

import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import rospy
import message_filters
from jsk_recognition_msgs.msg import PeoplePoseArray, PeoplePose

class HandPoseVisualizer():

    def __init__(self):
        self.debug = rospy.get_param('~debug', False)
        self.cv_bridge = CvBridge()
        self.approximate_sync = rospy.get_param('~approximate_sync', True)
        self.ksize = rospy.get_param('~ksize', 3)
        self.extract_joint = rospy.get_param('~extract_joint', 'RHand3')

        self.debug_img_pub = rospy.Publisher(
            '~debug_output', Image, queue_size=1)
        self.mask_img_pub = rospy.Publisher(
            '~output/mask', Image, queue_size=1)

        queue_size = rospy.get_param('~queue_size', 100)

        sub_hand_pose = message_filters.Subscriber(
            '~input/hand_pose', PeoplePoseArray, queue_size=queue_size)
        sub_rgb_image = message_filters.Subscriber(
            '~input/rgb_image', Image, queue_size=queue_size)
        sub_depth_image = message_filters.Subscriber(
            '~input/depth_image', Image, queue_size=queue_size)
        sub_cam_info = message_filters.Subscriber(
            '~input/camera_info', CameraInfo, queue_size=queue_size)
        self.subs = [sub_hand_pose, sub_rgb_image, sub_cam_info, sub_depth_image]
        if self.approximate_sync:
            slop = rospy.get_param('~slop', 0.1)
            sync = message_filters.ApproximateTimeSynchronizer(
                fs=self.subs, queue_size=queue_size, slop=slop)
        else:
            sync = message_filters.TimeSynchronizer(
                fs=self.subs, queue_size=queue_size)

        sync.registerCallback(self.callback)

    def callback(self, hand_pose_msg, rgb_msg, cam_info_msg, depth_msg):
        rgb_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        debug_image = rgb_image.copy()

        mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8)

        if len(hand_pose_msg.poses) == 0:
            return

        # left and right hand
        for people_pose in zip(hand_pose_msg.poses):
            poses = people_pose[0].poses
            limbs = people_pose[0].limb_names

            # each joint
            for pose, limb in zip(poses, limbs):
                if limb != self.extract_joint:
                    continue
                u, v = pose.position.x, pose.position.y
                lt = (int(u - self.ksize), int(v - self.ksize))
                rb = (int(u + self.ksize), int(v + self.ksize))
                debug_limb = list(limb)[-2] + list(limb)[-1]
                cv2.rectangle(debug_image, lt, rb, (255, 0, 0), 2)
                cv2.putText(
                    debug_image, debug_limb, lt, cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (0,0,255), 1, cv2.LINE_AA)

                mask[lt[1]:rb[1], lt[0]:rb[0]] = 255

        debug_img_msg = self.cv_bridge.cv2_to_imgmsg(debug_image, 'bgr8')
        debug_img_msg.header = rgb_msg.header
        self.debug_img_pub.publish(debug_img_msg)
        mask_msg = self.cv_bridge.cv2_to_imgmsg(mask, 'mono8')
        mask_msg.header = rgb_msg.header
        self.mask_img_pub.publish(mask_msg)

if __name__=='__main__':
    rospy.init_node('hand_pose_visualizer')
    vis = HandPoseVisualizer()
    rospy.spin()
