#!/usr/bin/env python

import os
import csv
import math
from enum import Enum

import numpy as np
from scipy import interpolate
from scipy.stats import norm
import scipy.spatial.distance as dist

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import rospy
import rospkg

from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
from jsk_rviz_plugins.msg import OverlayText
from sensor_msgs.msg import Image
from hand_motion_estimator_msgs.msg import Motion
from hand_motion_estimator_msgs.srv import GetHistogram, GetHistogramResponse

from std_msgs.msg import Float32

class HandMotionEstimator():

    def __init__(self):
        self.toggle = 1
        self.motion_labels = ['rot', 'trans', 'nonmove']

        self.state = Enum('state', 'buffered not_buffered isnan error')
        self.move_state = Enum('move_state', 'move nonmove')
        self.prev_move_state = self.move_state.nonmove
        self.motions_count = {self.motion_labels[0] : 0,
                              self.motion_labels[1] : 0,
                              self.motion_labels[2] : 0}
        self.segment_motion = ''

        self.interpolate_flow = rospy.get_param('~interpolate_flow', False)
        self.chunk_size = rospy.get_param('~chunk_size', 5)
        self.movement_thresh = rospy.get_param('~movement_thresh', 15)
        self.use_pca = rospy.get_param('~use_pca', False)
        self.interpolation_scale = rospy.get_param('~interpolation_scale', 2)
        self.flow_chunk = []

        self.pca = PCA()

        self.pub_overlay_text = rospy.Publisher(
            "~output/overlay_text", OverlayText, queue_size=1)
        self.pub_movement = rospy.Publisher(
            "~output/movement", Float32, queue_size=1)
        self.pub_motion = rospy.Publisher(
            "~output/motion", Motion, queue_size=1)

        rospy.Subscriber(
            "~input/hand_pose_box", BoundingBoxArray, self.callback)


    def calc_pca(self, chunk):
        self.pca.fit(np.array(chunk))
        pca_vec = self.pca.components_[0]
        return np.array(pca_vec)

    def spline_interpolate(self, flow_chunk):
        x_sample = flow_chunk.T[0]
        y_sample = flow_chunk.T[1]
        z_sample = flow_chunk.T[2]

        try:
            num_true_pts = len(x_sample) * self.interpolation_scale
            tck, u = interpolate.splprep([x_sample,y_sample,z_sample], s=2, k=3)
            x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
            u_fine = np.linspace(0,1,num_true_pts)
            x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
            interpolated_chunk = np.array([x_fine, y_fine, z_fine]).T
        except:
            return self.state.error

        return list(interpolated_chunk)

    def make_flow_chunk(self, pos):

        if len(self.flow_chunk) <= self.chunk_size:
            self.flow_chunk.append(pos)
            return self.state.not_buffered
        else:
            self.flow_chunk.pop(0)
            self.flow_chunk.append(pos)

            interpolated_chunk = None
            if self.interpolate_flow:
                interpolated_chunk = self.spline_interpolate(np.array(self.flow_chunk))
            else:
                interpolated_chunk = self.flow_chunk

            return interpolated_chunk

    def callback(self, boxes_msg):
        text_msg = OverlayText()
        text_msg.text_size = 15
        text_msg.font = "DejaVu Sans Mono"
        text_msg.line_width = 1

        if len(boxes_msg.boxes) > 1:
            rospy.logwarn('this node requre input boxes size is 1')
            return

        finger_box = boxes_msg.boxes[0]
        pos = np.array([finger_box.pose.position.x,
                        finger_box.pose.position.y,
                        finger_box.pose.position.z])

        interpolated_chunk = self.make_flow_chunk(pos)
        if interpolated_chunk == self.state.not_buffered:
            return
        elif interpolated_chunk == self.state.error:
            return

        pca_vec = self.calc_pca(interpolated_chunk)
        max_element = 0
        idx = 0
        for i, e in enumerate(pca_vec):
            if abs(e) > max_element:
                max_element = abs(e)
                idx = i

        most_move_axis = ['x', 'y', 'z'][idx]

        # last frame and last last frame movement
        movement = np.linalg.norm(
            interpolated_chunk[-1] - interpolated_chunk[-2]) * 1000
        self.pub_movement.publish(data=movement)


        motion = ''
        if most_move_axis == 'x':
            motion = 'trans'
        else:
            motion = 'rot'

        if movement < self.movement_thresh:
            move_state = self.move_state.nonmove
            motion = 'nonmove'
            self.flow_chunk = []
        else:
            move_state = self.move_state.move

        max_count = 0
        self.motions_count[motion] += 1
        if self.prev_move_state == self.move_state.nonmove and \
           move_state == self.move_state.move:
            for label in self.motion_labels:
                self.motions_count[label] = 0
        elif self.prev_move_state == self.move_state.move and \
             move_state == self.move_state.nonmove:
            for label in self.motion_labels:
                if self.motions_count[label] > max_count:
                    max_count = self.motions_count[label]
                    self.segment_motion = label
                self.motions_count[label] = 0
        text_msg.text = 'sequence motion: ' + motion + \
                        '\nmovement: ' + str(movement) + \
                        '\nlast segment motion: ' + self.segment_motion
        self.pub_overlay_text.publish(text_msg)

        motion_msg = Motion()
        motion_msg.header = boxes_msg.header
        motion_msg.motion = self.segment_motion
        self.pub_motion.publish(motion_msg)

        self.prev_move_state = move_state

if __name__=='__main__':
    rospy.init_node('hand_motion_estimator')
    motion_estimator = HandMotionEstimator()
    rospy.spin()
