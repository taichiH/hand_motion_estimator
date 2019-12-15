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
from cv_bridge import CvBridge
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
        self.bin_size = rospy.get_param('~bin_size', 18)
        self.movement_thresh = rospy.get_param('~movement_thresh', 15)
        self.use_pca = rospy.get_param('~use_pca', False)
        self.pca_frame = rospy.get_param('~pca_frame', 4)
        self.interpolation_scale = rospy.get_param('~interpolation_scale', 2)

        self.histogram = [0 for i in range(self.bin_size)]

        self.flow_chunk = []
        self.vec_pair = []

        self.pca = PCA()
        self.base_axis = np.array([1,0,0])

        self.motion_lst = ['rot', 'trans', 'other']
        self.data_path = rospy.get_param(
            '~model_path',
            os.path.join(
                rospkg.RosPack().get_path('hand_motion_estimator'),
                'trained_data/sample.csv'))
        self.model = rospy.get_param('~model', 'mlp')
        self.classifier = None
        rospy.loginfo('data_path: %s' %(self.data_path))
        # self.generate_model(self.data_path)

        self.cv_bridge = CvBridge()
        self.pub_histogram_image = rospy.Publisher(
            "~output/histogram_image", Image, queue_size=1)
        self.pub_cross_histogram_image = rospy.Publisher(
            "~output/cross_histogram_image", Image, queue_size=1)
        self.pub_overlay_text = rospy.Publisher(
            "~output/overlay_text", OverlayText, queue_size=1)
        self.pub_movement = rospy.Publisher(
            "~output/movement", Float32, queue_size=1)
        self.pub_motion = rospy.Publisher(
            "~output/motion", Motion, queue_size=1)

        rospy.Service(
            "~save_histogram", GetHistogram, self.service_callback)

        rospy.Subscriber(
            "~input/hand_pose_box", BoundingBoxArray, self.callback)

    def generate_model(self, data_path):
        if self.model == 'random_forest':
            self.classifier = RandomForestClassifier(
                max_depth=2, random_state=0)
        elif self.model == 'mlp':
            self.classifier = MLPClassifier(
                activation='relu', alpha=0.0001, batch_size='auto',
                solver="adam", random_state=0, max_iter=10000,
                hidden_layer_sizes=(100,200,100),
                learning_rate='constant', learning_rate_init=0.001)
        else:
            rospy.logwarn('please set classification model')

        test_data = [] #x
        trained_data = [] #y
        labels = []
        with open(self.data_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                print(row)
                test_data.append(map(lambda x : float(x), row[1:]))
                trained_data.append(float(row[0]))

        self.classifier.fit(np.array(test_data), np.array(trained_data))


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

    def make_angle_buffer(self, chunk, cross_hist):
        angle_buf = []

        if self.use_pca:
            self.pca_frame = 4
        else:
            self.pca_frame = 0
        for i in range(2 + self.pca_frame, len(chunk)):
            if self.use_pca:
                vec1 = self.calc_pca(chunk[i-1-self.pca_frame:i-1])
                vec2 = self.calc_pca(chunk[i-self.pca_frame:i])
            else:
                vec1 = chunk[i-1] - chunk[i-2]
                vec2 = chunk[i]   - chunk[i-1]

            cross = None
            if cross_hist:
                cross = np.cross(vec1, vec2)
            else:
                cross = vec1

            dot = np.dot(self.base_axis, cross)
            angle = np.arccos(
                dot / (np.linalg.norm(cross) * np.linalg.norm(self.base_axis)))
            angle = np.rad2deg(angle)

            if np.isnan(angle):
                continue
            angle_buf.append(angle)

        return angle_buf

    def make_histogram(self, angle_buf):
        hist = [0 for i in range(self.bin_size)]
        for angle in angle_buf:
            idx = int(math.floor(angle / 10.))
            hist[idx] += 1
        histogram = np.array(hist) / float(np.array(hist).max())
        return histogram

    def classify_motion(self, target_data):
        if self.classifier is None:
            return None

        motion_class = self.classifier.predict(np.array([target_data]))
        motion_class = int(motion_class[0])
        return self.motion_lst[motion_class]

    def service_callback(self, req):
        res = GetHistogramResponse()

        save_data =  [req.label]  + list(self.histogram)
        print('save_data', save_data)

        with open(self.data_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(save_data)

        res.success = True
        res.histogram = self.histogram
        return res

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
            # rospy.loginfo('flow_chunk: %s' %(self.state.not_buffered.name))
            return
        elif interpolated_chunk == self.state.error:
            # rospy.logwarn('failed calc spline')
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

        # print('movement: ', movement)
        # print('most moved axis is %s' %(most_move_axis))
        # print('motion: ', motion)

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
