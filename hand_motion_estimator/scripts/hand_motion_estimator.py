#!/usr/bin/env python

import os
import csv
import math
from enum import Enum

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import rospy
import rospkg
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox

class HandMotionEstimator():

    def __init__(self):
        self.state = Enum('state', 'buffered not_buffered isnan')

        self.chunk_size = rospy.get_param('~chunk_size', 3)
        self.angle_buf_size = rospy.get_param('~angle_buf_size', 10)
        self.bin_size = rospy.get_param('~bin_size', 18)
        self.histogram = [0 for i in range(self.bin_size)]

        self.flow_chunk = []
        self.vec_pair = []
        self.angle_buf = []

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
        self.generate_model(self.data_path)

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


    def make_flow_chunk(self, pos):
        if len(self.flow_chunk) < self.chunk_size:
            self.flow_chunk.append(pos)
            # 0 means waiting next position to buffer positions
            return self.state.not_buffered
        else:
            self.flow_chunk.pop(0)
            self.flow_chunk.append(pos)
            # 1 means done buffer positions
            return self.state.buffered

    def calc_pca(self):
        self.pca.fit(np.array(self.flow_chunk))
        pca_vec = self.pca.components_[0]
        return np.array(pca_vec)

    def make_vec_pair(self, vec):
        if len(self.vec_pair) < 2:
            self.vec_pair.append(vec)
            return self.state.not_buffered
        else:
            self.vec_pair.pop(0)
            self.vec_pair.append(vec)
            return self.state.buffered

    def make_angle_buffer(self):
        cross = np.cross(self.vec_pair[0], self.vec_pair[1])
        dot = np.dot(self.base_axis, cross)
        angle = np.arccos(
            dot / (np.linalg.norm(cross) * np.linalg.norm(self.base_axis)))
        angle = np.rad2deg(angle)

        if np.isnan(angle):
            return self.state.isnan

        if len(self.angle_buf) < self.angle_buf_size:
            self.angle_buf.append(angle)
            return self.state.not_buffered
        else:
            self.angle_buf.pop(0)
            self.angle_buf.append(angle)
            return self.state.buffered

    def make_histogram(self):
        hist = [0 for i in range(self.bin_size)]
        for angle in self.angle_buf:
            idx = int(math.floor(angle / 10.))
            hist[idx] += 1
        self.histogram = np.array(hist) / float(np.array(hist).max())

    def classify_motion(self, target_data):
        if self.classifier is None:
            return None

        motion_class = self.classifier.predict(np.array([target_data]))
        motion_class = int(motion_class[0])
        return self.motion_lst[motion_class]

    def callback(self, boxes_msg):
        print(boxes_msg.header.stamp)
        if len(boxes_msg.boxes) > 1:
            rospy.logwarn('this node requre input boxes size is 1')
            return

        finger_box = boxes_msg.boxes[0]
        pos = np.array([finger_box.pose.position.x,
                        finger_box.pose.position.y,
                        finger_box.pose.position.z])

        if self.make_flow_chunk(pos) == self.state.not_buffered:
            rospy.loginfo('flow_chunk state: %s' %(self.state.not_buffered.name))
            return

        pca_vec = self.calc_pca()
        if self.make_vec_pair(pca_vec) == self.state.not_buffered:
            rospy.loginfo('vec_pair state: %s' %(self.state.not_buffered.name))
            return

        if self.make_angle_buffer() == self.state.not_buffered:
            rospy.loginfo('angle_buf state: %s' %(self.state.not_buffered.name))
            return
        elif self.make_angle_buffer() == self.state.isnan:
            rospy.logwarn('angle_buf state: %s' %(self.state.isnan.name))
            return

        self.make_histogram()
        motion = self.classify_motion(self.histogram)
        print(motion)

        vis_hist = np.array(self.histogram)
        plt.cla()
        plt.bar([i for i in range(vis_hist.shape[0])], vis_hist)
        plt.pause(.001)

if __name__=='__main__':
    rospy.init_node('hand_motion_estimator')
    motion_estimator = HandMotionEstimator()
    rospy.spin()