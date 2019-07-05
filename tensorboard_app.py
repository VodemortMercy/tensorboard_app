# coding=utf8

import tensorflow as tf
import datetime
import numpy as np
import time


class TensorboardApp:

    def __init__(self):
        self.mysession = tf.Session()
        self.logdir = './mytensorboard'
        self.writer = tf.summary.FileWriter(self.logdir, self.mysession.graph)
        self.loss = tf.placeholder(tf.float32,[])
        self.image1 = tf.placeholder(tf.float32, shape=(1, 96, 96, 1))
        self.image2 = tf.placeholder(tf.float32, shape=(1, 96, 96, 1))
        self.image3 = tf.placeholder(tf.float32, shape=(1, 96, 96, 1))
        tf.summary.scalar('loss', self.loss)
        tf.summary.image('image1', self.image1)
        tf.summary.image('image2', self.image2)
        tf.summary.image('image3', self.image3)
        self.merged = tf.summary.merge_all()
        print('tensorboard app started')

    def fill_in_data(self, data, index):
        summary = self.mysession.run(self.merged, feed_dict={self.loss: data['scalar'],
                                                             self.image1: data['image1'],
                                                             self.image2: data['image2'],
                                                             self.image3: data['image3']})
        self.writer.add_summary(summary, index)
        self.writer.flush()
