import tensorflow as tf
import numpy as np
import sys
from network import *
# from model2 import Model2

class Model:
    @staticmethod
    def fcn3d(_Y,_dropout):
        # TODO weight decay loss tern

        # _Y=[None, 3, 224, 224, 3]
        conv3d1_1 = conv3d(_Y, filters=[3, 3, 3, 3, 64], bias=[64], padding='SAME',strides=1,name='conv3d1_1')
        conv3d1_2 = conv3d(conv3d1_1, filters=[3, 3, 3, 64, 64], bias=[64], padding='SAME', strides=1, name='conv3d1_2')
        # 5 frames, patch 5x5, in size 3, outsize 64
        # after conv1_2 = [batch_size, 3, 224, 224, 64]
        pool3d1 = maxpool3d(conv3d1_2, depth_k=1,k=2,name='pool3d1')
        # norm1=lrn(pool3d1,2,2e-5,0.75,name='norm1')
        norm3d1 = tf.contrib.layers.batch_norm(pool3d1)
        # after pool, pool3d1 = [None, 3, 112, 112, 64]

        conv3d2_1 = conv3d(norm3d1,filters=[3, 3, 3, 64, 128],bias=[128],padding='SAME',strides=1,name='conv3d2_1')
        conv3d2_2 = conv3d(conv3d2_1, filters=[3, 3, 3, 128, 128], bias=[128], padding='SAME', strides=1, name='conv3d2_2')
        # 3 frames, patch 5x5, in size 64, outsize 128
        # after conv2 = [None, 3, 112, 112, 128]
        pool3d2 = maxpool3d(conv3d2_2, depth_k=1,k=2,name='pool3d2')
        # norm2 = lrn(pool3d2, 2, 2e-5, 0.75, name='norm2')
        norm3d2 = tf.contrib.layers.batch_norm(pool3d2)
        # after pool, pool2 = [None, 3, 56, 56, 128]

        conv3d3_1 = conv3d(norm3d2,filters=[3, 3, 3, 128, 256],bias=[256],padding='SAME',strides=1,name='conv3d3_1')
        conv3d3_2 = conv3d(conv3d3_1, filters=[3, 3, 3, 256, 256], bias=[256], padding='SAME', strides=1, name='conv3d3_2')
        # 2 frames, patch 5x5, in size 128, outsize 256
        # after conv3 = [None, 3, 56, 56, 256]
        pool3d3 = maxpool3d(conv3d3_2,depth_k=2, k=2,name='pool3d3')
        # norm3 = lrn(pool3d3, 2, 2e-5, 0.75, name='norm3')
        norm3d3 = tf.contrib.layers.batch_norm(pool3d3)
        # after pool, pool3 = [None, 2, 28, 28, 256]

        conv3d4_1 = conv3d(norm3d3,filters=[2, 3, 3, 256, 256],bias=[256],padding='SAME',strides=1,name='conv3d4_1')
        conv3d4_2 = conv3d(conv3d4_1, filters=[2, 3, 3, 256, 256], bias=[256], padding='SAME', strides=1, name='conv3d4_2')
        # 2 frames, patch 5x5, in size 256, outsize 256
        # after conv3 = [None, 2, 28, 28, 256]
        pool3d4 = maxpool3d(conv3d4_2, depth_k=1,k=2,name='pool3d4')
        # norm4 = lrn(pool3d4, 2, 2e-5, 0.75, name='norm4')
        norm3d4 = tf.contrib.layers.batch_norm(pool3d4)
        # after pool, pool4 = [None, 1, 14, 14, 256]

        conv3d5_1 = conv3d(norm3d4,filters=[1, 3, 3, 256, 512],bias=[512],padding='SAME',strides=1,name='conv3d5_2')
        conv3d5_2 = conv3d(conv3d5_1, filters=[1, 3, 3, 512, 512], bias=[512], padding='SAME', strides=1, name='conv3d5_3')
        # 2 frames, patch 5x5, in size 256, outsize 512
        # after conv3 = [None, 1, 14, 14, 512]
        pool3d5 = maxpool3d(conv3d5_2, depth_k=1,k=2,name='pool3d5')
        # norm5 = lrn(pool3d5, 2, 2e-5, 0.75, name='norm5')
        norm3d5 = tf.contrib.layers.batch_norm(pool3d5)
        # after pool, pool5 = [None, 1, 7, 7, 512]

        conv3d6_1=conv3d(norm3d5,filters=[1,3,3,512,512],bias=[512],padding='SAME',strides=1,name='conv3d6_1')
        conv3d6_2 = conv3d(conv3d6_1, filters=[1, 3, 3, 512, 1024], bias=[1024], padding='SAME', strides=1, name='conv3d6_2')
        # after conv6_2, conv6_2 = [None, 1, 7, 7, 1024]
        # norm6 = lrn(conv6_2, 2, 2e-5, 0.75, name='norm6')
        norm3d6 = tf.contrib.layers.batch_norm(conv3d6_2)

        ##############################################################################

        batch_size = tf.shape(_Y)[0]
        output_shape = [batch_size, 1, 14, 14, 1024]
        deconv3d1 = deconv3d(conv3d6_2, filters=[1, 3, 3, 512, 1024], bias=[1024], output_shape=output_shape,
                           strides=[1, 1, 2, 2, 1], padding='SAME',name="deconv3d1")
        # filter_shape: [depth, height, width, output_channels, in_channels]
        # deconv0_out_shape = [batch_size,1,14,14,256]
        # norm3d7_1 = tf.contrib.layers.batch_norm(concated_fcn_8_2)
        conv3d7_1 = conv3d(deconv3d1, filters=[1, 3, 3, 512, 512], bias=[512], padding='SAME',
                          strides=1, name='conv3d7_1')
        conv3d7_2 = conv3d(conv3d7_1, filters=[1, 3, 3, 512, 512], bias=[512], padding='SAME',
                          strides=1, name='conv3d7_2')
        conv3d7_3 = conv3d(conv3d7_2, filters=[1, 3, 3, 512, 512], bias=[512], padding='SAME',
                          strides=1, name='conv3d7_3')
        #in,out
        # norm7 = lrn(conv7_2, 2, 2e-5, 0.75, name='norm7')
        norm3d7 = tf.contrib.layers.batch_norm(conv3d7_3)

        batch_size = tf.shape(_Y)[0]  # batch_size shape
        output_shape = [batch_size, 1, 28, 28, 256]
        deconv3d2 = deconv3d(norm3d7, filters=[1, 3, 3, 256, 512], bias=[256],output_shape=output_shape,
                           strides=[1,1,2,2,1],padding='SAME',name='deconv3d2')
        # filter_shape: [depth, height, width, output_channels, in_channels]
        # 1 frames, patch 2x2, output_channel 256, in_channels 256
        # after deconv1 = [batch_size, 2, 14, 14, 256]
        conv3d8_1=conv3d(deconv3d2,filters=[1, 3, 3, 256, 256],bias=[256],padding='SAME',
                        strides=1,name='conv3d8_1')
        conv3d8_2=conv3d(conv3d8_1,filters=[1, 3, 3, 256, 256],bias=[256],padding='SAME',
                        strides=1,name='conv3d8_2')
        conv3d8_3=conv3d(conv3d8_2,filters=[1, 3, 3, 256, 256],bias=[256],padding='SAME',
                        strides=1,name='conv3d8_3')
        # norm8 = lrn(conv8_2, 2, 2e-5, 0.75, name='norm8')
        norm3d8=tf.contrib.layers.batch_norm(conv3d8_3)

        output_shape = [batch_size, 1, 56, 56, 128]
        deconv3d3 = deconv3d(norm3d8,filters=[1, 3, 3, 128, 256],bias=[128],output_shape=output_shape,
                           strides=[1,1,2,2,1],padding='SAME',name='deconv3d3')
        # filter_shape: [depth, height, width, output_channels, in_channels]
        # 2 frames, patch 3x3, output_channel 256, in_channels 256
        conv3d9_1=conv3d(deconv3d3,filters=[1, 3, 3, 128, 128],bias=[128],padding='SAME',
                        strides=1,name='conv3d9_1')
        conv3d9_2=conv3d(conv3d9_1,filters=[1, 3, 3, 128, 128],bias=[128],padding='SAME',
                        strides=1,name='conv9_2')
        # norm9 = lrn(conv9_2, 2, 2e-5, 0.75, name='norm9')
        norm3d9 = tf.contrib.layers.batch_norm(conv3d9_2)

        output_shape = [batch_size, 1, 112, 112, 64]
        deconv3d4 = deconv3d(norm3d9, filters=[1, 3, 3, 64, 128], bias=[64], output_shape=output_shape,
                           strides=[1, 1, 2, 2, 1], padding='SAME', name='deconv3d4')
        # filter_shape: [depth, height, width, output_channels, in_channels]
        # 2 frames, patch 3x3, output_channel 128, in_channels 256
        conv3d10_1=conv3d(deconv3d4,filters=[1, 3, 3, 64, 64],bias=[64],padding='SAME',
                        strides=1,name='conv3d10_1')
        conv3d10_2=conv3d(conv3d10_1,filters=[1, 3, 3, 64, 64],bias=[64],padding='SAME',
                        strides=1,name='conv3d10_2')
        # norm10 = lrn(conv10_2, 2, 2e-5, 0.75, name='norm10')
        norm3d10 = tf.contrib.layers.batch_norm(conv3d10_2)

        output_shape = [batch_size, 1, 224, 224, 32]
        deconv3d5 = deconv3d(norm3d10, filters=[1, 3, 3, 32, 64], bias=[32], output_shape=output_shape,
                           strides=[1, 1, 2, 2, 1], padding='SAME', name='deconv3d5')
        # filter_shape: [depth, height, width, output_channels, in_channels]
        # 3 frames, patch 3x3, output_channel 32, in_channels 64
        conv3d11_1 = conv3d(deconv3d5, filters=[1, 3, 3, 32, 32], bias=[32], padding='SAME',
                          strides=1, name='conv3d11_1')
        conv3d11_2 = conv3d(conv3d11_1, filters=[1, 3, 3, 32, 1], bias=[1], padding='SAME',
                          strides=1, name='conv3d11_2')
        # norm11 = lrn(conv11_2, 2, 2e-5, 0.75, name='norm11')
        norm3d11 = tf.contrib.layers.batch_norm(conv3d11_2)

        conv3d12_1 = tf.nn.conv3d(norm3d11, filter=tf.Variable(tf.truncated_normal([1, 3, 3, 1, 1])),
                                         strides=[1,1,1,1,1],padding='SAME')+tf.Variable(tf.constant(0.1,shape=[1]),name='conv3d12_1')
        # conv3d12_1=leaky_relu(conv3d12_1,leak=0.1,name='Lrelu')
        conv3d12_1 = tf.nn.relu(conv3d12_1)
        # conv3d12_1 = tf.nn.sigmoid(conv3d12_1,name='sigmoid')

        return conv3d12_1