# -*- coding:gb2312 -*-
import tensorflow as tf
import sys
import os

from Sal3DFCN import Model

from network import *
from datetime import datetime
import cv2

def next_batch(num,frames,groupNo):
    global Data  #(400, 200, 200, 3)
    # global test_data
    global GroundTruth  #(400, 200, 200)
    # TestFrames=3455
    TestFrames = 1066
    # TestFrames = 3550 # Test_USVD
    FRAMES = TestFrames - frames
    GROUP=FRAMES/num   #591

    test_data_batch = np.zeros([batch_szie, frames, 224, 224, 3])
    test_data_batch_vgg = np.zeros([batch_szie, 224, 224, 3])
    test_label_batch = np.zeros([batch_szie, 1, 224, 224])
    bthGroup=np.zeros([GROUP,num,frames])

    fetch_frames = np.zeros([FRAMES, frames])
    start = 0
    for frame in range(FRAMES):
        batch_frames = range(start, start + frames)
        # print batch_frames[0],batch_frames[1],batch_frames[2]
        if batch_frames[frames-1] < TestFrames:
            for frm in range(frames):
                fetch_frames[start, frm] = batch_frames[frm]
            start = start + 1

    frm0=0
    for group in range(GROUP):
            for n in range(num):
                bthGroup[group,n,:]=fetch_frames[frm0,:]
                frm0=frm0+1

    for n in range(num):
        for f in range(frames):
            index = int(bthGroup[groupNo,n,f])
            # print index
            test_data_batch[n, f, :, :, :] = Data[index, :, :, :]
        test_data_batch_vgg[n, :, :, :]=test_data_batch[n, 1, :, :,:]

    return test_data_batch,test_data_batch_vgg,test_label_batch

print("loading data and lable ... ")

DataPath="./Datasets/Test_SegTrac/Test_SegTrac_VideoData.npy"
GroundTruthPath="./Datasets/Test_SegTrac/Test_SegTrac_VideoGroundTruth.npy"

Data=np.load(DataPath)
GroundTruth=np.load(GroundTruthPath)
print(Data.shape,GroundTruth.shape)
print("data and lable loaded")

# TestFrames=3455  #DAVIS
TestFrames = 1066  #Seg
# TestFrames = 3550 # Test_USVD
batch_szie=10 # train images=batch_size*frames
frames=3 # 3 frames

with tf.name_scope('inputs'):
    x=tf.placeholder(tf.float32,[batch_szie,frames,224,224,3])
    y=tf.placeholder(tf.float32,[None,1,224,224])
    keep_var=tf.placeholder(tf.float32)

pred=Model.fcn3d(x,keep_var)

saver = tf.train.Saver()

with tf.Session() as sess:

    if int((tf.__version__).split('.')[1]) < 12:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()

    print ('Resore Model...')
    # sess.run(init)
    saver.restore(sess, "myNet/saliency_haveVgg_7_3000_2.ckpt")
    print ('Start testing...')
    # step=1
    group=0
    # batches=3551/(batch_szie*frames)
    FRAMES=TestFrames-frames
    GROUP=FRAMES/batch_szie

    No=2
    while group<GROUP:

        test_batch_xs,test_data_batch_vgg,test_batch_ys=next_batch(batch_szie,frames,group)
        image=sess.run(pred,feed_dict={x:test_batch_xs,keep_var:0.5})
        # image=tf.reshape(pred,[batch_szie,frames,224,224])
        image1=np.reshape(image,[-1,1,224,224])
        image2 = test_data_batch_vgg[-1, :, :, :]

        # i=7
        for num in range(batch_szie):
            # for i in range(frames):
            save_image1=np.zeros([224,224])
            save_image2 = np.zeros([224, 224, 3])
            save_image1[:,:]=image1[num,0,:,:]
            # save_image2[:, :, :] = image2[num,:, :, :]
            final_save_image1=save_image1*255
            final_save_image2 = save_image2
            cv2.imwrite('/home/guanqun/Result1/' + str(No) + '.jpg', final_save_image1)
            No=No+1

        #print training step
        group = group+1
        print ('Testing group:', group)
        print ('========================')

    print ('Testing Finished!')
