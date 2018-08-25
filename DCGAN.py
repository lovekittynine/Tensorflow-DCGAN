#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 15:24:56 2018

@author: wsw
"""

# train DCGAN

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from Dataset import make_train_dataset
slim = tf.contrib.slim

tf.reset_default_graph()

def generator(xs,is_training=True):
    
    batchnorm_params = {'decay':0.9,
                        'updates_collections':None,
                        'is_training':is_training,
                        'scale':True,
                        }
    
    with slim.arg_scope([slim.conv2d_transpose],
                        kernel_size=[4,4],
                        padding='SAME',
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batchnorm_params,
                        activation_fn=tf.nn.relu,
                        stride=2):
        # Note:stride is upsampling factor
        net = slim.conv2d_transpose(xs,num_outputs=1024,stride=4,scope='deconv1')
        # 4x4x1024->8x8x512
        net = slim.conv2d_transpose(net,num_outputs=512,scope='deconv2')
        # 8x8x512->16x16x256
        net = slim.conv2d_transpose(net,num_outputs=256,scope='deconv3')
        # 16x16x256->32x32x128
        net = slim.conv2d_transpose(net,num_outputs=128,scope='deconv4')
        # 32x32x128->64x64x64
        net = slim.conv2d_transpose(net,num_outputs=64,scope='deconv5')
        # 64x64x64->64x64x3
        net = slim.conv2d(net,num_outputs=3,kernel_size=[3,3],
                          padding='SAME',
                          activation_fn=tf.nn.tanh,
                          scope='conv6')
        return net


def discriminator(xs,is_training=True):
    
    batchnorm_params = {'decay':0.9,
                        'updates_collections':None,
                        'is_training':is_training
                        }
    
    with slim.arg_scope([slim.conv2d],
                        kernel_size=[3,3],
                        padding='SAME',
                        activation_fn=tf.nn.leaky_relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batchnorm_params,
                        stride=2,
                        ):
            # 32x32x64
            net = slim.conv2d(xs,num_outputs=64,normalizer_fn=None,scope='conv1')
            # 16x16x128
            net = slim.conv2d(net,num_outputs=128,scope='conv2')
            # 8x8x256
            net = slim.conv2d(net,num_outputs=256,scope='conv3')
            # 4x4x512
            net = slim.conv2d(net,num_outputs=512,scope='conv4')
            # resize to 1x1
            net = slim.conv2d(net,num_outputs=512,stride=4,normalizer_fn=None,scope='conv5')
            net = slim.flatten(net,scope='flatten')
            net = slim.fully_connected(net,num_outputs=1,activation_fn=None,scope='fc6')
            return net


def DCGAN():
    
    # network config
    batchsize = 128
    learning_rate = 2e-4
    beta1 = 0.5
    # noise inputs
    z_inputs = tf.placeholder(tf.float32,shape=[batchsize,1,1,100])
    # encode = np.float32(np.random.randint(0,2,size=[batchsize,1,1,100]))
    # dataset
    dataset = make_train_dataset()
    train_iter = dataset.make_one_shot_iterator()
    image_batch = train_iter.get_next()
    
    # Generator
    with tf.variable_scope('Generator',reuse=tf.AUTO_REUSE):
        # generate images for train
        gen_images_train = generator(z_inputs,is_training=True)
        # generate images for test
        gen_images_test = generator(z_inputs,is_training=False)
        
    # Discriminator
    with tf.variable_scope('Discriminator',reuse=tf.AUTO_REUSE):
        # train for discriminator
        d_fake = discriminator(gen_images_train)
        d_real = discriminator(image_batch)
        # train for generator
        # g_fake = discriminator(gen_images_train,is_training=False)
        
    # get G_vars and D_vars
    G_vars = slim.get_trainable_variables(scope='Generator')
    D_vars = slim.get_trainable_variables(scope='Discriminator')
    
    # model summary
    G_varNums,G_varBytes = slim.model_analyzer.analyze_vars(G_vars,print_info=False)
    print('Generator trainable variable Nums:',G_varNums,'variable Bytes',G_varBytes)
    D_varNums,D_varBytes = slim.model_analyzer.analyze_vars(D_vars,print_info=False)
    print('Discriminator trainable variable Nums:',D_varNums,'variable Bytes',D_varBytes)
    
  
    with tf.name_scope('G_loss'):
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake,
                                                         labels=tf.ones_like(d_fake)))
    with tf.name_scope('D_loss'):
        d_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake,
                                                              labels=tf.zeros_like(d_fake))
        d_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real,
                                                              labels=tf.ones_like(d_real))
        d_loss = tf.reduce_mean(d_fake_loss+d_real_loss)
        
    with tf.name_scope('optimizer'):
        global_step = tf.train.create_global_step()
        G_optimizer = tf.train.AdamOptimizer(learning_rate,beta1=beta1)
        D_optimizer = tf.train.AdamOptimizer(learning_rate,beta1=beta1)
        train_D = D_optimizer.minimize(d_loss,global_step,var_list=D_vars)
        train_G = G_optimizer.minimize(g_loss,var_list=G_vars)
        
    with tf.name_scope('accuracy'):
        predict_label = tf.where(tf.nn.sigmoid(d_fake)>0.5,tf.ones_like(d_fake),tf.zeros_like(d_fake))
        gt_label = tf.ones_like(d_fake)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predict_label,gt_label),dtype=tf.float32))
        
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord)
        try:
            epoch=1
            while not coord.should_stop():
                noise = np.random.normal(size=[batchsize,1,1,100]).astype(np.float32)
                # train discriminator
                d_loss_value,accu,_ = sess.run([d_loss,accuracy,train_D],
                                               feed_dict={z_inputs:noise})
                # train generator
                for i in range(2):
                    g_loss_value,_ = sess.run([g_loss,train_G],
                                              feed_dict={z_inputs:noise})
                step = global_step.eval()
                fmt = 'Epoch[{:03d}]-Step:{:05d}-G_loss:{:.3f}-D_loss:{:.3f}-D_accu:{:.3f}'.\
                format(epoch,step,d_loss_value,g_loss_value,accu)
                
                if step%10 == 0:
                    print(fmt)
                    
                if step%400==0:
                    epoch += 1
                    generate_imgs = sess.run(gen_images_test,
                                             feed_dict={z_inputs:noise})
                    save_img(epoch,generate_imgs)
        except tf.errors.OutOfRangeError:
            print('train finished!!!')
            coord.request_stop()
        coord.join(threads)
    
                    
def save_img(epoch,imgs):
    fig,axes = plt.subplots(5,5)
    for i in range(5):
        for j in range(5):
            axes[i,j].imshow((imgs[i*5+j]+1.0)/2)
            axes[i,j].axis('off')
    if not os.path.exists('./images'):
        os.makedirs('./images')
    fig.savefig('./images/%d.png'%epoch)
    plt.close()
    
if __name__ == '__main__':
    DCGAN()
