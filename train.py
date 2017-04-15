import scipy.misc as misc
import time
import tensorflow as tf
from architecture import netD, netG, pullaway_loss
import numpy as np
import random
import ntpath
import sys
import cv2
import os
from skimage import color
import argparse
import data_ops

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--DATASET',    required=True,help='The DATASET to use')
   parser.add_argument('--DATA_DIR',   required=True,help='Directory where data is')
   parser.add_argument('--BATCH_SIZE', required=True,help='Batch size',type=int)
   a = parser.parse_args()

   DATASET        = a.DATASET
   DATA_DIR       = a.DATA_DIR
   BATCH_SIZE     = a.BATCH_SIZE
   CHECKPOINT_DIR = 'checkpoints/'+DATASET+'/'
   IMAGES_DIR     = CHECKPOINT_DIR+'images/'

   try: os.mkdir('checkpoints/')
   except: pass
   try: os.mkdir(CHECKPOINT_DIR)
   except: pass
   try: os.mkdir(IMAGES_DIR)
   except: pass
   
   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   z           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 100), name='z')

   train_images_list = data_ops.loadData(DATA_DIR, DATASET)
   filename_queue    = tf.train.string_input_producer(train_images_list)
   real_images       = data_ops.read_input_queue(filename_queue, BATCH_SIZE)

   # generated images
   gen_images = netG(z, BATCH_SIZE)

   errD_real, embeddings_real, decoded_real = netD(real_images, BATCH_SIZE)
   errD_fake, embeddings_fake, decoded_fake = netD(gen_images, BATCH_SIZE)

   # cost functions
   #errD = tf.reduce_mean(errD_real - errD_fake)
   margin = 20
   errD = margin - errD_fake+errD_real
   pt_loss = pullaway_loss(embeddings_fake, BATCH_SIZE)
   errG = errD_fake# + 0.1*pt_loss

   # tensorboard summaries
   tf.summary.scalar('d_loss', errD)
   tf.summary.scalar('g_loss', errG)
   merged_summary_op = tf.summary.merge_all()

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]

   # optimize G
   G_train_op = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5).minimize(errG, var_list=g_vars, global_step=global_step)

   # optimize D
   D_train_op = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5).minimize(errD, var_list=d_vars)

   saver = tf.train.Saver(max_to_keep=1)
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess  = tf.Session()
   sess.run(init)

   summary_writer = tf.summary.FileWriter(CHECKPOINT_DIR+'/'+'logs/', graph=tf.get_default_graph())

   tf.add_to_collection('G_train_op', G_train_op)
   tf.add_to_collection('D_train_op', D_train_op)
   
   # restore previous model if there is one
   ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         pass
   
   
   step    = sess.run(global_step)
   coord   = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(sess, coord=coord)

   while True:
      
      start = time.time()

      batch_z = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
      sess.run(D_train_op, feed_dict={z:batch_z})
      sess.run(G_train_op, feed_dict={z:batch_z})

      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op], feed_dict={z:batch_z})
      summary_writer.add_summary(summary, step)

      print 'step:',step,'D loss:',D_loss,'G_loss:',G_loss,'time:',time.time()-start
      step += 1
    
      if step%500 == 0:
         print 'Saving model...'
         saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
         saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')
         batch_z  = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
         gen_imgs = sess.run([gen_images], feed_dict={z:batch_z})

         data_ops.saveImage(gen_imgs[0], step, IMAGES_DIR)
         print 'Done saving'



