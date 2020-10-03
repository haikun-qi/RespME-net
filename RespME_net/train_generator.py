"""Part of the training engine related to Python generators of array data.
"""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import warnings
import numpy as np
import os
import random
from keras.engine.training_utils import iter_sequence_infinite
from keras.utils.data_utils import Sequence
from keras.utils.data_utils import GeneratorEnqueuer
from keras.utils.data_utils import OrderedEnqueuer
import scipy.io as sio

import tensorflow as tf

import model_flow as mm
from image_warp import image_warp



def restore_networks(sess, ckpt_dir, ckpt):

    if ckpt is not None:
        saver = tf.train.import_meta_graph(ckpt_dir + 'model.meta')
        saver.restore(sess, ckpt.model_checkpoint_path)
        saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
        print(ckpt.all_model_checkpoint_paths[-1])
    else:
        sessFileName = ckpt_dir+'model'
        saver = tf.train.Saver(max_to_keep=100)
        sess.run(tf.global_variables_initializer())
        savedFile = saver.save(sess, sessFileName)
        print("Model meta graph saved in::%s" % savedFile)

    return saver


def my_fit_generator(params,
                     generator,
                     ckpt_dir,
                     val_data_list = None,
                     learning_rate=1e-3,
                     lr_decay_step = 1.0,
                     lr_decay_rate = 1.0,
                     epochs = 1,
                     max_queue_size=10,
                     workers=1,
                     use_multiprocessing=False,
                     shuffle=True,
                     save_step = 1,
                     initial_epoch = 0):
    """See docstring for `Model.fit_generator`."""

    wait_time = 0.01  # in seconds
    epoch = initial_epoch


    is_sequence = isinstance(generator, Sequence)
    if not is_sequence and use_multiprocessing and workers > 1:
        warnings.warn(
            UserWarning('Using a generator with `use_multiprocessing=True`'
                        ' and multiple workers may duplicate your data.'
                        ' Please consider using the`keras.utils.Sequence'
                        ' class.'))

    if is_sequence:
        steps_per_epoch = len(generator)
    else:
        raise ValueError('`steps_per_epoch=None` is only valid for a'
                         ' generator based on the '
                         '`keras.utils.Sequence`'
                         ' class. Please specify `steps_per_epoch` '
                         'or use the `keras.utils.Sequence` class.')

    enqueuer = None

    try:
        if workers > 0:
            if is_sequence:
                enqueuer = OrderedEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing,
                    shuffle=shuffle)
            else:
                enqueuer = GeneratorEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing,
                    wait_time=wait_time)
            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()
        else:
            if is_sequence:
                output_generator = iter_sequence_infinite(generator)
            else:
                output_generator = generator



        # for test graph
        tf.reset_default_graph()
        input_shape = (None, 64, 64, 64, 1)
        Iref = tf.placeholder(tf.float32, shape=input_shape, name='Iref')
        Imov = tf.placeholder(tf.float32, shape=input_shape, name='Imov')

        out = mm.flownet(Iref, Imov)
        flowTst = tf.identity(tf.squeeze(out), name='flowTst')

        Iwarp = image_warp(Imov, out)
        Iwarp = tf.identity(tf.squeeze(Iwarp), name='Iwarp')

        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        sessFileNameTst = ckpt_dir + 'modelTst'
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            savedFile = saver.save(sess, sessFileNameTst, latest_filename='checkpointTst')
        print('testing model saved:' + savedFile)

        # %%   construct training map
        tf.reset_default_graph()
        input_shape = (params['batch_size'],)+( 64, 64, 64, 1)
        Iref = tf.placeholder(tf.float32, shape=input_shape, name='Iref')  # [batch, nx, ny, nz, channel]
        Imov = tf.placeholder(tf.float32, shape=input_shape, name='Imov')  # [batch, nx, ny, nz]

        flows, out, Iref_out, Imov_out, border_mask = mm.flownet(Iref, Imov, training=True, augment=True)
        # flowT = tf.identity(tf.squeeze(out), name='flowT')

        lr = tf.placeholder(tf.float32, name='learning_rate')
        loss = mm.unsupervised_loss(flows, Iref_out, Imov_out, border_mask, params)
        tf.summary.scalar('loss', loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            gvs = optimizer.compute_gradients(loss)
            # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            # opToRun=optimizer.apply_gradients(capped_gvs)
            opToRun = optimizer.apply_gradients(gvs)

        # %% training code
        totalLoss, ep, eval_Loss = [], 0, []
        lossT = tf.placeholder(tf.float32)
        lossE = tf.placeholder(tf.float32)

        lossSumT = tf.summary.scalar("TrnLoss", lossT)
        lossSumE = tf.summary.scalar("TestLoss", lossE)

        sessFileName = ckpt_dir + 'model'

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            saver = restore_networks(sess, ckpt_dir, ckpt)
            writer = tf.summary.FileWriter(ckpt_dir, sess.graph)

            while epoch < epochs:

                if (epoch+1)%lr_decay_step ==0:
                    learning_rate = learning_rate / lr_decay_rate

                steps_done = 0
                while steps_done < steps_per_epoch:
                    generator_output = next(output_generator)
                    img_ref = generator_output[...,0][...,np.newaxis]
                    img_mov = generator_output[...,1][...,np.newaxis]
                    feed_dict = {lr: learning_rate, Iref: img_ref, Imov:img_mov}

                    _,_, trn_loss = sess.run(
                        [opToRun,update_ops, loss],
                        feed_dict=feed_dict)
                    steps_done += 1

                    if (epoch * steps_per_epoch + steps_done) % params['display_interval']==0 or (epoch==0 and steps_done==1):
                        ep = ep + 1
                        lossSum = sess.run(lossSumT, feed_dict={lossT: trn_loss})
                        writer.add_summary(lossSum, ep)

                        if val_data_list is not None:
                            eval_data = get_eval(val_data_list)
                            img_ref = eval_data[...,0][...,np.newaxis]
                            img_mov = eval_data[...,1][...,np.newaxis]
                            tst_loss = sess.run(loss,feed_dict={Iref: img_ref, Imov:img_mov})
                            writer.add_summary(sess.run(lossSumE, feed_dict={lossE:tst_loss}), ep)

                            print("-- train: epoch = {}, steps_done/steps per epoch = {}/{}, Train loss = {}, Test loss = {}"
                                  .format(epoch + 1, steps_done, steps_per_epoch, trn_loss, tst_loss))
                        else:
                            print(
                                "-- train: epoch = {}, steps_done/steps per epoch = {}/{}, Train loss = {}"
                                .format(epoch + 1, steps_done, steps_per_epoch, trn_loss))

                saver.save(sess, sessFileName, global_step=epoch, write_meta_graph=True)
                epoch += 1

            # Epoch finished.
            writer.close()

    finally:
        if enqueuer is not None:
            enqueuer.stop()


def get_eval(val_data_list, batch_size=16, dims=(64,64,64)):
    crop_size = np.array([192, 192, 80])
    X = np.empty((batch_size,)+ dims+ (2,), dtype='float32')
    filename=random.choice(val_data_list)
    mat_contents = sio.loadmat(filename)
    x_data_ori = mat_contents['msense']
    x_data_ori = np.array(x_data_ori)
    rand_bxyz = [[]]
    i=0
    while(i<batch_size):
        bin_num = random.choice([1, 2, 3])
        xx = np.random.choice(range(crop_size[0] - dims[0] + 1))
        yy = np.random.choice(range(crop_size[1] - dims[1] + 1))
        zz = np.random.choice(range(crop_size[2] - dims[2] + 1))
        ### load data & normalization
        if ([bin_num, xx, yy, zz] not in rand_bxyz):
            im1 = x_data_ori[:, :, :, 0]
            im2 = x_data_ori[:, :, :, bin_num]
            im1 = (im1 - np.min(im1)) / (np.max(im1) - np.min(im1))
            im2 = (im2 - np.min(im2)) / (np.max(im2) - np.min(im2))
            x_data = [im1, im2]
            x_data = np.array(x_data)

            ### crop to the central part
            x_data = np.transpose(x_data, [1, 2, 3, 0])
            x_shape = np.shape(x_data)[0:-1]
            x_shape = np.array(x_shape)
            limit = x_shape - crop_size + 1
            stx = limit[0] //2
            sty = limit[1] // 2
            stz = limit[2] //2
            x_data = x_data[stx:stx + crop_size[0], sty:sty + crop_size[1], stz:stz + crop_size[2], :]

            ### get the patch
            X[i,] = x_data[xx:xx + dims[0], yy:yy + dims[1], zz:zz + dims[2], :]
            rand_bxyz+=[[bin_num,xx,yy,zz]]
            i+=1
    return X

