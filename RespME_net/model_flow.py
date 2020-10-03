
from keras import backend as Ks
from keras.layers import Concatenate, Conv3D, BatchNormalization, Conv3DTranspose
from keras.optimizers import *
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU
from loss_fn import compute_losses, create_border_mask
from augment import random_affine
from util import downsample, resize_3D

Ks.set_image_data_format('channels_last')

BATCH_NORM = True


def _leaky_relu(x):
    with tf.variable_scope('leaky_relu'):
        return LeakyReLU(0.1)(x)
        # return tf.maximum(0.1 * x, x)

def flownet(im1, im2, channel_mult=1, training=False, full_res=False,augment=False):
    """Given two images, returns flow predictions in decreasing resolution.
    """


    m = channel_mult

    with tf.name_scope('flownet'):
        # -------------------------------------------------------------------------
        # Data & mask augmentation
        border_mask = create_border_mask(im1, 0.02)
        if augment:
            [im1, im2,mask_local] = random_affine(
                [im1, im2, border_mask],
                max_translation_x=1.0, max_translation_y=2.0, max_translation_z=2.0,
                max_rot=10.0, horizontal_flipping=False,
                min_scale=0.8, max_scale=1.2)
            border_mask = mask_local * border_mask


        with tf.variable_scope('Wts', reuse=tf.AUTO_REUSE):

            im1_max = tf.reduce_max(im1, axis=[1, 2, 3, 4], keepdims=True)
            im2_max = tf.reduce_max(im2, axis=[1, 2, 3, 4], keepdims=True)
            im1, im2 = im1 / im1_max, im2 / im2_max
            im1 = tf.stop_gradient(im1)
            im2 = tf.stop_gradient(im2)

            # feature extraction
            Conv1 = Conv3D(int(64*m), 5, strides = 2, activation=None,
                           padding='same', kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(1e-4))
            Conv2 = Conv3D(int(128*m), 3, strides = 2, activation=None,
                           padding='same', kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(1e-4))
            Conv3 = Conv3D(int(128 * m), 3, strides=1, activation=None,
                           padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4))
            # for im1
            conv1_1 = Conv1(im1)
            if BATCH_NORM:
                conv1_1 = BatchNormalization(axis=-1)(conv1_1)
            conv1_1 = _leaky_relu(conv1_1)
            conv2_1 = Conv2(conv1_1)
            if BATCH_NORM:
                conv2_1 =BatchNormalization(axis=-1)(conv2_1)
            conv2_1 = _leaky_relu(conv2_1)
            conv3_1 = Conv3(conv2_1)
            if BATCH_NORM:
                conv3_1 = BatchNormalization(axis=-1)(conv3_1)
            conv3_1 = _leaky_relu(conv3_1)
            # for im2
            conv1_2 = Conv1(im2)
            if BATCH_NORM:
                conv1_2 = BatchNormalization(axis=-1)(conv1_2)
            conv1_2 = _leaky_relu(conv1_2)
            conv2_2 = Conv2(conv1_2)
            if BATCH_NORM:
                conv2_2 = BatchNormalization(axis=-1)(conv2_2)
            conv2_2 = _leaky_relu(conv2_2)
            conv3_2 = Conv3(conv2_2)
            if BATCH_NORM:
                conv3_2 = BatchNormalization(axis=-1)(conv3_2)
            conv3_2 = _leaky_relu(conv3_2)

            # encoding part
            conv_concat = Conv3D(int(128 * m), 3,strides=1, activation=None,
                           padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.l2(1e-4))(Concatenate(axis=-1)([conv3_1, conv3_2]))
            if BATCH_NORM:
                conv_concat = BatchNormalization(axis=-1)(conv_concat)
            conv_concat = _leaky_relu(conv_concat)

            conv4 = Conv3D(int(256 * m), 3, strides=2, activation=None,
                           padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.l2(1e-4))(conv_concat)
            if BATCH_NORM:
                conv4 = BatchNormalization(axis=-1)(conv4)
            conv4 = _leaky_relu(conv4)
            conv4_1 = Conv3D(int(256 * m), 3, strides=1, activation=None,
                           padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.l2(1e-4))(conv4)
            if BATCH_NORM:
                conv4_1 = BatchNormalization(axis=-1)(conv4_1)
            conv4_1 = _leaky_relu(conv4_1)

            conv5 = Conv3D(int(512 * m), 3, strides=2, activation=None,
                           padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.l2(1e-4))(conv4_1)
            if BATCH_NORM:
                conv5 = BatchNormalization(axis=-1)(conv5)
            conv5 = _leaky_relu(conv5)
            conv5_1 = Conv3D(int(512 * m), 3, strides=1, activation=None,
                           padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.l2(1e-4))(conv5)
            if BATCH_NORM:
                conv5_1 = BatchNormalization(axis=-1)(conv5_1)
            conv5_1 = _leaky_relu(conv5_1)

            # decoding part
            res = _flownet_upconv(conv5_1, conv4_1, conv_concat, conv1_1,
                                  channel_mult=channel_mult, full_res=full_res)

            if full_res:
                final_flow_fw = res[0]
            else:
                down_factor = 2
                _, height, width, thick, _ = im1.shape.as_list()
                final_flow_fw = resize_3D(res[0], [height, width, thick]) * down_factor
                res = [final_flow_fw] + res
            if training:
                return res, final_flow_fw, im1, im2, border_mask # for loss calculation
            else:
                return final_flow_fw




def _flownet_upconv(conv5_1, conv4_1, conv3_1, conv1_a,
                    channel_mult=1, full_res=False, channels=3):
    with tf.variable_scope('flownet_upconv'):
        m = channel_mult
        flow5 = Conv3D(channels, 3, activation=None,
                           padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.l2(1e-4))(conv5_1)  # /16

        deconv4 = Conv3DTranspose(int(256 * m), 4, strides=2,activation=None,
                           padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.l2(1e-4))(conv5_1)
        if BATCH_NORM:
            deconv4 = BatchNormalization(axis=-1)(deconv4)
        deconv4 = _leaky_relu(deconv4)

        flow5_up4 = Conv3DTranspose(channels, (4,4,4), strides=(2,2,2),activation=None,
                           padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.l2(1e-4))(flow5)

        concat4 = Concatenate(axis=-1)([conv4_1, deconv4, flow5_up4])
        flow4 = Conv3D(channels, 3, strides=1,activation=None,
                           padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.l2(1e-4))(concat4)  #/8

        deconv3 = Conv3DTranspose(int(128 * m), 4, strides=2, activation=None,
                           padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.l2(1e-4))(concat4)
        if BATCH_NORM:
            deconv3 = BatchNormalization(axis=-1)(deconv3)
        deconv3 = _leaky_relu(deconv3)

        flow4_up3 = Conv3DTranspose(channels, 4, strides=2,activation=None,
                           padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.l2(1e-4))(flow4)

        concat3 = Concatenate(axis=-1)([conv3_1, deconv3, flow4_up3])
        flow3 = Conv3D(channels, 3, strides=1, activation=None,
                           padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.l2(1e-4))(concat3) # /4

        deconv2 = Conv3DTranspose(int(64 * m), 4, strides=2,activation=None,
                           padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.l2(1e-4))(concat3)
        if BATCH_NORM:
            deconv2 = BatchNormalization(axis=-1)(deconv2)
        deconv2 = _leaky_relu(deconv2)

        flow3_up2 = Conv3DTranspose(channels, 4, strides=2,activation=None,
                           padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.l2(1e-4))(flow3)

        concat2 = Concatenate(axis=-1)([conv1_a, deconv2, flow3_up2])
        flow2 = Conv3D(channels, 3, strides=1,activation=None,
                           padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.l2(1e-4))(concat2)  # /2

        flows = [flow2, flow3, flow4, flow5]

        if full_res:
            with tf.variable_scope('full_res'):
                deconv1 = Conv3DTranspose(int(32 * m), 4, strides=2,activation=None,
                           padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.l2(1e-4))(concat2)
                if BATCH_NORM:
                    deconv1 = BatchNormalization(axis=-1)(deconv1)
                deconv1 = _leaky_relu(deconv1)

                flow2_up1 = Conv3DTranspose(channels, 4, strides=2,activation=None,
                           padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.l2(1e-4))(flow2)

                concat1 = Concatenate(axis=-1)([deconv1, flow2_up1])
                flow1 = Conv3D(channels, 3, activation=None,
                           padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.l2(1e-4))(concat1) # 1
                flows = [flow1] + flows

        return flows


def unsupervised_loss(flows_fw, im1, im2, border_mask, params, full_resolution=True):

    #########  loss calculation ##########
    LOSSES = ['smooth_2nd', 'ncc', 'gradient']
    loss_weights = dict()
    loss_weights['smooth_2nd']=params['smooth_2nd_weight']
    loss_weights['ncc'] = params['ncc_weight']
    loss_weights['gradient'] = params['gradient_weight']

    #

    if full_resolution:
        layer_weights = [650.0, 500.0, 250.0, 130.0, 70.0]
        mask_s = border_mask
        im1_s, im2_s = im1, im2
    else:
        layer_weights = [500.0, 250.0, 130.0, 70.0]
        down_factor = 2
        _,height, width, thick, _ = im1.shape.as_list()
        im1_s = downsample(im1, down_factor)
        im2_s = downsample(im2, down_factor)
        mask_s = downsample(border_mask, down_factor)
        flows_fw = flows_fw[1:]

    combined_loss = 0.0
    flow_enum = enumerate(flows_fw)

    for i, flow_fw_s in flow_enum:
        layer_name = "loss" + str(i + 1)
        flow_scale = 1.0 / (2 ** i)
        with tf.variable_scope(layer_name):
            layer_weight = layer_weights[i]

            losses = compute_losses(im1_s, im2_s, flow_fw_s * flow_scale, border_mask=mask_s)

            layer_loss = 0.0
            for loss in LOSSES:
                layer_loss += loss_weights[loss] * losses[loss]

            combined_loss += layer_weight * layer_loss
            im1_s = downsample(im1_s, 2)
            im2_s = downsample(im2_s, 2)
            mask_s = downsample(mask_s, 2)

    regularization_loss = tf.losses.get_regularization_loss()
    final_loss = combined_loss + 0.0001 * regularization_loss

    return final_loss