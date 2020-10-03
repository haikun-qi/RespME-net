

import tensorflow as tf
import numpy as np
from image_warp import image_warp

def compute_losses(im1, im2, flow_fw, border_mask=None, flow_bw=None):

    losses = {}

    if flow_bw is None:

        im2_warped = image_warp(im2, flow_fw)
        if border_mask is None:
            mask_fw = create_outgoing_mask(flow_fw)
        else:
            mask_fw = border_mask

        losses['ncc'] = ncc_loss(im1, im2_warped, mask_fw)
        losses['smooth_2nd'] = second_order_loss(flow_fw, mask_fw)
        losses['gradient'] = gradient_loss(im1, im2_warped, mask_fw)

        return losses

    else:
        im2_warped = image_warp(im2, flow_fw)
        im1_warped = image_warp(im1, flow_bw)

        im_diff_fw = im1 - im2_warped
        im_diff_bw = im2 - im1_warped

        if border_mask is None:
            mask_fw = create_outgoing_mask(flow_fw)
            mask_bw = create_outgoing_mask(flow_bw)
        else:
            mask_fw = border_mask
            mask_bw = border_mask

        # flow_bw_warped = image_warp(flow_bw, flow_fw)
        # flow_fw_warped = image_warp(flow_fw, flow_bw)
        # flow_diff_fw = flow_fw + flow_bw_warped
        # flow_diff_bw = flow_bw + flow_fw_warped

        # losses['photo'] = (photometric_loss(im_diff_fw, mask_fw) +
        #                    photometric_loss(im_diff_bw, mask_bw))

        losses['gradient'] = (gradient_loss(im1, im2_warped, mask_fw) +
                              gradient_loss(im2, im1_warped, mask_bw))

        # losses['smooth_1st'] = (smoothness_loss(flow_fw) +
        #                         smoothness_loss(flow_bw))

        losses['smooth_2nd'] = (second_order_loss(flow_fw, mask_fw) +
                                second_order_loss(flow_bw, mask_bw))

        # losses['fb'] = (charbonnier_loss(flow_diff_fw, mask_fw) +
        #                 charbonnier_loss(flow_diff_bw, mask_bw))
        losses['ncc'] = ncc_loss(im1, im2_warped, mask_fw) + \
                        ncc_loss(im2, im1_warped, mask_bw)

        return losses






def photometric_loss(im_diff, mask):
    return charbonnier_loss(im_diff, mask, beta=255)


def conv2d(x, weights):
    return tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')

def conv3d(x, weights):
    return tf.nn.conv3d(x, weights, strides=[1, 1, 1, 1, 1], padding='SAME')


def l1_norm(x, mask=None):
    """Compute the l1-norm of tensor x.
    All positions where mask == 0 are not taken into account.

    Args:
        x: a tensor of shape [num_batch, height, width, depth, channels].
        mask: a mask of shape [num_batch, height, width, depth, mask_channels],
            where mask channels must be either 1 or the same number as
            the number of channels of x. Entries should be 0 or 1.
    Returns:
        loss as tf.float32
    """
    with tf.variable_scope('l1_norm'):
        x = tf.abs(x)
        if mask is not None:
            x = tf.multiply(x, mask)
        return tf.reduce_sum(x)


def _second_order_deltas(flow):
    with tf.variable_scope('_second_order_deltas'):

        weight_array=np.zeros([3, 3, 3, 1, 13])
        weight_array[0][0][0][0][0]=1
        weight_array[2][2][2][0][0] = 1
        weight_array[1][1][1][0][0] = -2

        weight_array[0][1][0][0][1] = 1
        weight_array[2][1][2][0][1] = 1
        weight_array[1][1][1][0][1] = -2

        weight_array[0][2][0][0][2] = 1
        weight_array[2][0][2][0][2] = 1
        weight_array[1][1][1][0][2] = -2

        weight_array[1][0][0][0][3] = 1
        weight_array[1][2][2][0][3] = 1
        weight_array[1][1][1][0][3] = -2

        weight_array[1][1][0][0][4] = 1
        weight_array[1][1][2][0][4] = 1
        weight_array[1][1][1][0][4] = -2

        weight_array[1][2][0][0][5] = 1
        weight_array[1][0][2][0][5] = 1
        weight_array[1][1][1][0][5] = -2

        weight_array[2][0][0][0][6] = 1
        weight_array[0][2][2][0][6] = 1
        weight_array[1][1][1][0][6] = -2

        weight_array[2][1][0][0][7] = 1
        weight_array[0][1][2][0][7] = 1
        weight_array[1][1][1][0][7] = -2

        weight_array[2][2][0][0][8] = 1
        weight_array[0][0][2][0][8] = 1
        weight_array[1][1][1][0][8] = -2

        weight_array[0][0][1][0][9] = 1
        weight_array[2][2][1][0][9] = 1
        weight_array[1][1][1][0][9] = -2

        weight_array[0][1][1][0][10] = 1
        weight_array[2][1][1][0][10] = 1
        weight_array[1][1][1][0][10] = -2

        weight_array[0][2][1][0][11] = 1
        weight_array[2][0][1][0][11] = 1
        weight_array[1][1][1][0][11] = -2

        weight_array[1][0][1][0][12] = 1
        weight_array[1][2][1][0][12] = 1
        weight_array[1][1][1][0][12] = -2

        weights = tf.constant(weight_array, dtype=tf.float32)

        flow_u, flow_v, flow_s= tf.split(axis=4, num_or_size_splits=3, value=flow)
        delta_u = conv3d(flow_u, weights)
        delta_v = conv3d(flow_v, weights)
        delta_s = conv3d(flow_s, weights)
        return delta_u, delta_v, delta_s


def second_order_loss(flow, mask=None):
    with tf.variable_scope('second_order_loss'):
        delta_u, delta_v, delta_s = _second_order_deltas(flow)
        loss_u = charbonnier_loss(delta_u, mask)
        loss_v = charbonnier_loss(delta_v, mask)
        loss_s = charbonnier_loss(delta_s, mask)
        return loss_u + loss_v + loss_s


def _gradient_delta(im1, im2_warped):
    with tf.variable_scope('gradient_delta'):
        channels = im1.shape.as_list()[-1]
        filter_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] # sobel filter
        filter_y = np.transpose(filter_x)
        filter_x=np.array(filter_x)
        filter_x =np.expand_dims(filter_x,2)
        filter_x = np.tile(filter_x,[1, 1, 3])

        filter_y=np.expand_dims(filter_y,2)
        filter_y = np.tile(filter_y,[1, 1, 3])

        filter_z = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] # sobel filter
        filter_z = np.array(filter_z)
        filter_z = np.expand_dims(filter_z,2)
        filter_z = np.transpose(filter_z,[2, 0, 1])
        filter_z = np.tile(filter_z,[3, 1, 1])

        weight_array = np.zeros([3, 3, 3, 1, 3])
        weight_array[:, :, :, 0, 0] = filter_x
        weight_array[:, :, :, 0, 1] = filter_y
        weight_array[:, :, :, 0, 2] = filter_z
        weight_array = np.tile(weight_array,[1,1,1,channels, 1])
        weights = tf.constant(weight_array, dtype=tf.float32)


        im1_grad = conv3d(im1, weights)
        im2_warped_grad = conv3d(im2_warped, weights)
        diff = im1_grad - im2_warped_grad
        return diff



def gradient_loss(im1, im2_warped, mask=None):
    with tf.variable_scope('gradient_loss'):
        mask_x = create_mask(im1, [[0, 0], [1, 1], [0,0]])
        mask_y = create_mask(im1, [[1, 1], [0, 0],[0,0]])
        mask_z = create_mask(im1, [[0,0],[0,0],[1,1]])

        gradient_mask = tf.concat(axis=4, values=[mask_x, mask_y, mask_z])
        diff = _gradient_delta(im1, im2_warped)
        return charbonnier_loss(diff, mask * gradient_mask)

def ncc_loss(im1, im2, mask=None):
    with tf.variable_scope('ncc_loss'):
        if mask is not None:
            im1 = tf.multiply(mask, im1)
            im2 = tf.multiply(mask, im2)

        batch, height, width, thick, channels = tf.unstack(tf.shape(im1))

        im1=tf.reshape(im1,[batch, channels*height*width*thick])
        im2=tf.reshape(im2,[batch, channels*height*width*thick])

        mean_im1 = tf.reduce_mean(im1,axis=1)
        mean_im1 = mean_im1[:, tf.newaxis]
        # mean_im1 = tf.tile(mean_im1, [1, height*width*thick*channels])

        mean_im2 = tf.reduce_mean(im2,axis=1)
        mean_im2 = mean_im2[:, tf.newaxis]
        # mean_im2 = tf.tile(mean_im2, [1, height * width * thick*channels])

        im1 = im1 - mean_im1
        im2 = im2 - mean_im2

        ncc_val = tf.div(tf.reduce_sum(tf.multiply(im1,im2), axis=1),
                  tf.sqrt(tf.multiply(tf.reduce_sum(tf.square(im1),axis=1), tf.reduce_sum(tf.square(im2),axis=1))))
        ncc_val = tf.reduce_mean(ncc_val)
        return 1-tf.abs(ncc_val)

def charbonnier_loss(x, mask=None, truncate=None, alpha=0.45, beta=1.0, epsilon=0.001):
    """Compute the generalized charbonnier loss of the difference tensor x.
    All positions where mask == 0 are not taken into account.

    Args:
        x: a tensor of shape [num_batch, height, width, channels].
        mask: a mask of shape [num_batch, height, width, mask_channels],
            where mask channels must be either 1 or the same number as
            the number of channels of x. Entries should be 0 or 1.
    Returns:
        loss as tf.float32
    """
    with tf.variable_scope('charbonnier_loss'):
        batch, height, width, thick, channels = tf.unstack(tf.shape(x))
        normalization = tf.cast(batch * height * width * thick *channels, tf.float32)

        error = tf.pow(tf.square(x * beta) + tf.square(epsilon), alpha)

        if mask is not None:
            error = tf.multiply(mask, error)

        if truncate is not None:
            error = tf.minimum(error, truncate)
        # return tf.reduce_sum(error)
        return tf.reduce_sum(error) / normalization


def create_mask(tensor, paddings):
    with tf.variable_scope('create_mask'):
        _, height, width, thick, _ = tensor.get_shape().as_list()
        inner_width = height - (paddings[0][0] + paddings[0][1])
        inner_height = width - (paddings[1][0] + paddings[1][1])
        inner_thick = thick -(paddings[2][0]+paddings[2][1])
        inner = tf.ones([inner_width, inner_height, inner_thick])

        mask3d = tf.pad(inner, paddings)

        mask4d = tf.tile(tf.expand_dims(mask3d, 0), [tf.shape(tensor)[0], 1, 1, 1])
        mask5d = tf.expand_dims(mask4d, 4)
        return tf.stop_gradient(mask5d)


def create_border_mask(tensor, border_ratio=0.05):
    with tf.variable_scope('create_border_mask'):
        _, height, width, thick, _ = tensor.get_shape().as_list()
        min_dim = min(min(height, width),thick)
        sz = int(min_dim * border_ratio)+1
        # border_mask = create_mask(tensor, [[sz, sz], [sz, sz],[2,2]])
        border_mask = create_mask(tensor, [[sz, sz], [sz, sz],[sz,sz]])
        return tf.stop_gradient(border_mask)


def create_outgoing_mask(flow):
    """Computes a mask that is zero at all positions where the flow
    would carry a pixel over the image boundary."""
    with tf.variable_scope('create_outgoing_mask'):
        num_batch, height, width, thick, _ = tf.unstack(tf.shape(flow))

        grid_x = tf.reshape(tf.range(width), [1, 1, width, 1])
        grid_x = tf.tile(grid_x, [num_batch, height, 1, thick])
        grid_y = tf.reshape(tf.range(height), [1, height, 1, 1])
        grid_y = tf.tile(grid_y, [num_batch, 1, width, thick])
        grid_z = tf.reshape(tf.range(thick), [1, 1, 1, thick])
        grid_z = tf.tile(grid_z, [num_batch, width, height, 1])

        flow_u, flow_v, flow_s = tf.unstack(flow, 3, 4)
        pos_x = tf.cast(grid_x, dtype=tf.float32) + flow_u
        pos_y = tf.cast(grid_y, dtype=tf.float32) + flow_v
        pos_z = tf.cast(grid_z, dtype=tf.float32) + flow_s

        inside_x = tf.logical_and(pos_x <= tf.cast(width - 1, tf.float32),
                                  pos_x >=  0.0)
        inside_y = tf.logical_and(pos_y <= tf.cast(height - 1, tf.float32),
                                  pos_y >=  0.0)
        inside_z = tf.logical_and(pos_z <= tf.cast(thick - 1, tf.float32),
                                  pos_z >= 0.0)
        inside1 = tf.logical_and(inside_x, inside_y)
        inside = tf.logical_and(inside1, inside_z)
        return tf.expand_dims(tf.cast(inside, tf.float32), 4)