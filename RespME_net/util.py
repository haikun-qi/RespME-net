
import tensorflow as tf
from skimage.transform import warp
import numpy as np

# from ..ops import downsample as downsample_ops

def summarized_placeholder(name, prefix=None, key=tf.GraphKeys.SUMMARIES):
    prefix = '' if not prefix else prefix + '/'
    p = tf.placeholder(tf.float32, name=name)
    tf.summary.scalar(prefix + name, p, collections=[key])
    return p


def resize_area(tensor, like):
    _, h, w, _ = tf.unstack(tf.shape(like))
    return tf.stop_gradient(tf.image.resize_area(tensor, [h, w]))


def resize_bilinear(tensor, like):
    _, h, w, _ = tf.unstack(tf.shape(like))
    return tf.stop_gradient(tf.image.resize_bilinear(tensor, [h, w]))


def downsample(tensor, num):
    _, height, width, thick, _ = tensor.shape.as_list()

    # if height%2==0 and width%2==0 and thick%2==0:
    #     return downsample_ops(tensor, num)
    # else:
    return resize_3D(tensor, [int(height / num), int(width / num), int(thick / num)])
    # return tf.image.resize_area(tensor,tf.constant([int(height/num),int(width/num), int(thick/num)]))


def resize_3D(input_tensor, new_size):
    b_size, x_size, y_size, z_size, c_size = \
        input_tensor.get_shape().as_list()
    x_size_new, y_size_new, z_size_new = new_size

    # resize y-z
    squeeze_b_x = tf.reshape(
        input_tensor, [-1, y_size, z_size, c_size])
    resize_b_x = tf.image.resize_images(
        squeeze_b_x, [y_size_new, z_size_new])
    resume_b_x = tf.reshape(
        resize_b_x, [-1, x_size, y_size_new, z_size_new, c_size])

    # resize x-y
    #   first reorient
    reoriented = tf.transpose(resume_b_x, [0, 3, 2, 1, 4])
    #   squeeze and 2d resize
    squeeze_b_z = tf.reshape(
        reoriented, [-1, y_size_new, x_size, c_size])
    resize_b_z = tf.image.resize_images(
        squeeze_b_z, [y_size_new, x_size_new])
    resume_b_z = tf.reshape(
        resize_b_z, [-1, z_size_new, y_size_new, x_size_new, c_size])

    return tf.transpose(resume_b_z, [0, 3, 2, 1, 4])


def image_warp_np(im, flow):
    im=np.squeeze(im)
    flow=np.squeeze(flow)
    im=im.astype('float32')
    flow=flow.astype('float32')

    height=im.shape[0]
    width=im.shape[1]
    thick=im.shape[2]
    posx, posy, posz = np.mgrid[:height,:width,:thick]
    # flow=np.reshape(flow, [-1, 3])
    vx=flow[:,:,:,0]
    vy=flow[:,:,:,1]
    vz=flow[:,:,:,2]

    coords0=posx+vx
    coords1=posy+vy
    coords2=posz+vz
    coords = np.array([coords0, coords1, coords2])

    warped=warp(im, coords)

    return warped