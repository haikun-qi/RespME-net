# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf


def transformer(U, theta, trans, out_size, name='SpatialTransformer', **kwargs):
    """Spatial Transformer Layer
    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and edited by David Dao for Tensorflow.
    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, thick, num_channels].
    theta: float
        The output of the
        localisation network should be [num_batch, 12].
    out_size: tuple of 3 ints
        The size of the output of the network (height, width, thick)
    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
    Notes
    -----
    To initialize the network to the identity transform init
    ``theta`` to :
        identity = np.array([[1., 0., 0.],
                             [0., 1., 0.]])
        identity = identity.flatten()
        theta = tf.Variable(initial_value=identity)
    """

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, z,out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            thick = tf.shape(im)[3]
            channels = tf.shape(im)[4]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            z = tf.cast(z, 'float32')

            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            thick_f = tf.cast(thick, 'float32')

            out_height = out_size[0]
            out_width = out_size[1]
            out_thick = out_size[2]

            zero = tf.zeros([], dtype='int32')
            max_x = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_y = tf.cast(tf.shape(im)[2] - 1, 'int32')
            max_z = tf.cast(tf.shape(im)[3] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            zero_f=tf.zeros([],dtype='float32')
            x = (x + 1.0)*(height_f) / 2.0
            y = (y + 1.0)*(width_f) / 2.0
            z = (z + 1.0)*(thick_f) / 2.0
            x = tf.clip_by_value(x, zero_f, height_f - 1.0)
            y = tf.clip_by_value(y, zero_f, width_f - 1.0)
            z = tf.clip_by_value(z, zero_f, thick_f - 1.0)

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1
            z0 = tf.cast(tf.floor(z), 'int32')
            z1 = z0 + 1

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            z0 = tf.clip_by_value(z0, zero, max_z)
            z1 = tf.clip_by_value(z1, zero, max_z)

            dim1=height*width*thick
            base = _repeat(tf.range(num_batch)*dim1, out_height*out_width*out_thick)

            base_x0y0 = base + x0 * width * thick + y0 * thick  # top left
            base_x1y0 = base + x1 * width * thick + y0 * thick  # bottom left
            base_x0y1 = base + x0 * width * thick + y1 * thick  # top right
            base_x1y1 = base + x1 * width * thick + y1 * thick  # bottom right

            # the z0 plane
            idx_a1 = base_x0y0 + z0  # top left
            idx_b1 = base_x1y0 + z0  # bottom left
            idx_c1 = base_x0y1 + z0  # top right
            idx_d1 = base_x1y1 + z0  # bottom right

            # the z1 plane
            idx_a2 = base_x0y0 + z1  # top left
            idx_b2 = base_x1y0 + z1  # bottom left
            idx_c2 = base_x0y1 + z1  # top right
            idx_d2 = base_x1y1 + z1  # bottom right

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')

            Ia1 = tf.gather(im_flat, idx_a1)
            Ia2 = tf.gather(im_flat, idx_a2)
            Ib1 = tf.gather(im_flat, idx_b1)
            Ib2 = tf.gather(im_flat, idx_b2)
            Ic1 = tf.gather(im_flat, idx_c1)
            Ic2 = tf.gather(im_flat, idx_c2)
            Id1 = tf.gather(im_flat, idx_d1)
            Id2 = tf.gather(im_flat, idx_d2)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            z0_f = tf.cast(z0, 'float32')
            z1_f = tf.cast(z1, 'float32')

            # on the z0 plane
            wa1 = tf.expand_dims((x1_f-x) * (y1_f-y) * (z1_f-z), 1)  # top left pixel
            wb1 = tf.expand_dims((x-x0_f) * (y1_f-y) * (z1_f-z), 1)  # bottom left pixel
            wc1 = tf.expand_dims((x1_f-x) * (y-y0_f) * (z1_f-z), 1)  # top right pixel
            wd1 = tf.expand_dims((x-x0_f) * (y-y0_f) * (z1_f-z), 1)  # bottom right pixel
            # on the z1 plane
            wa2 = tf.expand_dims((x1_f-x) * (y1_f-y) * (z-z0_f), 1)  # top left pixel
            wb2 = tf.expand_dims((x-x0_f) * (y1_f-y) * (z-z0_f), 1)  # bottom left pixel
            wc2 = tf.expand_dims((x1_f-x) * (y-y0_f) * (z-z0_f), 1)  # top right pixel
            wd2 = tf.expand_dims((x-x0_f) * (y-y0_f) * (z-z0_f), 1)  # bottom right pixel

            output = tf.add_n([wa1*Ia1, wa2*Ia2, wb1*Ib1, wb2*Ib2, wc1*Ic1, wc2*Ic2, wd1*Id1, wd2*Id2])
            return output

    def _meshgrid(height, width, thick):
        with tf.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t1 = tf.tile(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),[1,width])
            x_t = tf.tile(tf.expand_dims(x_t1,2),[1,1,thick])

            y_t1 = tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1),[1,0])
            y_t2 = tf.tile(y_t1, [height,1])
            y_t2=tf.expand_dims(y_t2,2)
            y_t = tf.tile(y_t2,[1,1,thick])

            z_t = tf.tile(tf.transpose(tf.expand_dims(tf.expand_dims(tf.linspace(-1.0, 1.0, thick), 1),2),
                                       [2, 1, 0]),[height,width,1])

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))
            z_t_flat = tf.reshape(z_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, z_t_flat, ones])
            return grid

    def _transform(theta, trans, input_dim, out_size):
        with tf.variable_scope('_transform'):
            num_batch = tf.shape(input_dim)[0]

            num_channels = tf.shape(input_dim)[4]
            theta = tf.reshape(theta, (-1, 3, 4))
            theta = tf.cast(theta, 'float32')

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            # height_f = tf.cast(height, 'float32')
            # width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            out_thick = out_size[2]
            grid = _meshgrid(out_height, out_width, out_thick)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, tf.stack([num_batch, 4, -1]))

            # Transform A x (x_t, y_t, z_t, 1)^T -> (x_s, y_s, z_s)
            trans = tf.expand_dims(trans,2)
            trans = tf.tile(trans,[1, 1, tf.shape(grid)[2]])

            T_g = tf.matmul(theta, grid) + trans
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            z_s = tf.slice(T_g, [0, 2, 0], [-1, -1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])
            z_s_flat = tf.reshape(z_s, [-1])

            input_transformed = _interpolate(
                input_dim, x_s_flat, y_s_flat, z_s_flat,
                out_size)

            output = tf.reshape(
                input_transformed, tf.stack([num_batch, out_height, out_width, out_thick,num_channels]))
            return output

    with tf.variable_scope(name):
        output = _transform(theta, trans, U, out_size)
        return output


# def batch_transformer(U, thetas, out_size, name='BatchSpatialTransformer'):
#     """Batch Spatial Transformer Layer
#     Parameters
#     ----------
#     U : float
#         tensor of inputs [num_batch,height,width,thick, num_channels]
#     thetas : float
#         a set of transformations for each input [num_batch,num_transforms,12]
#     out_size : int
#         the size of the output [out_height,out_width, out_thick]
#     Returns: float
#         Tensor of size [num_batch*num_transforms,out_height,out_width,out_thick, num_channels]
#     """
#     with tf.variable_scope(name):
#         num_batch, num_transforms = map(int, thetas.get_shape().as_list()[:2])
#         indices = [[i]*num_transforms for i in xrange(num_batch)]
#         input_repeated = tf.gather(U, tf.reshape(indices, [-1]))
#         return transformer(input_repeated, thetas, out_size)
