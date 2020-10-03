
import tensorflow as tf


def image_warp(im, flow):
    """Performs a backward warp of an image using the predicted flow.

    Args:
        im: Batch of images. [num_batch, height, width, thick, channels]
        flow: Batch of flow vectors. [num_batch, height, width, thick, 3]
    Returns:
        warped: transformed image of the same shape as the input image.
    """
    with tf.variable_scope('image_warp'):

        num_batch, height, width, thick, channels = tf.unstack(tf.shape(im))

        max_x = tf.cast(height - 1, 'int32')
        max_y = tf.cast(width - 1, 'int32')
        max_z = tf.cast(thick - 1, 'int32')
        zero = tf.zeros([], dtype='int32')

        # We have to flatten our tensors to vectorize the interpolation
        im_flat = tf.reshape(im, [-1, channels])
        flow_flat = tf.reshape(flow, [-1, 3])

        # Floor the flow, as the final indices are integers
        # The fractional part is used to control the bilinear interpolation.
        flow_floor = tf.to_int32(tf.floor(flow_flat))
        bilinear_weights = flow_flat - tf.floor(flow_flat)

        # Construct base indices which are displaced with the flow
        # the 1st dimension
        grid_x = tf.tile(tf.expand_dims(tf.range(height), 1), [1, width])
        grid_x = tf.tile(tf.expand_dims(grid_x,2),[1, 1, thick])
        pos_x = tf.tile(tf.reshape(grid_x,[-1]),[num_batch])
        # the 2nd dimension

        grid_y = (tf.tile(tf.expand_dims(tf.range(width), 1), [1, height]))
        grid_y = tf.transpose(grid_y, [1, 0])
        grid_y = tf.tile(tf.expand_dims(grid_y,2),[1, 1, thick])
        pos_y = tf.tile(tf.reshape(grid_y, [-1]), [num_batch])
        # the third dimension
        grid_z = tf.tile(tf.expand_dims(tf.range(thick),1), [1,width]) #zy
        grid_z = tf.transpose(tf.tile(tf.expand_dims(grid_z,2),[1,1,height]),[2,1,0])
        pos_z = tf.tile(tf.reshape(grid_z,[-1]),[num_batch])


        x = flow_floor[:, 0]
        y = flow_floor[:, 1]
        z = flow_floor[:, 2]
        xw = bilinear_weights[:, 0]
        yw = bilinear_weights[:, 1]
        zw = bilinear_weights[:, 2]

        x0 = pos_x + x
        x1 = x0 + 1
        y0 = pos_y + y
        y1 = y0 + 1
        z0 = pos_z + z
        z1 = z0 + 1
        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        z0 = tf.clip_by_value(z0, zero, max_z)
        z1 = tf.clip_by_value(z1, zero, max_z)

        # Compute interpolation weights for 8 adjacent pixels
        # expand to num_batch * height * width*thick x 1 for broadcasting in add_n below
        # on the z0 plane
        wa = tf.expand_dims((1 - xw) * (1 - yw) * (1 - zw), 1)  # top left pixel
        wb = tf.expand_dims(xw * (1 - yw) * (1 - zw), 1)  # bottom left pixel
        wc = tf.expand_dims((1-xw) * yw * (1 - zw), 1)  # top right pixel
        wd = tf.expand_dims(xw * yw * (1 - zw), 1)  # bottom right pixel
        # on the z1 plane
        wa2 = tf.expand_dims((1 - xw) * (1 - yw) * zw, 1)  # top left pixel
        wb2 = tf.expand_dims(xw * (1 - yw) * zw, 1)  # bottom left pixel
        wc2 = tf.expand_dims((1 - xw) * yw * zw, 1)  # top right pixel
        wd2 = tf.expand_dims(xw * yw * zw, 1)  # bottom right pixel

        dim1 = width * height * thick
        batch_offsets = tf.range(num_batch) * dim1
        base_grid = tf.tile(tf.expand_dims(batch_offsets, 1), [1, dim1])
        base = tf.reshape(base_grid, [-1])

        base_x0y0 = base + x0 * width*thick + y0*thick #top left
        base_x1y0 = base + x1 * width*thick + y0*thick #bottom left
        base_x0y1 = base + x0 * width*thick + y1*thick # top right
        base_x1y1 = base + x1 * width*thick + y1*thick # bottom right

        # the z0 plane
        idx_a = base_x0y0 + z0 #top left
        idx_b = base_x1y0 + z0  #bottom left
        idx_c = base_x0y1 + z0  # top right
        idx_d = base_x1y1 + z0  # bottom right

        # the z1 plane
        idx_a2 = base_x0y0 + z1  # top left
        idx_b2 = base_x1y0 + z1  # bottom left
        idx_c2 = base_x0y1 + z1  # top right
        idx_d2 = base_x1y1 + z1  # bottom right


        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        Ia2 = tf.gather(im_flat, idx_a2)
        Ib2 = tf.gather(im_flat, idx_b2)
        Ic2 = tf.gather(im_flat, idx_c2)
        Id2 = tf.gather(im_flat, idx_d2)

        warped_flat = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id,
                                wa2*Ia2, wb2*Ib2, wc2*Ic2, wd2*Id2])
        warped = tf.reshape(warped_flat, [tf.shape(im)[0], tf.shape(im)[1], tf.shape(im)[2], tf.shape(im)[3], tf.shape(im)[4]])

        return warped


