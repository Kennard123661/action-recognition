import os
import tensorflow as tf
import tf_slim as slim

from nets.optical_flow import PRETRAINED_FLOW_MODEL_DIR
SELFLOW_PRETRAINED_DIR = os.path.join(PRETRAINED_FLOW_MODEL_DIR, 'selflow')
SINTEL_PRETRAINED_DIR = os.path.join(SELFLOW_PRETRAINED_DIR, 'sintel')
KITTI_PRETRAINED_DIR = os.path.join(SELFLOW_PRETRAINED_DIR, 'kitti')


def flow_resize(flow, out_size, is_scale=True, method=tf.image.ResizeMethod.BILINEAR):
    """ method: 0 mean bilinear, 1 means nearest """
    flow_size = tf.cast(tf.shape(input=flow)[-3:-1], dtype=tf.float32)
    flow = tf.image.resize(flow, out_size, method=method)
    if is_scale:
        scale = tf.cast(out_size, dtype=tf.float32) / flow_size
        scale = tf.stack([scale[1], scale[0]])
        flow = tf.multiply(flow, scale)
    return flow


def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W, )
    - y: flattened tensor of shape (B*H*W, )
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(input=x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)
    return tf.gather_nd(img, indices)


def tf_warp(img, flow, H=256, W=256):
    x,y = tf.meshgrid(tf.range(W), tf.range(H))
    x = tf.expand_dims(x,0)
    x = tf.expand_dims(x,-1)

    y = tf.expand_dims(y,0)
    y = tf.expand_dims(y,-1)

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    grid = tf.concat([x,y],axis = -1)
    flows = grid+flow

    max_y = tf.cast(H - 1, tf.int32)
    max_x = tf.cast(W - 1, tf.int32)
    zero = tf.zeros([], dtype=tf.int32)

    x = flows[:, :, :, 0]
    y = flows[:, :, :, 1]
    x0 = x
    y0 = y
    x0 = tf.cast(x0, tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(y0,  tf.int32)
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, tf.float32)
    x1 = tf.cast(x1, tf.float32)
    y0 = tf.cast(y0, tf.float32)
    y1 = tf.cast(y1, tf.float32)

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
    return out


def feature_extractor(x, train=True, trainable=True, reuse=None, regularizer=None, name='feature_extractor'):
    with tf.compat.v1.variable_scope(name, reuse=reuse, regularizer=regularizer):
        with slim.arg_scope([slim.conv2d], activation_fn=lrelu, kernel_size=3, padding='SAME', trainable=trainable):
            net = {}
            net['conv1_1'] = slim.conv2d(x, 16, stride=2, scope='conv1_1')
            net['conv1_2'] = slim.conv2d(net['conv1_1'], 16, stride=1, scope='conv1_2')

            net['conv2_1'] = slim.conv2d(net['conv1_2'], 32, stride=2, scope='conv2_1')
            net['conv2_2'] = slim.conv2d(net['conv2_1'], 32, stride=1, scope='conv2_2')

            net['conv3_1'] = slim.conv2d(net['conv2_2'], 64, stride=2, scope='conv3_1')
            net['conv3_2'] = slim.conv2d(net['conv3_1'], 64, stride=1, scope='conv3_2')

            net['conv4_1'] = slim.conv2d(net['conv3_2'], 96, stride=2, scope='conv4_1')
            net['conv4_2'] = slim.conv2d(net['conv4_1'], 96, stride=1, scope='conv4_2')

            net['conv5_1'] = slim.conv2d(net['conv4_2'], 128, stride=2, scope='conv5_1')
            net['conv5_2'] = slim.conv2d(net['conv5_1'], 128, stride=1, scope='conv5_2')

            net['conv6_1'] = slim.conv2d(net['conv5_2'], 192, stride=2, scope='conv6_1')
            net['conv6_2'] = slim.conv2d(net['conv6_1'], 192, stride=1, scope='conv6_2')
    return net


def context_network(x, flow, train=True, trainable=True, reuse=None, regularizer=None, name='context_network'):
    x_input = tf.concat([x, flow], axis=-1)
    with tf.compat.v1.variable_scope(name, reuse=reuse, regularizer=regularizer):
        with slim.arg_scope([slim.conv2d], activation_fn=lrelu, kernel_size=3, padding='SAME', trainable=trainable):
            net = {}
            net['dilated_conv1'] = slim.conv2d(x_input, 128, rate=1, scope='dilated_conv1')
            net['dilated_conv2'] = slim.conv2d(net['dilated_conv1'], 128, rate=2, scope='dilated_conv2')
            net['dilated_conv3'] = slim.conv2d(net['dilated_conv2'], 128, rate=4, scope='dilated_conv3')
            net['dilated_conv4'] = slim.conv2d(net['dilated_conv3'], 96, rate=8, scope='dilated_conv4')
            net['dilated_conv5'] = slim.conv2d(net['dilated_conv4'], 64, rate=16, scope='dilated_conv5')
            net['dilated_conv6'] = slim.conv2d(net['dilated_conv5'], 32, rate=1, scope='dilated_conv6')
            net['dilated_conv7'] = slim.conv2d(net['dilated_conv6'], 2, rate=1, activation_fn=None,
                                               scope='dilated_conv7')
    refined_flow = net['dilated_conv7']
    return refined_flow


def estimator_network(x1, cost_volume, flow, train=True, trainable=True, reuse=None, regularizer=None,
                      name='estimator'):
    net_input = tf.concat([cost_volume, x1, flow], axis=-1)
    with tf.compat.v1.variable_scope(name, reuse=reuse, regularizer=regularizer):
        with slim.arg_scope([slim.conv2d], activation_fn=lrelu, kernel_size=3, padding='SAME', trainable=trainable):
            net = {}
            net['conv1'] = slim.conv2d(net_input, 128, scope='conv1')
            net['conv2'] = slim.conv2d(net['conv1'], 128, scope='conv2')
            net['conv3'] = slim.conv2d(net['conv2'], 96, scope='conv3')
            net['conv4'] = slim.conv2d(net['conv3'], 64, scope='conv4')
            net['conv5'] = slim.conv2d(net['conv4'], 32, scope='conv5')
            net['conv6'] = slim.conv2d(net['conv5'], 2, activation_fn=None, scope='conv6')
    return net


def compute_cost_volume(x1, x2, H, W, channel, d=9):
    x1 = tf.nn.l2_normalize(x1, axis=3)
    x2 = tf.nn.l2_normalize(x2, axis=3)

    x2_patches = tf.image.extract_patches(x2, [1, d, d, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
    x2_patches = tf.reshape(x2_patches, [-1, H, W, d, d, channel])
    x1_reshape = tf.reshape(x1, [-1, H, W, 1, 1, channel])
    x1_dot_x2 = tf.multiply(x1_reshape, x2_patches)

    cost_volume = tf.reduce_sum(input_tensor=x1_dot_x2, axis=-1)
    # cost_volume = tf.reduce_mean(x1_dot_x2, axis=-1)
    cost_volume = tf.reshape(cost_volume, [-1, H, W, d * d])
    return cost_volume


def estimator(x0, x1, x2, flow_fw, flow_bw, train=True, trainable=True, reuse=None, regularizer=None, name='estimator'):
    # warp x2 according to flow
    if train:
        x_shape = x1.get_shape().as_list()
    else:
        x_shape = tf.shape(input=x1)
    H = x_shape[1]
    W = x_shape[2]
    channel = x_shape[3]
    x2_warp = tf_warp(x2, flow_fw, H, W)
    x0_warp = tf_warp(x0, flow_bw, H, W)

    # ---------------cost volume-----------------

    cost_volume_fw = compute_cost_volume(x1, x2_warp, H, W, channel, d=9)
    cost_volume_bw = compute_cost_volume(x1, x0_warp, H, W, channel, d=9)

    cv_concat_fw = tf.concat([cost_volume_fw, cost_volume_bw], -1)
    cv_concat_bw = tf.concat([cost_volume_bw, cost_volume_fw], -1)

    flow_concat_fw = tf.concat([flow_fw, -flow_bw], -1)
    flow_concat_bw = tf.concat([flow_bw, -flow_fw], -1)

    net_fw = estimator_network(x1, cv_concat_fw, flow_concat_fw, train=train, trainable=trainable, reuse=reuse,
                               regularizer=regularizer, name=name)
    net_bw = estimator_network(x1, cv_concat_bw, flow_concat_bw, train=train, trainable=trainable, reuse=True,
                               regularizer=regularizer, name=name)

    return net_fw, net_bw


def pyramid_processing_three_frame(batch_img, x0_feature, x1_feature, x2_feature, train=True, trainable=True,
                                   reuse=None, regularizer=None, is_scale=True):
    x_shape = tf.shape(input=x1_feature['conv6_2'])
    initial_flow_fw = tf.zeros([x_shape[0], x_shape[1], x_shape[2], 2], dtype=tf.float32, name='initial_flow_fw')
    initial_flow_bw = tf.zeros([x_shape[0], x_shape[1], x_shape[2], 2], dtype=tf.float32, name='initial_flow_bw')
    flow_fw = {}
    flow_bw = {}
    net_fw, net_bw = estimator(x0_feature['conv6_2'], x1_feature['conv6_2'], x2_feature['conv6_2'],
                               initial_flow_fw, initial_flow_bw, train=train, trainable=trainable, reuse=reuse,
                               regularizer=regularizer, name='estimator_level_6')
    flow_fw['level_6'] = net_fw['conv6']
    flow_bw['level_6'] = net_bw['conv6']

    for i in range(4):
        feature_name = 'conv%d_2' % (5 - i)
        level = 'level_%d' % (5 - i)
        feature_size = tf.shape(input=x1_feature[feature_name])[1:3]
        initial_flow_fw = flow_resize(flow_fw['level_%d' % (6 - i)], feature_size, is_scale=is_scale)
        initial_flow_bw = flow_resize(flow_bw['level_%d' % (6 - i)], feature_size, is_scale=is_scale)
        net_fw, net_bw = estimator(x0_feature[feature_name], x1_feature[feature_name], x2_feature[feature_name],
                                   initial_flow_fw, initial_flow_bw, train=train, trainable=trainable, reuse=reuse,
                                   regularizer=regularizer, name='estimator_level_%d' % (5 - i))
        flow_fw[level] = net_fw['conv6']
        flow_bw[level] = net_bw['conv6']

    flow_concat_fw = tf.concat([flow_fw['level_2'], -flow_bw['level_2']], -1)
    flow_concat_bw = tf.concat([flow_bw['level_2'], -flow_fw['level_2']], -1)

    x_feature = tf.concat([net_fw['conv5'], net_bw['conv5']], axis=-1)
    flow_fw['refined'] = context_network(x_feature, flow_concat_fw, train=train, trainable=trainable, reuse=reuse,
                                         regularizer=regularizer, name='context_network')
    flow_size = tf.shape(input=batch_img)[1:3]
    flow_fw['full_res'] = flow_resize(flow_fw['refined'], flow_size, is_scale=is_scale)

    x_feature = tf.concat([net_bw['conv5'], net_fw['conv5']], axis=-1)
    flow_bw['refined'] = context_network(x_feature, flow_concat_bw, train=train, trainable=trainable, reuse=True,
                                         regularizer=regularizer, name='context_network')
    flow_bw['full_res'] = flow_resize(flow_bw['refined'], flow_size, is_scale=is_scale)

    return flow_fw, flow_bw


def pyramid_processing(batch_img0, batch_img1, batch_img2, train=True, trainable=True, reuse=None, regularizer=None,
                       is_scale=True):
    x0_feature = feature_extractor(batch_img0, train=train, trainable=trainable, regularizer=regularizer,
                                   name='feature_extractor')
    x1_feature = feature_extractor(batch_img1, train=train, trainable=trainable, reuse=True, regularizer=regularizer,
                                   name='feature_extractor')
    x2_feature = feature_extractor(batch_img2, train=train, trainable=trainable, reuse=True, regularizer=regularizer,
                                   name='feature_extractor')

    flow_fw, flow_bw = pyramid_processing_three_frame(batch_img0, x0_feature, x1_feature, x2_feature, train=train,
                                                      trainable=trainable,
                                                      reuse=reuse, regularizer=regularizer, is_scale=is_scale)
    return flow_fw, flow_bw


def pyramid_processing_five_frame(batch_img0, batch_img1, batch_img2, batch_img3, batch_img4, train=True,
                                  trainable=True, regularizer=None, is_scale=True):
    x0_feature = feature_extractor(batch_img0, train=train, trainable=trainable, regularizer=regularizer,
                                   name='feature_extractor')
    x1_feature = feature_extractor(batch_img1, train=train, trainable=trainable, reuse=True, regularizer=regularizer,
                                   name='feature_extractor')
    x2_feature = feature_extractor(batch_img2, train=train, trainable=trainable, reuse=True, regularizer=regularizer,
                                   name='feature_extractor')
    x3_feature = feature_extractor(batch_img3, train=train, trainable=trainable, reuse=True, regularizer=regularizer,
                                   name='feature_extractor')
    x4_feature = feature_extractor(batch_img4, train=train, trainable=trainable, reuse=True, regularizer=regularizer,
                                   name='feature_extractor')

    flow_fw_12, flow_bw_10 = pyramid_processing_three_frame(batch_img0, x0_feature, x1_feature, x2_feature, train=train,
                                                            trainable=trainable, reuse=None, regularizer=regularizer,
                                                            is_scale=is_scale)
    flow_fw_23, flow_bw_21 = pyramid_processing_three_frame(batch_img0, x1_feature, x2_feature, x3_feature, train=train,
                                                            trainable=trainable, reuse=True, regularizer=regularizer,
                                                            is_scale=is_scale)
    flow_fw_34, flow_bw_32 = pyramid_processing_three_frame(batch_img0, x2_feature, x3_feature, x4_feature, train=train,
                                                            trainable=trainable, reuse=True, regularizer=regularizer,
                                                            is_scale=is_scale)

    return flow_fw_12, flow_bw_10, flow_fw_23, flow_bw_21, flow_fw_34, flow_bw_32


def lrelu(x, leak=0.2, name='leaky_relu'):
    return tf.maximum(x, leak * x)
