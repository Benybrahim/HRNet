"""
Implementation of HRNet Network paper:
"""
import os
import logging

import tensorflow as tf

logger = logging.getLogger(__name__)
layers = tf.keras.layers
BN_MOMENTUM = 0.1

kernel_initializer='he_normal'
use_bias=False
upsampling_interpolation='nearest'
# batch norm axis=3 ??

def HRNet(input_shape, classes):

    inputs = layers.Input(input_shape, name='input')

    # STAGE 1
    x = conv2d_pad(inputs, 64, 3, 2, 1, name='stage1_stem_conv1')
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, name='stage1_stem_bn1')(x)
    x = layers.Activation('relu', name='stage1_stem_relu1')(x)
    x = conv2d_pad(x, 64, 3, 2, 1, name='stage1_stem_conv2')
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, name='stage1_stem_bn2')(x)
    x = layers.Activation('relu', name='stage1_stem_relu2')(x)
    x = bottleneck_block(x, 256, downsample=True, name='stage1_bottleneck1')
    x = bottleneck_block(x, 256, downsample=False, name='stage1_bottleneck2')
    x = bottleneck_block(x, 256, downsample=False, name='stage1_bottleneck3')
    x = bottleneck_block(x, 256, downsample=False, name='stage1_bottleneck4')
    x1, x2 = transion_layer1(x, filters=[32, 64], name='stage1_transition')

    # STAGE 2
    x1 = make_branch(x1, 32, name='stage2_branch1')
    x2 = make_branch(x2, 64, name='stage2_branch2')
    x1, x2 = fuse_layer1([x1, x2], filters=[32, 64], name='stage2_fuse')
    x1, x2, x3 = transition_layer2([x1, x2], filters=[32, 64, 128],
                                   name='stage2_transition')

    # STAGE 3
    for i in range(4):
        x1 = make_branch(x1, 32, name=f'stage3_{i+1}_branch1')
        x2 = make_branch(x2, 64, name=f'stage3_{i+1}_branch2')
        x3 = make_branch(x3, 128, name=f'stage3_{i+1}_branch3')
        x1, x2, x3 = fuse_layer2([x1, x2, x3], filters=[32, 64, 128],
                                 name=f'stage3_{i+1}_fuse')

    x1, x2, x3, x4 = transition_layer3([x1, x2, x3], filters=[32, 64, 128, 256],
                                       name='stage3_transition')

    # STAGE 4
    for i in range(3):
        x1 = make_branch(x1, 32, name=f'stage4_{i+1}_branch1')
        x2 = make_branch(x2, 64, name=f'stage4_{i+1}_branch2')
        x3 = make_branch(x3, 128, name=f'stage4_{i+1}_branch3')
        x4 = make_branch(x4, 256, name=f'stage4_{i+1}_branch4')
        if i != 2:
            x1, x2, x3, x4 = fuse_layer3([x1, x2, x3, x4],
                                         filters=[32, 64, 128, 256],
                                         name=f'stage4_{i + 1}_fuse')
        else:
            x = fuse_layer4([x1, x2, x3, x4], 32, name=f'stage4_{i + 1}_fuse')

    outputs = layers.Conv2D(classes, 1, activation='sigmoid', name='output')(x)

    model = tf.keras.Model(inputs, outputs)
    return model


    # Module 1

def conv2d_pad(inputs, filters, kernel_size, strides=1, padding=0, use_bias=False, name='conv'):
    x = layers.ZeroPadding2D(padding, name=f'{name}_pad')(inputs)
    x = layers.Conv2D(filters, kernel_size, strides, use_bias=use_bias, name=name)(x)
    return x

#def conv2d_bn(inputs, filters, kernel_size, strides=1, padding=0, use_bias=False, activation=None, name='conv'):
#    x = conv2d_pad(inputs, filters[0], 3, 1, 1, name=f'{name}_conv1')
#    x = layers.BatchNormalization(name=f'{name}_bn1')(x1)
#    if activation:
#        x = layers.Activation(activation, name=f'{name}_branch1_out')(x1)
#    return x

def bottleneck_block(inputs, filters, strides=1, downsample=False, name='bottleneck'):
    expansion = 4

    residual = inputs

    x = conv2d_pad(inputs, filters//expansion, 1, 1, 0, name=f'{name}_conv1')
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}_bn1')(x)
    x = layers.Activation('relu', name=f'{name}_relu1')(x)

    x = conv2d_pad(x, filters//expansion, 3, strides, 1, name=f'{name}_conv2')
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}_bn2')(x)
    x = layers.Activation('relu', name=f'{name}_relu2')(x)

    x = conv2d_pad(x, filters, 1, 1, 0, name=f'{name}_conv3')
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}_bn3')(x)

    if downsample:
        residual = conv2d_pad(inputs, filters, 1, strides, 0, name=f'{name}_down_conv')
        residual = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}_downsample_bn')(residual)

    x = layers.Add(name=f'{name}_res')([x, residual])
    x = layers.Activation('relu', name=f'{name}_out')(x)

    return x

def transion_layer1(inputs, filters=[32, 64], name='stage1_transition'):
    x1 = conv2d_pad(inputs, filters[0], 3, 1, 1, name=f'{name}_conv1')
    x1 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}_bn1')(x1)
    x1 = layers.Activation('relu', name=f'{name}_branch1_out')(x1)

    x2 = conv2d_pad(inputs, filters[1], 3, 2, 1, name=f'{name}_conv2')
    x2 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}_bn2')(x2)
    x2 = layers.Activation('relu', name=f'{name}_branch2_out')(x2)

    return [x1, x2]

def make_branch(inputs, filters, name='branch'):
    x = basic_block(inputs, filters, downsample=False, name=f'{name}_basic1')
    x = basic_block(x, filters, downsample=False, name=f'{name}_basic2')
    x = basic_block(x, filters, downsample=False, name=f'{name}_basic3')
    x = basic_block(x, filters, downsample=False, name=f'{name}_basic4')
    return x

def basic_block(inputs, filters, strides=1, downsample=False, name='basic'):
    expansion = 1

    residual = inputs

    x = conv2d_pad(inputs, filters//expansion, 3, strides, 1, name=f'{name}_conv1')
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}_bn1')(x)
    x = layers.Activation('relu', name=f'{name}_relu1')(x)

    x = conv2d_pad(x, filters//expansion, 3, 1, 1, name=f'{name}_conv2')
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}_bn2')(x)

    if downsample:
        residual = conv2d_pad(inputs, filters, 1, strides, 0, name=f'{name}_down_conv')
        residual = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}_down_bn')(residual)

    x = layers.Add(name=f'{name}_res')([x, residual])
    x = layers.Activation('relu', name=f'{name}_out')(x)

    return x

def fuse_layer1(inputs, filters=[32, 64], name='stage2_fuse'):
    x1, x2 = inputs

    x11 = x1
    x21 = conv2d_pad(x2, filters[0], 1, 1, 0, name=f'{name}_conv_2_1')
    x21 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}_bn_2_1')(x21)
    x21 = layers.UpSampling2D(2, name=f'{name}_up2_1')(x21)
    x1 = layers.Add(name=f'{name}_add1')([x11, x21])
    x1 = layers.Activation('relu', name=f'{name}_branch1_out')(x1)

    x22 = x2
    x12 = conv2d_pad(x1, filters[1], 3, 2, 1, name=f'{name}_conv1_2')
    x12 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}_bn1_2')(x12)
    x2 = layers.Add(name=f'{name}_add2')([x12, x22])
    x2 = layers.Activation('relu', name=f'{name}_branch2_out')(x2)

    return [x1, x2]

def transition_layer2(inputs, filters, name='stage2_transition'):
    x1, x2 = inputs

    x1 = conv2d_pad(x1, filters[0], 3, 1, 1, name=f'{name}_conv1')
    x1 = layers.BatchNormalization(name=f'{name}_bn1')(x1)
    x1 = layers.Activation('relu', name=f'{name}_branch1_out')(x1)

    x21 = conv2d_pad(x2, filters[1], 3, 1, 1, name=f'{name}_conv2')
    x21 = layers.BatchNormalization(name=f'{name}_bn2')(x21)
    x21 = layers.Activation('relu', name=f'{name}_branch2_out')(x21)

    x22 = conv2d_pad(x2, filters[2], 3, 2, 1, name=f'{name}_conv3')
    x22 = layers.BatchNormalization(name=f'{name}_bn3')(x22)
    x22 = layers.Activation('relu', name=f'{name}_branch3_out')(x22)

    return [x1, x21, x22]

def fuse_layer2(inputs, filters=[32, 64, 128], name='stage3_fuse'):
    x1, x2, x3 = inputs

    # branch 1
    x11 = x1

    x21 = conv2d_pad(x2, filters[0], 1, 1, 0, name=f'{name}_conv2_1')
    x21 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}_bn2_1')(x21)
    x21 = layers.UpSampling2D(2, name=f'{name}_up2_1')(x21)

    x31 = conv2d_pad(x3, filters[0], 1, 1, 0, name=f'{name}_conv3_1')
    x31 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}_bn3_1')(x31)
    x31 = layers.UpSampling2D(4, name=f'{name}_up3_1')(x31)

    x1 = layers.Add(name=f'{name}_add1')([x11, x21, x31])
    x1 = layers.Activation('relu', name=f'{name}_branch1_out')(x1)

    # branch 2
    x22 = x2

    x12 = conv2d_pad(x1, filters[1], 3, 2, 1, name=f'{name}_conv1_2')
    x12 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}_bn1_2')(x12)

    x32 = conv2d_pad(x3, filters[1], 1, 1, 0, name=f'{name}_conv3_2')
    x32 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}_bn2_2')(x32)
    x32 = layers.UpSampling2D(2, name=f'{name}_up3_2')(x32)

    x2 = layers.Add(name=f'{name}_add2')([x12, x22, x32])
    x2 = layers.Activation('relu', name=f'{name}_branch2_out')(x2)

    # branch 3
    x33 = x3

    x13 = conv2d_pad(x1, filters[0], 3, 2, 1, name=f'{name}_conv1_3_1')
    x13 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}_bn1_3_1')(x13)
    x13 = layers.Activation('relu', name=f'{name}_relu1_3_1')(x13)
    x13 = conv2d_pad(x13, filters[2], 3, 2, 1, name=f'{name}_conv1_3_2')
    x13 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}_bn_1_3_2')(x13)

    x23 = conv2d_pad(x2, filters[2], 3, 2, 1, name=f'{name}_conv2_3')
    x23 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}_bn2_3')(x23)

    x3 = layers.Add(name=f'{name}_add3')([x13, x23, x33])
    x3 = layers.Activation('relu', name=f'{name}_branch3_out')(x3)

    return [x1, x2, x3]









    x01 = conv2d_pad(x0, 128, 3, 2, 1, name=f'{name}_conv1')
    x01 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}_bn1')(
        x01)

    x0 = layers.Add(name=f'{name}_add1')([x01, x11, x21])


    x0 = layers.Add(name=f'{name}_add1')([x0, x11])
    x1 = layers.Add(name=f'{name}_add2')([x1, x01])

    x0 = layers.Activation('relu', name=f'{name}_relu1')(x0)
    x1 = layers.Activation('relu', name=f'{name}_relu2')(x1)

    return [x0, x1]

def transition_layer3(inputs, filters, name='stage3_transition'):
    x1, x2, x3 = inputs

    x1 = conv2d_pad(x1, filters[0], 3, 1, 1, name=f'{name}_conv1')
    x1 = layers.BatchNormalization(name=f'{name}_bn1')(x1)
    x1 = layers.Activation('relu', name=f'{name}_branch1_out')(x1)

    x2 = conv2d_pad(x2, filters[1], 3, 1, 1, name=f'{name}_conv2')
    x2 = layers.BatchNormalization(name=f'{name}_bn2')(x2)
    x2 = layers.Activation('relu', name=f'{name}_branch2_out')(x2)

    x31 = conv2d_pad(x3, filters[2], 3, 1, 1, name=f'{name}_conv3')
    x31 = layers.BatchNormalization(name=f'{name}_bn3')(x31)
    x31 = layers.Activation('relu', name=f'{name}_branch3_out')(x31)

    x32 = conv2d_pad(x3, filters[3], 3, 2, 1, name=f'{name}_conv4')
    x32 = layers.BatchNormalization(name=f'{name}_bn4')(x32)
    x32 = layers.Activation('relu', name=f'{name}_branch4_out')(x32)

    return [x1, x2, x31, x32]

def fuse_layer3(inputs, filters=[32, 64, 128, 256], name='stage4_fuse'):
    x1, x2, x3, x4 = inputs

    # branch 1
    x11 = x1

    x21 = conv2d_pad(x2, filters[0], 1, 1, 0, name=f'{name}_conv2_1')
    x21 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}bn2_1')(x21)
    x21 = layers.UpSampling2D(2, name=f'{name}_up2_1')(x21)

    x31 = conv2d_pad(x3, filters[0], 1, 1, 0, name=f'{name}_conv3_1')
    x31 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}bn3_1')(x31)
    x31 = layers.UpSampling2D(4, name=f'{name}_up3_1')(x31)

    x41 = conv2d_pad(x4, filters[0], 1, 1, 0, name=f'{name}_conv4_1')
    x41 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}bn4_1')(x41)
    x41 = layers.UpSampling2D(8, name=f'{name}_up4_1')(x41)

    x1 = layers.Add(name=f'{name}_add1')([x11, x21, x31, x41])
    x1 = layers.Activation('relu', name=f'{name}_branch1_out')(x1)

    # branch 2
    x22 = x2

    x12 = conv2d_pad(x1, filters[1], 3, 2, 1, name=f'{name}_conv1_2')
    x12 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}bn1_2')(x12)

    x32 = conv2d_pad(x3, filters[1], 1, 1, 0, name=f'{name}_conv3_2')
    x32 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}bn3_2')(x32)
    x32 = layers.UpSampling2D(2, name=f'{name}_up3_2')(x32)

    x42 = conv2d_pad(x4, filters[1], 1, 1, 0, name=f'{name}_conv4_2')
    x42 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}bn4_2')(x42)
    x42 = layers.UpSampling2D(4, name=f'{name}_up4_2')(x42)

    x2 = layers.Add(name=f'{name}_add2')([x12, x22, x32, x42])
    x2 = layers.Activation('relu', name=f'{name}_branch2_out')(x2)

    # branch 3
    x33 = x3

    x13 = conv2d_pad(x1, filters[0], 3, 2, 1, name=f'{name}_conv1_3_1')
    x13 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}bn1_3_1')(x13)
    x13 = layers.Activation('relu', name=f'{name}_relu1_3_1')(x13)
    x13 = conv2d_pad(x13, filters[2], 3, 2, 1, name=f'{name}_conv1_3_2')
    x13 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}bn_1_3_2')(x13)

    x23 = conv2d_pad(x2, filters[2], 3, 2, 1, name=f'{name}_conv2_3')
    x23 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}bn2_3')(x23)

    x43 = conv2d_pad(x4, filters[2], 1, 1, 0, name=f'{name}_conv4_3')
    x43 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}bn4_3')(x43)
    x43 = layers.UpSampling2D(2, name=f'{name}_up4_3')(x43)

    x3 = layers.Add(name=f'{name}_add3')([x13, x23, x33, x43])
    x3 = layers.Activation('relu', name=f'{name}_branch3_out')(x3)

    # branch 4
    x44 = x4

    x14 = conv2d_pad(x1, filters[0], 3, 2, 1, name=f'{name}_conv1_4_1')
    x14 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}_bn1_4_1')(x14)
    x14 = layers.Activation('relu', name=f'{name}_relu1_4_1')(x14)
    x14 = conv2d_pad(x14, filters[0], 3, 2, 1, name=f'{name}_conv1_4_2')
    x14 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}_bn1_4_2')(x14)
    x14 = layers.Activation('relu', name=f'{name}_relu1_4_2')(x14)
    x14 = conv2d_pad(x14, filters[3], 3, 2, 1, name=f'{name}_conv1_4_3')
    x14 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}_bn1_4_3')(x14)

    x24 = conv2d_pad(x2, filters[1], 3, 2, 1, name=f'{name}_conv2_4_1')
    x24 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}_bn2_4_1')(x24)
    x24 = layers.Activation('relu', name=f'{name}_relu2_4_1')(x24)
    x24 = conv2d_pad(x24, filters[3], 3, 2, 1, name=f'{name}_conv2_4_2')
    x24 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}_bn2_4_2')(x24)

    x34 = conv2d_pad(x3, filters[3], 3, 2, 1, name=f'{name}_conv3_4')
    x34 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}_n3_4')(x34)

    x4 = layers.Add(name=f'{name}_add4')([x14, x24, x34, x44])
    x4 = layers.Activation('relu', name=f'{name}_branch4_out')(x4)

    return [x1, x2, x3, x4]









    x01 = conv2d_pad(x0, 128, 3, 2, 1, name=f'{name}_conv1')
    x01 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}_bn1')(
        x01)

    x0 = layers.Add(name=f'{name}_add1')([x01, x11, x21])


    x0 = layers.Add(name=f'{name}_add1')([x0, x11])
    x1 = layers.Add(name=f'{name}_add2')([x1, x01])

    x0 = layers.Activation('relu', name=f'{name}_relu1')(x0)
    x1 = layers.Activation('relu', name=f'{name}_relu2')(x1)

    return [x0, x1]

def fuse_layer4(inputs, filters=32, name='final_fuse'):
    x1, x2, x3, x4 = inputs

    x11 = x1

    x21 = conv2d_pad(x2, filters, 1, 1, 0, name=f'{name}_conv2_1')
    x21 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}bn2_1')(x21)
    x21 = layers.UpSampling2D(2, name=f'{name}_up2_1')(x21)

    x31 = conv2d_pad(x3, filters, 1, 1, 0, name=f'{name}_conv3_1')
    x31 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}bn3_1')(x31)
    x31 = layers.UpSampling2D(4, name=f'{name}_up3_1')(x31)

    x41 = conv2d_pad(x4, filters, 1, 1, 0, name=f'{name}_conv4_1')
    x41 = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f'{name}bn4_1')(x41)
    x41 = layers.UpSampling2D(8, name=f'{name}_up4_1')(x41)

    x = layers.Concatenate(name=f'{name}_out')([x11, x21, x31, x41])
    return x


if __name__=='__main__':
    # test
    # (256, 192) or (384, 288)
    model = HRNet((384, 288, 3), 17)
    print(model.summary())



