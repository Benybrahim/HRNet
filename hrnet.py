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
# batch norm axis=3 ??

def HRNet(input_shape):

    x = layers.Input(input_shape)

    # STAGE 1

    # stem block
    x = conv2d_pad(x, 64, 3, 2, 1, use_bias=False)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM)(x)
    x = layers.Activation('relu')(x)

    x = conv2d_pad(x, 64, 3, 2, 1, use_bias=False)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM)(x)
    x = layers.Activation('relu')(x)

    # bottleneck blocks
    x = bottleneck_block(x, 256, downsample=True)
    x = bottleneck_block(x, 256, downsample=False)
    x = bottleneck_block(x, 256, downsample=False)
    x = bottleneck_block(x, 256, downsample=False)

    # transition block 1
    x0, x1 = transion_block1(x, filters=[32, 64])


    model = tf.keras.Model(x, [x0, x1])
    return model
    # STAGE 2
    #x0 = basic_block(x0, 32)
    #x1 = basic_block(x1, 64)

    # Module 1

def conv2d_pad(inputs, filters, kernel_size, strides=1, padding=0, use_bias=False):
    x = layers.ZeroPadding2D(padding)(inputs)
    x = layers.Conv2D(filters, kernel_size, strides, use_bias=use_bias)(x)
    return x

def bottleneck_block(inputs, filters, strides=1, downsample=False):
    expansion = 4

    residual = inputs

    x = conv2d_pad(inputs, filters//expansion, 1, 1, 0, use_bias=False)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM)(x)
    x = layers.Activation('relu')(x)

    x = conv2d_pad(x, filters//expansion, 3, strides, 1, use_bias=False)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM)(x)
    x = layers.Activation('relu')(x)

    x = conv2d_pad(x, filters, 1, 1, 0, use_bias=False)
    x = layers.Activation('relu')(x)

    if downsample:
        residual = conv2d_pad(inputs, filters, 1, strides, 0, use_bias=False)
        residual = layers.BatchNormalization(momentum=BN_MOMENTUM)(residual)

    x = layers.Add()([x, residual])
    x = layers.Activation('relu')(x)

    return x

def transion_block1(inputs, filters=[32, 64]):
    x0 = conv2d_pad(inputs, filters[0], 3, 1, 1, use_bias=False)
    x0 = layers.BatchNormalization(momentum=BN_MOMENTUM)(x0)
    x0 = layers.Activation('relu')(x0)

    x1 = conv2d_pad(inputs, filters[1], 3, 2, 1, use_bias=False)
    x1 = layers.BatchNormalization(momentum=BN_MOMENTUM)(x1)
    x1 = layers.Activation('relu')(x1)

    return [x0, x1]

def basic_block(inputs, filters, strides=1, downsample=False):
    expansion = 1

    residual = inputs

    x = conv2d_pad(filters//expansion, 3, strides, 1, use_bias=False)(inputs)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM)(x) ## check batch axis=3
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters//expansion, 3, 1, 1, use_bias=False)(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM)(x)

    if downsample:
        residual = layers.Conv2D(filters, 1, strides, 0, use_bias=False)(inputs)
        residual = layers.BatchNormalization(momentum=BN_MOMENTUM)(residual)

    x = layers.Add()([x, residual])
    x = layers.Activation('relu')(x)

    return x







