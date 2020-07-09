"""
Implementation of HRNet Network paper:
https://arxiv.org/pdf/1902.09212.pdf
"""
import tensorflow as tf

layers = tf.keras.layers
initializers = tf.keras.initializers

BN_MOMENTUM = 0.1
CONV_BIAS = False
CONV_INIT = 'he_normal'
UP_MODE ='nearest'


def HRNet(input_shape, classes, weights=None):

    inputs = layers.Input(input_shape, name='input')

    # STAGE 1
    x = conv2d_bn(inputs, 64, 3, 2, padding='same', activation='relu',
                  name='stage1_stem_conv1')
    x = conv2d_bn(x, 64, 3, 2, padding='same', activation='relu',
                  name='stage1_stem_conv2')
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

    # create model
    model = tf.keras.Model(inputs, outputs)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model


def conv2d_bn(inputs, filters, kernel_size, strides=1, padding='valid',
              activation=None, name=None):
    block, idx = name.split('conv')
    x = layers.Conv2D(filters, kernel_size, strides, padding,
                      kernel_initializer=CONV_INIT,
                      use_bias=CONV_BIAS,
                      name=name)(inputs)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM,
                                  name=f'{block}bn{idx}')(x)
    if activation:
        x = layers.Activation(activation, name=f'{block}relu{idx}')(x)
    return x

def bottleneck_block(inputs, filters, strides=1, downsample=False,
                     name='bottleneck'):
    expansion = 4

    residual = inputs

    x = conv2d_bn(inputs, filters//expansion, 1, padding='same',
                  activation='relu', name=f'{name}_conv1')
    x = conv2d_bn(x, filters//expansion, 3, strides, padding='same',
                  activation='relu', name=f'{name}_conv2')
    x = conv2d_bn(x, filters, 1, 1, name=f'{name}_conv3')

    if downsample:
        residual = conv2d_bn(inputs, filters, 1, strides, name=f'{name}_down_conv')

    x = layers.Add(name=f'{name}_res')([x, residual])
    x = layers.Activation('relu', name=f'{name}_out')(x)

    return x

def transion_layer1(inputs, filters=[32, 64], name='stage1_transition'):
    x1 = conv2d_bn(inputs, filters[0], 3, 1, padding='same', activation='relu',
                   name=f'{name}_conv1')
    x2 = conv2d_bn(inputs, filters[1], 3, 2, padding='same', activation='relu',
                   name=f'{name}_conv2')
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

    x = conv2d_bn(inputs, filters//expansion, 3, strides, padding='same',
                  activation='relu', name=f'{name}_conv1')
    x = conv2d_bn(x, filters//expansion, 3, 1, padding='same',
                  name=f'{name}_conv2')

    if downsample:
        residual = conv2d_bn(inputs, filters, 1, strides,
                             name=f'{name}_down_conv')

    x = layers.Add(name=f'{name}_res')([x, residual])
    x = layers.Activation('relu', name=f'{name}_out')(x)

    return x

def fuse_layer1(inputs, filters=[32, 64], name='stage2_fuse'):
    x1, x2 = inputs

    x11 = x1

    x21 = conv2d_bn(x2, filters[0], 1, 1, name=f'{name}_conv_2_1')
    x21 = layers.UpSampling2D(2, name=f'{name}_up2_1')(x21)
    x1 = layers.Add(name=f'{name}_add1')([x11, x21])
    x1 = layers.Activation('relu', name=f'{name}_branch1_out')(x1)

    x22 = x2
    x12 = conv2d_bn(x1, filters[1], 3, 2, padding='same', name=f'{name}_conv1_2')
    x2 = layers.Add(name=f'{name}_add2')([x12, x22])
    x2 = layers.Activation('relu', name=f'{name}_branch2_out')(x2)

    return [x1, x2]

def transition_layer2(inputs, filters, name='stage2_transition'):
    x1, x2 = inputs
    x1 = conv2d_bn(x1, filters[0], 3, 1, padding='same', activation='relu',
                   name=f'{name}_conv1')
    x21 = conv2d_bn(x2, filters[1], 3, 1, padding='same', activation='relu',
                    name=f'{name}_conv2')
    x22 = conv2d_bn(x2, filters[2], 3, 2, padding='same', activation='relu',
                    name=f'{name}_conv3')
    return [x1, x21, x22]

def fuse_layer2(inputs, filters=[32, 64, 128], name='stage3_fuse'):
    x1, x2, x3 = inputs

    # branch 1
    x11 = x1

    x21 = conv2d_bn(x2, filters[0], 1, 1, name=f'{name}_conv2_1')
    x21 = layers.UpSampling2D(2, name=f'{name}_up2_1')(x21)

    x31 = conv2d_bn(x3, filters[0], 1, 1, name=f'{name}_conv3_1')
    x31 = layers.UpSampling2D(4, name=f'{name}_up3_1')(x31)

    x1 = layers.Add(name=f'{name}_add1')([x11, x21, x31])
    x1 = layers.Activation('relu', name=f'{name}_branch1_out')(x1)

    # branch 2
    x22 = x2

    x12 = conv2d_bn(x1, filters[1], 3, 2, padding='same', name=f'{name}_conv1_2')

    x32 = conv2d_bn(x3, filters[1], 1, 1, name=f'{name}_conv3_2')
    x32 = layers.UpSampling2D(2, name=f'{name}_up3_2')(x32)

    x2 = layers.Add(name=f'{name}_add2')([x12, x22, x32])
    x2 = layers.Activation('relu', name=f'{name}_branch2_out')(x2)

    # branch 3
    x33 = x3

    x13 = conv2d_bn(x1, filters[0], 3, 2, padding='same', activation='relu',
                    name=f'{name}_conv1_3_1')
    x13 = conv2d_bn(x13, filters[2], 3, 2, padding='same',
                    name=f'{name}_conv1_3_2')

    x23 = conv2d_bn(x2, filters[2], 3, 2, padding='same', name=f'{name}_conv2_3')

    x3 = layers.Add(name=f'{name}_add3')([x13, x23, x33])
    x3 = layers.Activation('relu', name=f'{name}_branch3_out')(x3)

    return [x1, x2, x3]

def transition_layer3(inputs, filters, name='stage3_transition'):
    x1, x2, x3 = inputs

    x1 = conv2d_bn(x1, filters[0], 3, 1, padding='same', activation='relu',
                   name=f'{name}_conv1')
    x2 = conv2d_bn(x2, filters[1], 3, 1, padding='same', activation='relu',
                   name=f'{name}_conv2')
    x31 = conv2d_bn(x3, filters[2], 3, 1, padding='same', activation='relu',
                    name=f'{name}_conv3')
    x32 = conv2d_bn(x3, filters[3], 3, 2, padding='same', activation='relu',
                    name=f'{name}_conv4')

    return [x1, x2, x31, x32]

def fuse_layer3(inputs, filters=[32, 64, 128, 256], name='stage4_fuse'):
    x1, x2, x3, x4 = inputs

    # branch 1
    x11 = x1

    x21 = conv2d_bn(x2, filters[0], 1, 1, name=f'{name}_conv2_1')
    x21 = layers.UpSampling2D(2, name=f'{name}_up2_1')(x21)

    x31 = conv2d_bn(x3, filters[0], 1, 1, name=f'{name}_conv3_1')
    x31 = layers.UpSampling2D(4, name=f'{name}_up3_1')(x31)

    x41 = conv2d_bn(x4, filters[0], 1, 1, name=f'{name}_conv4_1')
    x41 = layers.UpSampling2D(8, name=f'{name}_up4_1')(x41)

    x1 = layers.Add(name=f'{name}_add1')([x11, x21, x31, x41])
    x1 = layers.Activation('relu', name=f'{name}_branch1_out')(x1)

    # branch 2
    x22 = x2

    x12 = conv2d_bn(x1, filters[1], 3, 2, padding='same', name=f'{name}_conv1_2')

    x32 = conv2d_bn(x3, filters[1], 1, 1, name=f'{name}_conv3_2')
    x32 = layers.UpSampling2D(2, name=f'{name}_up3_2')(x32)

    x42 = conv2d_bn(x4, filters[1], 1, 1, name=f'{name}_conv4_2')
    x42 = layers.UpSampling2D(4, name=f'{name}_up4_2')(x42)

    x2 = layers.Add(name=f'{name}_add2')([x12, x22, x32, x42])
    x2 = layers.Activation('relu', name=f'{name}_branch2_out')(x2)

    # branch 3
    x33 = x3

    x13 = conv2d_bn(x1, filters[0], 3, 2, padding='same', activation='relu',
                    name=f'{name}_conv1_3_1')
    x13 = conv2d_bn(x13, filters[2], 3, 2, padding='same',
                    name=f'{name}_conv1_3_2')

    x23 = conv2d_bn(x2, filters[2], 3, 2, padding='same',
                    name=f'{name}_conv2_3')

    x43 = conv2d_bn(x4, filters[2], 1, 1, name=f'{name}_conv4_3')
    x43 = layers.UpSampling2D(2, name=f'{name}_up4_3')(x43)

    x3 = layers.Add(name=f'{name}_add3')([x13, x23, x33, x43])
    x3 = layers.Activation('relu', name=f'{name}_branch3_out')(x3)

    # branch 4
    x44 = x4

    x14 = conv2d_bn(x1, filters[0], 3, 2, padding='same', activation='relu',
                    name=f'{name}_conv1_4_1')
    x14 = conv2d_bn(x14, filters[0], 3, 2, padding='same', activation='relu',
                    name=f'{name}_conv1_4_2')
    x14 = conv2d_bn(x14, filters[3], 3, 2, padding='same',
                    name=f'{name}_conv1_4_3')

    x24 = conv2d_bn(x2, filters[1], 3, 2, padding='same', activation='relu',
                    name=f'{name}_conv2_4_1')
    x24 = conv2d_bn(x24, filters[3], 3, 2, padding='same',
                    name=f'{name}_conv2_4_2')

    x34 = conv2d_bn(x3, filters[3], 3, 2, padding='same', name=f'{name}_conv3_4')

    x4 = layers.Add(name=f'{name}_add4')([x14, x24, x34, x44])
    x4 = layers.Activation('relu', name=f'{name}_branch4_out')(x4)

    return [x1, x2, x3, x4]

def fuse_layer4(inputs, filters=32, name='final_fuse'):
    x1, x2, x3, x4 = inputs

    x11 = x1

    x21 = conv2d_bn(x2, filters, 1, 1, name=f'{name}_conv2_1')
    x21 = layers.UpSampling2D(2, name=f'{name}_up2_1')(x21)

    x31 = conv2d_bn(x3, filters, 1, 1, name=f'{name}_conv3_1')
    x31 = layers.UpSampling2D(4, name=f'{name}_up3_1')(x31)

    x41 = conv2d_bn(x4, filters, 1, 1, name=f'{name}_conv4_1')
    x41 = layers.UpSampling2D(8, name=f'{name}_up4_1')(x41)

    x = layers.Concatenate(name=f'{name}_out')([x11, x21, x31, x41])
    return x


if __name__=='__main__':
    # test
    # (256, 192) or (384, 288)
    model = HRNet((320, 320, 3), 17)
    print(model.summary())



