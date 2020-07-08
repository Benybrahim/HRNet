"""
Implementation of HRNet Network paper:
"""
import os
import logging

import tensorflow as tf

logger = logging.getLogger(__name__)
layers = tf.keras.layers
BN_MOMENTUM = 0.1


class Conv2DPad(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides, padding, use_bias=False):
        super(Conv2DPad, self).__init__()
        self.zero_pad = layers.ZeroPadding2D(padding)
        self.conv = layers.Conv2D(filters, kernel_size, strides, use_bias=use_bias)
    def call(self, inputs):
        x = self.zero_pad(inputs)
        x = self.conv(x)
        return x

class BasicBlock(layers.Layer):
    expansion = 1

    def __input__(self, filters, strides=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2DPad(filters, 3, strides, 1, use_bias=False)
        self.bn1 = layers.BatchNormalization(momentum=BN_MOMENTUM)
        self.relu = layers.Activation('relu')
        self.conv2 = Conv2DPad(filters, 3, 1, 1, use_bias=False)
        self.bn2 = layers.BatchNormalization(momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.strides = strides

    def call(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(tf.keras.Model):
    expansion = 4

    def __init__(self, filters, strides=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2DPad(filters, 1, 1, 0, use_bias=False)
        self.bn1 = layers.BatchNormalization(momentum=BN_MOMENTUM)
        self.conv2 = Conv2DPad(filters, 3, strides, 1, use_bias=False)
        self.bn2 = layers.BatchNormalization(momentum=BN_MOMENTUM)
        self.conv3 = Conv2DPad(filters*self.expansion, 1, 1, 0, use_bias=False)
        self.bn3 = layers.BatchNormalization(momentum=BN_MOMENTUM)
        self.relu = layers.Activation('relu')
        self.downsample = downsample
        self.strides = strides

    def call(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class HighResolutionModule(tf.keras.models.Model): # try tf.Module
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = layers.Activation('relu')

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         strides=1):
        downsample = None
        if strides != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = tf.keras.models.Sequential([
                Conv2DPad(num_channels[branch_index] * block.expansion, kernel=1, strides=strides, usee_bias=False),
                layers.BatchNormalization(momentum=BN_MOMENTUM)
            ])

        layers_ = []
        layers_.append(
            block(num_channels[branch_index], strides, downsample)
        )
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers_.append(
                block(num_channels[branch_index])
            )

        return tf.keras.models.Sequential(layers_)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return branches

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        tf.keras.models.Sequential([
                            Conv2DPad(num_inchannels[i], 1, 1, 0, use_bias=False),
                            layers.BatchNormalization(),
                            layers.UpSampling2D(size=2**(j-i), interpolation='nearest')
                        ])
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                tf.keras.models.Sequential([
                                    Conv2DPad(num_outchannels_conv3x3, 3, 2, 1, use_bias=False),
                                    layers.BatchNormalization()
                                ])
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                tf.keras.models.Sequential([
                                    Conv2DPad(num_outchannels_conv3x3, 3, 2, 1, use_bias=False),
                                    layers.BatchNormalization(),
                                    layers.Activation('relu')
                                ])
                            )
                    fuse_layer.append(tf.keras.models.Sequential(conv3x3s))
            fuse_layers.append(fuse_layer)

        return fuse_layers

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}

class PoseHighResolutionNet(tf.keras.models.Model):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA']
        super(PoseHighResolutionNet, self).__init__()

        # stem net
        self.conv1 = Conv2DPad(64, 3, 2, 1, use_bias=False)
        self.bn1 = layers.BatchNormalization(momentum=BN_MOMENTUM)
        self.conv2 = Conv2DPad(64, 3, 2, 1, use_bias=False)
        self.bn2 = layers.BatchNormalization(momentum=BN_MOMENTUM)
        self.relu = layers.Activation('relu')
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)

        self.final_layer = Conv2DPad(
            filters=cfg['MODEL']['NUM_JOINTS'],
            kernel_size=extra['FINAL_CONV_KERNEL'],
            strides=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )

        self.pretrained_layers = extra['PRETRAINED_LAYERS']

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        tf.keras.models.Sequential([
                            Conv2DPad(num_channels_cur_layer[i], 3, 1, 1, use_bias=False),
                            layers.BatchNormalization(num_channels_cur_layer[i]),
                            layers.Activation('relu')
                        ])
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        tf.keras.models.Sequential([
                            Conv2DPad(outchannels, 3, 2, 1, use_bias=False),
                            layers.BatchNormalization(outchannels),
                            layers.Activation('relu')
                        ])
                    )
                transition_layers.append(tf.keras.models.Sequential(conv3x3s))

        return transition_layers

    def _make_layer(self, block, planes, blocks, strides=1):
        downsample = None
        if strides != 1 or self.inplanes != planes * block.expansion:
            downsample = tf.keras.models.Sequential([
                Conv2DPad(planes * block.expansion, 1, strides, use_bias=False),
                layers.BatchNormalization(momentum=BN_MOMENTUM)
            ])
        layers_ = []
        layers_.append(block(planes, strides, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers_.append(block(planes))

        return tf.keras.models.Sequential(layers_)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return tf.keras.models.Sequential(modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        x = self.final_layer(y_list[0])

        return x

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for layer in self.modules():
            if isinstance(layer, Conv2DPad):
                layer.Conv2D()
                layer.w = tf.keras.initializers.RandomNormal(std=0.001)
                for name, _ in layer.named_parameters():
                    if name in ['bias']:
                        layer.bias = tf.keras.initializers.Constant(value=0)
            elif isinstance(layer, layers.BatchNormalization):
                layer.weight = tf.keras.initializers.Constant(value=1)
                layer.bias = tf.keras.initializers.Constant(value=0)
            elif isinstance(layer, layers.Conv2DTranspose):
                layer.w = tf.keras.initializers.RandomNormal(std=0.001)
                for name, _ in layer.named_parameters():
                    if name in ['bias']:
                        layer.bias = tf.keras.initializers.Constant(value=0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = tf.keras.models.load_model(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHighResolutionNet(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.load_weights(cfg['MODEL']['PRETRAINED'])

    return model



if __name__=='__main__':
    # test
    def aa():
        coco = 2
        return None
    go = aa()
    print(go.coco)
    #model.build((1, 320, 320, 3))
    #BasicBlock(filters, strides=1, downsample=None)
    #Bottleneck(strides=1, downsample=None)
    #HighResolutionModule(num_branches, blocks, num_blocks, num_inchannels,
    #                     num_channels, fuse_method, multi_scale_output=True)
    #print(model.summary())

