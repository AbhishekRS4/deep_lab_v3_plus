# @author : Abhishek R S

import os
import h5py
import numpy as np
import tensorflow as tf

"""
DeepLabv3+

# Reference
- [Deep Residual Learning for Image Recognition]
  (https://arxiv.org/abs/1512.03385)
- [DeepLabv3+](https://arxiv.org/pdf/1802.02611.pdf)

# Pretrained model weights
- [Download pretrained resnet-50 model]
  (https://github.com/fchollet/deep-learning-models/releases/)
"""

class DeepLab3Plus:
    def __init__(self, pretrained_weights, is_training, data_format="channels_first", num_classes=15):
        self._weights_h5 = h5py.File(pretrained_weights, "r")
        self._is_training = is_training
        self._data_format = data_format
        self._num_classes = num_classes
        self._padding = "SAME"
        self._feature_map_axis = None
        self._encoder_data_format = None
        self._encoder_pool_kernel = None
        self._input_size = [512, 1024]
        self._atrous_rate = [6, 12, 18]
        self._encoder_conv_strides = [1, 1, 1, 1]
        self._encoder_pool_strides = None
        self._avg_pool_axes = None
        self._initializer = tf.contrib.layers.xavier_initializer_conv2d()

        """
        based on the data format set appropriate pool_kernel and pool_strides
        always use channels_first i.e. NCHW as the data format on a GPU
        """

        if data_format == "channels_first":
            self._encoder_data_format = "NCHW"
            self._encoder_pool_kernel = [1, 1, 3, 3]
            self._encoder_pool_strides = [1, 1, 2, 2]
            self._avg_pool_axes = [2, 3]
            self._feature_map_axis = 1
        else:
            self._encoder_data_format = "NHWC"
            self._encoder_pool_kernel = [1, 3, 3, 1]
            self._encoder_pool_strides = [1, 2, 2, 1]
            self._avg_pool_axes = [1, 2]
            self._feature_map_axis = -1

    # build resnet-50 encoder
    def resnet50_encoder(self, features):
        # input : BGR format with image_net mean subtracted
        # bgr mean : [103.939, 116.779, 123.68]

        if self._data_format == "channels_last":
            features = tf.transpose(features, perm=[0, 2, 3, 1])

        # Stage 0
        self.stage0 = self._res_conv_layer(
            features, "conv1", strides=self._encoder_pool_strides)
        self.stage0 = self._res_batchnorm_layer(self.stage0, "bn_conv1")
        self.stage0 = tf.nn.relu(self.stage0, name="relu1")

        # Stage 1
        self.stage1 = tf.nn.max_pool(
            self.stage0, ksize=self._encoder_pool_kernel, strides=self._encoder_pool_strides,
            padding=self._padding, data_format=self._encoder_data_format, name="pool1"
        )

        # Stage 2
        self.stage2 = self._res_conv_block(
            input_layer=self.stage1, stage="2a", strides=self._encoder_conv_strides)
        self.stage2 = self._res_identity_block(input_layer=self.stage2, stage="2b")
        self.stage2 = self._res_identity_block(input_layer=self.stage2, stage="2c")

        # Stage 3
        self.stage3 = self._res_conv_block(
            input_layer=self.stage2, stage="3a", strides=self._encoder_pool_strides)
        self.stage3 = self._res_identity_block(input_layer=self.stage3, stage="3b")
        self.stage3 = self._res_identity_block(input_layer=self.stage3, stage="3c")
        self.stage3 = self._res_identity_block(input_layer=self.stage3, stage="3d")

        # Stage 4
        self.stage4 = self._res_conv_block(
            input_layer=self.stage3, stage="4a", strides=self._encoder_pool_strides)
        self.stage4 = self._res_identity_block(input_layer=self.stage4, stage="4b")
        self.stage4 = self._res_identity_block(input_layer=self.stage4, stage="4c")
        self.stage4 = self._res_identity_block(input_layer=self.stage4, stage="4d")
        self.stage4 = self._res_identity_block(input_layer=self.stage4, stage="4e")
        self.stage4 = self._res_identity_block(input_layer=self.stage4, stage="4f")

        # Stage 5
        self.stage5 = self._res_conv_block(
            input_layer=self.stage4, stage="5a", strides=self._encoder_conv_strides)
        self.stage5 = self._res_identity_block(input_layer=self.stage5, stage="5b")
        self.stage5 = self._res_identity_block(input_layer=self.stage5, stage="5c")

    # build deep_lab_v3+
    def deeplabv3_plus(self):
        self.aspp_out = self._atrous_spatial_pyramid_pool_block(self.stage5, name="aspp_")

        self.low_level_features = self._get_conv2d_layer(
            self.stage2, 48, [1, 1], [1, 1], name="low_level_conv")
        self.low_level_features = self._get_relu_activation(
            self.low_level_features, name="low_level_relu")

        if self._data_format == "channels_first":
            low_level_features_size = tf.shape(self.low_level_features)[2:]
        else:
            low_level_features_size = tf.shape(self.low_level_features)[1:3]

        self.up1 = self._get_upsample_layer(
            self.aspp_out, low_level_features_size, name="upsample1")
        self.up1_concat = tf.concat([self.up1, self.low_level_features],
            axis=self._feature_map_axis, name="decoder_concat")

        self.decoder_conv1 = self._get_conv2d_layer(
            self.up1_concat, 256, [3, 3], [1, 1], name="decoder_conv1")
        self.decoder_conv1 = self._get_relu_activation(self.decoder_conv1, name="decoder_relu1")
        self.decoder_conv2 = self._get_conv2d_layer(
            self.decoder_conv1, 256, [3, 3], [1, 1], name="decoder_conv2")
        self.decoder_conv2 = self._get_relu_activation(self.decoder_conv2, name="decoder_relu2")
        self.decoder_conv3 = self._get_conv2d_layer(
            self.decoder_conv2, self._num_classes, [1, 1], [1, 1], name="decoder_conv3")

        self.logits = self._get_upsample_layer(
            self.decoder_conv3, self._input_size, name="logits")

    # build atrous spatial pyramid pool block
    def _atrous_spatial_pyramid_pool_block(self, input_layer, depth=256, name="aspp_"):
        if self._data_format == "channels_first":
            _inputs_size = tf.shape(input_layer)[2:]
        else:
            _inputs_size = tf.shape(input_layer)[1:3]

        _conv1x1 = self._get_conv2d_layer(
            input_layer, depth, [1, 1], [1, 1], name=name + "conv1x1")
        _conv1x1 = self._get_relu_activation(_conv1x1, name=name + "conv1x1_relu")

        _conv3x3_1 = self._get_conv2d_layer(input_layer, depth, [3, 3], [1, 1],
            dilation_rate=self._atrous_rate[0], name=name + "conv3x3_1")
        _conv3x3_1 = self._get_relu_activation(_conv3x3_1, name=name + "conv3x3_1_relu")

        _conv3x3_2 = self._get_conv2d_layer(input_layer, depth, [3, 3], [1, 1],
            dilation_rate=self._atrous_rate[1], name=name + "conv3x3_2")
        _conv3x3_2 = self._get_relu_activation(_conv3x3_2, name=name + "conv3x3_2_relu")

        _conv3x3_3 = self._get_conv2d_layer(input_layer, depth, [3, 3], [1, 1],
            dilation_rate=self._atrous_rate[2], name=name + "conv3x3_3")
        _conv3x3_3 = self._get_relu_activation(_conv3x3_3, name=name + "conv3x3_3_relu")

        _avg_pool = tf.reduce_mean(input_layer, self._avg_pool_axes,
            name=name + "avg_pool", keepdims=True)
        _img_lvl_conv1x1 = self._get_conv2d_layer(
            _avg_pool, depth, [1, 1], [1, 1], name=name + "img_lvl_conv1x1")
        _img_lvl_conv1x1 = self._get_relu_activation(
            _img_lvl_conv1x1, name=name + "img_lvl_conv1x1_relu")

        _img_lvl_upsample = self._get_upsample_layer(
            _img_lvl_conv1x1, _inputs_size, name=name + "upsample")

        _concat_features = tf.concat(
            [_conv1x1, _conv3x3_1, _conv3x3_2, _conv3x3_3, _img_lvl_upsample],
            axis=self._feature_map_axis, name=name + "concat")
        _conv1x1_aspp_out = self._get_conv2d_layer(_concat_features, depth,
            [1, 1], [1, 1], name=name + "out_conv1x1")
        _conv1x1_aspp_out = self._get_relu_activation(
            _conv1x1_aspp_out, name=name + "out_conv1x1_relu")
        _conv1x1_aspp_out = self._get_dropout_layer(
            _conv1x1_aspp_out, rate=0.1, name=name + "dropout")

        return _conv1x1_aspp_out

    # return convolution2d layer
    def _get_conv2d_layer(self, input_layer, num_filters, kernel_size, strides, dilation_rate=1, use_bias=True, name="conv"):
        conv_2d_layer = tf.layers.conv2d(inputs=input_layer, filters=num_filters, kernel_size=kernel_size,
            strides=strides, use_bias=use_bias, padding=self._padding, data_format=self._data_format,
            kernel_initializer=self._initializer, dilation_rate=dilation_rate, name=name)
        return conv_2d_layer

    # return bilinear upsampling layer
    def _get_upsample_layer(self, input_layer, target_size, name="upsample"):
        if self._data_format == "channels_first":
            input_layer = tf.transpose(input_layer, perm=[0, 2, 3, 1])

        _upsampled = tf.image.resize_bilinear(input_layer, target_size, name=name)

        if self._data_format == "channels_first":
            _upsampled = tf.transpose(_upsampled, perm=[0, 3, 1, 2])

        return _upsampled

    # return relu activation function
    def _get_relu_activation(self, input_layer, name="relu"):
        relu_layer = tf.nn.relu(input_layer, name=name)
        return relu_layer

    # return dropout layer
    def _get_dropout_layer(self, input_layer, rate=0.1, name="dropout"):
        dropout_layer = tf.layers.dropout(inputs=input_layer, rate=rate, training=self._is_training, name=name)
        return dropout_layer

    # return batch normalization layer
    def _get_batchnorm_layer(self, input_layer, name="bn"):
        bn_layer = tf.layers.batch_normalization(input_layer, axis=self._feature_map_axis, training=self._is_training, name=name)
        return bn_layer

    #---------------------------------------#
    # pretrained resnet50 encoder functions #
    #---------------------------------------#
    #-----------------------#
    # convolution layer     #
    #-----------------------#
    def _res_conv_layer(self, input_layer, name, strides=[1, 1, 1, 1]):
        W = tf.constant(self._weights_h5[name][name + "_W_1:0"])
        b = self._weights_h5[name][name + "_b_1:0"]
        b = tf.constant(np.reshape(b, (b.shape[0])))
        x = tf.nn.conv2d(input_layer, filter=W, strides=strides,
             padding=self._padding, data_format=self._encoder_data_format, name=name)
        x = tf.nn.bias_add(x, b, data_format=self._encoder_data_format)

        return x

    #-----------------------#
    # batchnorm layer       #
    #-----------------------#
    def _res_batchnorm_layer(self, input_layer, name):
        if self._encoder_data_format == "NCHW":
            input_layer = tf.transpose(input_layer, perm=[0, 2, 3, 1])

        mean = tf.constant(self._weights_h5[name][name + "_running_mean_1:0"])
        std = tf.constant(self._weights_h5[name][name + "_running_std_1:0"])
        beta = tf.constant(self._weights_h5[name][name + "_beta_1:0"])
        gamma = tf.constant(self._weights_h5[name][name + "_gamma_1:0"])

        bn = tf.nn.batch_normalization(input_layer, mean=mean, variance=std,
            offset=beta, scale=gamma, variance_epsilon=1e-12, name=name)

        if self._encoder_data_format == "NCHW":
            bn = tf.transpose(bn, perm=[0, 3, 1, 2])

        return bn

    #-----------------------#
    # convolution block     #
    #-----------------------#
    def _res_conv_block(self, input_layer, stage, strides):
        x = self._res_conv_layer(input_layer, name="res" + stage + "_branch2a", strides=strides)
        x = self._res_batchnorm_layer(x, name="bn" + stage + "_branch2a")
        x = tf.nn.relu(x, name="relu" + stage + "_branch2a")

        x = self._res_conv_layer(x, name="res" + stage + "_branch2b")
        x = self._res_batchnorm_layer(x, name="bn" + stage + "_branch2b")
        x = tf.nn.relu(x, name="relu" + stage + "_branch2b")

        x = self._res_conv_layer(x, name="res" + stage + "_branch2c")
        x = self._res_batchnorm_layer(x, name="bn" + stage + "_branch2c")

        shortcut = self._res_conv_layer(input_layer, name="res" + stage + "_branch1", strides=strides)
        shortcut = self._res_batchnorm_layer(shortcut, name="bn" + stage + "_branch1")

        x = tf.add(x, shortcut, name="add" + stage)
        x = tf.nn.relu(x, name="relu" + stage)

        return x

    #-----------------------#
    # identity block        #
    #-----------------------#
    def _res_identity_block(self, input_layer, stage):
        x = self._res_conv_layer(input_layer, name="res" + stage + "_branch2a")
        x = self._res_batchnorm_layer(x, name="bn" + stage + "_branch2a")
        x = tf.nn.relu(x, name="relu" + stage + "_branch2a")

        x = self._res_conv_layer(x, name="res" + stage + "_branch2b")
        x = self._res_batchnorm_layer(x, name="bn" + stage + "_branch2b")
        x = tf.nn.relu(x, name="relu" + stage + "_branch2b")

        x = self._res_conv_layer(x, name="res" + stage + "_branch2c")
        x = self._res_batchnorm_layer(x, name="bn" + stage + "_branch2c")

        x = tf.add(x, input_layer, name="add" + stage)
        x = tf.nn.relu(x, name="relu" + stage)

        return x
