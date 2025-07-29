import tensorflow as tf
from tensorflow.keras import layers


class UpconvBlock(tf.keras.layers.Layer):
    """Upconvolution block for U-Net decoder in TensorFlow v2"""

    def __init__(self, in_channels, out_channels, weight_decay=0.0001, **kwargs):
        super(UpconvBlock, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_decay = weight_decay

        # Batch normalization parameters
        self.bn_params = {
            "momentum": 0.997,
            "epsilon": 1e-5,
        }

        # Define layers
        self.conv1 = layers.Conv2D(
            filters=out_channels,
            kernel_size=3,
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            name="conv1",
        )

        self.bn1 = layers.BatchNormalization(**self.bn_params, name="bn1")

        self.conv2 = layers.Conv2D(
            filters=out_channels,
            kernel_size=3,
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            name="conv2",
        )

        self.bn2 = layers.BatchNormalization(**self.bn_params, name="bn2")

        self.activation = layers.ReLU()

    def call(self, inputs, training=False):
        """Forward pass"""
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.activation(x)

        return x

    def get_config(self):
        """Get layer configuration for serialization"""
        config = super(UpconvBlock, self).get_config()
        config.update(
            {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "weight_decay": self.weight_decay,
            }
        )
        return config


def upconvBlock(inputs, in_channels, out_channels, weight_decay=0.0001, training=True):
    """Legacy function wrapper for backward compatibility"""
    block = UpconvBlock(in_channels, out_channels, weight_decay)
    return block(inputs, training=training)


def upsample(inputs, size, method="bilinear"):
    """Upsample feature maps using bilinear interpolation"""
    return tf.image.resize(inputs, size, method=method)


def conv2d_transpose_strided(
    inputs,
    filters,
    kernel_size,
    stride=2,
    padding="same",
    activation="relu",
    weight_decay=0.0001,
    name=None,
):
    """Transposed convolution with stride for upsampling"""
    return layers.Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        name=name,
    )(inputs)


class DeconvBlock(tf.keras.layers.Layer):
    """Deconvolution block using transposed convolution"""

    def __init__(self, filters, kernel_size=3, stride=2, weight_decay=0.0001, **kwargs):
        super(DeconvBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight_decay = weight_decay

        self.deconv = layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            name="deconv",
        )

        self.bn = layers.BatchNormalization(momentum=0.997, epsilon=1e-5, name="bn")

        self.activation = layers.ReLU()

    def call(self, inputs, training=False):
        """Forward pass"""
        x = self.deconv(inputs)
        x = self.bn(x, training=training)
        x = self.activation(x)
        return x

    def get_config(self):
        """Get layer configuration for serialization"""
        config = super(DeconvBlock, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "weight_decay": self.weight_decay,
            }
        )
        return config


class FeatureFusionBlock(tf.keras.layers.Layer):
    """Feature fusion block for combining features from different scales"""

    def __init__(self, out_channels, weight_decay=0.0001, **kwargs):
        super(FeatureFusionBlock, self).__init__(**kwargs)
        self.out_channels = out_channels
        self.weight_decay = weight_decay

        self.conv = layers.Conv2D(
            filters=out_channels,
            kernel_size=1,
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            name="fusion_conv",
        )

        self.bn = layers.BatchNormalization(
            momentum=0.997, epsilon=1e-5, name="fusion_bn"
        )

        self.activation = layers.ReLU()

    def call(self, inputs, training=False):
        """Forward pass - inputs should be a list of tensors to fuse"""
        if isinstance(inputs, list):
            # Concatenate along channel dimension
            x = tf.concat(inputs, axis=-1)
        else:
            x = inputs

        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.activation(x)

        return x

    def get_config(self):
        """Get layer configuration for serialization"""
        config = super(FeatureFusionBlock, self).get_config()
        config.update(
            {
                "out_channels": self.out_channels,
                "weight_decay": self.weight_decay,
            }
        )
        return config
