import tensorflow as tf
from keras import layers

from v2.upconvBlock import UpconvBlock
from v2.vgg import VGG16Backbone


class CRAFTNet(tf.keras.Model):
    """CRAFT Network implemented in TensorFlow v2 using Keras Model API"""

    def __init__(self, weight_decay=0.0001, **kwargs):
        super(CRAFTNet, self).__init__(**kwargs)
        self.weight_decay = weight_decay

        # VGG16 backbone
        self.vgg_backbone = VGG16Backbone()

        # Additional layers after VGG
        self.pool5 = layers.MaxPool2D(
            pool_size=(3, 3), strides=1, padding="same", name="pool5"
        )

        # Atrous convolution layer
        self.atrous_conv = layers.Conv2D(
            filters=1024,
            kernel_size=3,
            dilation_rate=6,
            padding="same",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            name="atrous_conv",
        )

        self.conv6 = layers.Conv2D(
            filters=1024,
            kernel_size=1,
            padding="same",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            name="conv6",
        )

        # U-Net upsampling blocks
        self.upconv_block1 = UpconvBlock(512, 256, weight_decay=weight_decay)
        self.upconv_block2 = UpconvBlock(256, 128, weight_decay=weight_decay)
        self.upconv_block3 = UpconvBlock(128, 64, weight_decay=weight_decay)
        self.upconv_block4 = UpconvBlock(64, 32, weight_decay=weight_decay)

        # Final convolution layers
        self.final_conv1 = layers.Conv2D(
            filters=32,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            name="final_conv1",
        )

        self.final_conv2 = layers.Conv2D(
            filters=32,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            name="final_conv2",
        )

        self.final_conv3 = layers.Conv2D(
            filters=16,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            name="final_conv3",
        )

        self.final_conv4 = layers.Conv2D(
            filters=16,
            kernel_size=1,
            padding="same",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            name="final_conv4",
        )

        # Output layer (2 channels: text region and affinity)
        self.output_conv = layers.Conv2D(
            filters=2,
            kernel_size=1,
            padding="same",
            activation="sigmoid",  # Sigmoid for probability outputs
            name="output_conv",
        )

        # Batch normalization parameters
        self.bn_params = {
            "momentum": 0.997,
            "epsilon": 1e-5,
        }

    def call(self, inputs, training=False):
        """Forward pass of the CRAFT network"""

        # VGG16 feature extraction
        vgg_features = self.vgg_backbone(inputs, training=training)

        # Extract features at different scales
        f = [
            vgg_features["conv2_2"],  # 1/4 scale
            vgg_features["conv3_3"],  # 1/8 scale
            vgg_features["conv4_3"],  # 1/16 scale
            vgg_features["conv5_3"],  # 1/32 scale
        ]

        # Start with the deepest features
        net = f[3]  # conv5_3

        # Additional processing after VGG
        net = self.pool5(net)
        net = self.atrous_conv(net)
        net = self.conv6(net)

        # U-Net decoder with skip connections
        # First upsampling block
        net = tf.concat([net, f[3]], axis=3)  # Concatenate with conv5_3
        net = self.upconv_block1(net, training=training)
        net = tf.image.resize(net, (64, 64), method="bilinear")  # Upsample to 1/8

        # Second upsampling block
        net = tf.concat([net, f[2]], axis=3)  # Concatenate with conv4_3
        net = self.upconv_block2(net, training=training)
        net = tf.image.resize(net, (128, 128), method="bilinear")  # Upsample to 1/4

        # Third upsampling block
        net = tf.concat([net, f[1]], axis=3)  # Concatenate with conv3_3
        net = self.upconv_block3(net, training=training)
        net = tf.image.resize(net, (256, 256), method="bilinear")  # Upsample to 1/2

        # Fourth upsampling block
        net = tf.concat([net, f[0]], axis=3)  # Concatenate with conv2_2
        net = self.upconv_block4(net, training=training)

        # Final convolution layers
        net = self.final_conv1(net)
        net = self.final_conv2(net)
        net = self.final_conv3(net)
        net = self.final_conv4(net)

        # Output layer
        output = self.output_conv(net)

        return output

    def get_config(self):
        """Get model configuration for serialization"""
        config = super(CRAFTNet, self).get_config()
        config.update(
            {
                "weight_decay": self.weight_decay,
            }
        )
        return config


def create_craft_model(input_shape=(512, 512, 3), weight_decay=0.0001):
    """Factory function to create CRAFT model"""
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    model = CRAFTNet(weight_decay=weight_decay)
    outputs = model(inputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="CRAFT")


# Legacy function for backward compatibility
def CRAFT_net(inputs, is_training=True, reuse=None, weight_decay=0.0001):
    """Legacy function wrapper for backward compatibility"""
    model = CRAFTNet(weight_decay=weight_decay)
    outputs = model(inputs, training=is_training)

    # Return outputs and empty end_points dict for compatibility
    end_points = {}
    return outputs, end_points
