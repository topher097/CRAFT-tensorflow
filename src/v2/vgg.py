# import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras.applications import VGG16


# class VGG16Backbone(tf.keras.Model):
#     """VGG16 backbone for CRAFT network using TensorFlow v2"""

#     def __init__(self, pretrained=True, **kwargs):
#         super(VGG16Backbone, self).__init__(**kwargs)

#         # Load pretrained VGG16 without top layers
#         self.vgg16 = VGG16(
#             weights="imagenet" if pretrained else None,
#             include_top=False,
#             input_shape=(None, None, 3),
#         )

#         # Make VGG16 layers non-trainable initially (can be fine-tuned later)
#         for layer in self.vgg16.layers:
#             layer.trainable = True

#         # Extract specific layers for feature extraction
#         self.feature_layers = {
#             "conv1_1": "block1_conv1",
#             "conv1_2": "block1_conv2",
#             "conv2_1": "block2_conv1",
#             "conv2_2": "block2_conv2",
#             "conv3_1": "block3_conv1",
#             "conv3_2": "block3_conv2",
#             "conv3_3": "block3_conv3",
#             "conv4_1": "block4_conv1",
#             "conv4_2": "block4_conv2",
#             "conv4_3": "block4_conv3",
#             "conv5_1": "block5_conv1",
#             "conv5_2": "block5_conv2",
#             "conv5_3": "block5_conv3",
#         }

#     def call(self, inputs, training=False):
#         """Forward pass through VGG16 backbone"""
#         features = {}
#         x = inputs

#         # Forward pass through VGG16 and collect intermediate features
#         for layer in self.vgg16.layers:
#             x = layer(x)

#             # Store features from specific layers
#             for feature_name, layer_name in self.feature_layers.items():
#                 if layer.name == layer_name:
#                     features[feature_name] = x

#         return features

#     def load_pretrained_weights(self, weights_path):
#         """Load pretrained weights from checkpoint"""
#         # This would need to be implemented based on your specific checkpoint format
#         # For now, using ImageNet weights from Keras applications
#         pass


# def vgg_16(
#     inputs,
#     num_classes=1000,
#     is_training=True,
#     dropout_keep_prob=0.5,
#     spatial_squeeze=True,
#     scope="vgg_16",
#     fc_conv_padding="VALID",
#     global_pool=False,
# ):
#     """Legacy VGG16 function for backward compatibility"""

#     # Create VGG16 backbone
#     vgg_backbone = VGG16Backbone(pretrained=True)

#     # Get features
#     features = vgg_backbone(inputs, training=is_training)

#     # Create end_points dictionary for compatibility
#     end_points = {}
#     for key, value in features.items():
#         end_points[f"vgg_16/{key}"] = value

#     # Return the final feature map and end_points
#     final_features = features["conv5_3"]

#     return final_features, end_points


# def vgg_arg_scope(weight_decay=0.0005):
#     """VGG argument scope for backward compatibility"""
#     # In TF2, this is handled by the model architecture directly
#     # Return empty dict for compatibility
#     return {}


# class CustomVGG16(tf.keras.Model):
#     """Custom VGG16 implementation with more control over layers"""

#     def __init__(self, weight_decay=0.0005, **kwargs):
#         super(CustomVGG16, self).__init__(**kwargs)
#         self.weight_decay = weight_decay

#         # Block 1
#         self.conv1_1 = layers.Conv2D(
#             64,
#             3,
#             padding="same",
#             activation="relu",
#             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
#             name="conv1_1",
#         )
#         self.conv1_2 = layers.Conv2D(
#             64,
#             3,
#             padding="same",
#             activation="relu",
#             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
#             name="conv1_2",
#         )
#         self.pool1 = layers.MaxPool2D(2, strides=2, name="pool1")

#         # Block 2
#         self.conv2_1 = layers.Conv2D(
#             128,
#             3,
#             padding="same",
#             activation="relu",
#             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
#             name="conv2_1",
#         )
#         self.conv2_2 = layers.Conv2D(
#             128,
#             3,
#             padding="same",
#             activation="relu",
#             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
#             name="conv2_2",
#         )
#         self.pool2 = layers.MaxPool2D(2, strides=2, name="pool2")

#         # Block 3
#         self.conv3_1 = layers.Conv2D(
#             256,
#             3,
#             padding="same",
#             activation="relu",
#             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
#             name="conv3_1",
#         )
#         self.conv3_2 = layers.Conv2D(
#             256,
#             3,
#             padding="same",
#             activation="relu",
#             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
#             name="conv3_2",
#         )
#         self.conv3_3 = layers.Conv2D(
#             256,
#             3,
#             padding="same",
#             activation="relu",
#             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
#             name="conv3_3",
#         )
#         self.pool3 = layers.MaxPool2D(2, strides=2, name="pool3")

#         # Block 4
#         self.conv4_1 = layers.Conv2D(
#             512,
#             3,
#             padding="same",
#             activation="relu",
#             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
#             name="conv4_1",
#         )
#         self.conv4_2 = layers.Conv2D(
#             512,
#             3,
#             padding="same",
#             activation="relu",
#             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
#             name="conv4_2",
#         )
#         self.conv4_3 = layers.Conv2D(
#             512,
#             3,
#             padding="same",
#             activation="relu",
#             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
#             name="conv4_3",
#         )
#         self.pool4 = layers.MaxPool2D(2, strides=2, name="pool4")

#         # Block 5
#         self.conv5_1 = layers.Conv2D(
#             512,
#             3,
#             padding="same",
#             activation="relu",
#             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
#             name="conv5_1",
#         )
#         self.conv5_2 = layers.Conv2D(
#             512,
#             3,
#             padding="same",
#             activation="relu",
#             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
#             name="conv5_2",
#         )
#         self.conv5_3 = layers.Conv2D(
#             512,
#             3,
#             padding="same",
#             activation="relu",
#             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
#             name="conv5_3",
#         )
#         self.pool5 = layers.MaxPool2D(2, strides=2, name="pool5")

#     def call(self, inputs, training=False):
#         """Forward pass through custom VGG16"""
#         features = {}

#         # Block 1
#         x = self.conv1_1(inputs)
#         features["conv1_1"] = x
#         x = self.conv1_2(x)
#         features["conv1_2"] = x
#         x = self.pool1(x)

#         # Block 2
#         x = self.conv2_1(x)
#         features["conv2_1"] = x
#         x = self.conv2_2(x)
#         features["conv2_2"] = x
#         x = self.pool2(x)

#         # Block 3
#         x = self.conv3_1(x)
#         features["conv3_1"] = x
#         x = self.conv3_2(x)
#         features["conv3_2"] = x
#         x = self.conv3_3(x)
#         features["conv3_3"] = x
#         x = self.pool3(x)

#         # Block 4
#         x = self.conv4_1(x)
#         features["conv4_1"] = x
#         x = self.conv4_2(x)
#         features["conv4_2"] = x
#         x = self.conv4_3(x)
#         features["conv4_3"] = x
#         x = self.pool4(x)

#         # Block 5
#         x = self.conv5_1(x)
#         features["conv5_1"] = x
#         x = self.conv5_2(x)
#         features["conv5_2"] = x
#         x = self.conv5_3(x)
#         features["conv5_3"] = x
#         x = self.pool5(x)

#         return features

#     def get_config(self):
#         """Get model configuration for serialization"""
#         config = super(CustomVGG16, self).get_config()
#         config.update(
#             {
#                 "weight_decay": self.weight_decay,
#             }
#         )
#         return config
import tensorflow as tf
from keras.applications import VGG16
from keras.models import Model


class VGG16Backbone(tf.keras.Model):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__(**kwargs)
        # Load VGG16 with imagenet weights, no top
        base_model = VGG16(
            weights="imagenet" if pretrained else None,
            include_top=False,
            input_shape=(None, None, 3),
        )
        # Specify the layers you want to extract
        layer_names = [
            "block2_conv2",  # conv2_2
            "block3_conv3",  # conv3_3
            "block4_conv3",  # conv4_3
            "block5_conv3",  # conv5_3
        ]
        outputs = [base_model.get_layer(name).output for name in layer_names]
        self.feature_extractor = Model(inputs=base_model.input, outputs=outputs)
        for layer in self.feature_extractor.layers:
            layer.trainable = True

    def call(self, inputs, training=False):
        features = self.feature_extractor(inputs, training=training)
        # Return as a dict for compatibility
        return {
            "conv2_2": features[0],
            "conv3_3": features[1],
            "conv4_3": features[2],
            "conv5_3": features[3],
        }
