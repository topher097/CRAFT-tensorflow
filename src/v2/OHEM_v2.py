import tensorflow as tf


class MSE_OHEM_Loss(tf.keras.losses.Loss):
    """Online Hard Example Mining (OHEM) MSE Loss for TensorFlow v2"""

    def __init__(
        self, ratio=3.0, reduction=tf.keras.losses.Reduction.AUTO, name="mse_ohem_loss"
    ):
        """
        Initialize OHEM MSE Loss

        Args:
            ratio: Ratio of hard examples to keep (e.g., 3.0 means keep top 1/3 hardest examples)
            reduction: Type of reduction to apply to loss
            name: Name of the loss function
        """
        super(MSE_OHEM_Loss, self).__init__(reduction=reduction, name=name)
        self.ratio = ratio

    def call(self, y_true, y_pred):
        """
        Compute OHEM MSE loss

        Args:
            y_true: Ground truth labels [batch_size, height, width, channels]
            y_pred: Predicted labels [batch_size, height, width, channels]

        Returns:
            OHEM MSE loss value
        """
        # Compute pixel-wise MSE loss
        mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)

        # Flatten the loss for easier processing
        batch_size = tf.shape(y_true)[0]
        loss_flat = tf.reshape(mse_loss, [batch_size, -1])

        # Calculate number of pixels to keep based on ratio
        num_pixels = tf.shape(loss_flat)[1]
        num_hard = tf.cast(tf.cast(num_pixels, tf.float32) / self.ratio, tf.int32)
        num_hard = tf.maximum(num_hard, 1)  # Keep at least 1 pixel

        # Sort losses and keep only the hardest examples
        sorted_loss, _ = tf.nn.top_k(loss_flat, k=num_hard, sorted=True)

        # Return mean of hard examples
        return tf.reduce_mean(sorted_loss)

    def get_config(self):
        """Get configuration for serialization"""
        config = super(MSE_OHEM_Loss, self).get_config()
        config.update({"ratio": self.ratio})
        return config


class WeightedMSELoss(tf.keras.losses.Loss):
    """Weighted MSE Loss for text detection"""

    def __init__(
        self,
        pos_weight=1.0,
        neg_weight=1.0,
        reduction=tf.keras.losses.Reduction.AUTO,
        name="weighted_mse_loss",
    ):
        """
        Initialize Weighted MSE Loss

        Args:
            pos_weight: Weight for positive samples
            neg_weight: Weight for negative samples
            reduction: Type of reduction to apply to loss
            name: Name of the loss function
        """
        super(WeightedMSELoss, self).__init__(reduction=reduction, name=name)
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def call(self, y_true, y_pred):
        """
        Compute weighted MSE loss

        Args:
            y_true: Ground truth labels [batch_size, height, width, channels]
            y_pred: Predicted labels [batch_size, height, width, channels]

        Returns:
            Weighted MSE loss value
        """
        # Compute pixel-wise MSE loss
        mse_loss = tf.square(y_true - y_pred)

        # Create weight mask based on ground truth
        pos_mask = tf.cast(y_true > 0.5, tf.float32)
        neg_mask = tf.cast(y_true <= 0.5, tf.float32)

        # Apply weights
        weighted_loss = (
            pos_mask * self.pos_weight + neg_mask * self.neg_weight
        ) * mse_loss

        return tf.reduce_mean(weighted_loss)

    def get_config(self):
        """Get configuration for serialization"""
        config = super(WeightedMSELoss, self).get_config()
        config.update({"pos_weight": self.pos_weight, "neg_weight": self.neg_weight})
        return config


class CombinedCRAFTLoss(tf.keras.losses.Loss):
    """Combined loss for CRAFT model (character + affinity)"""

    def __init__(
        self,
        char_weight=1.0,
        affinity_weight=1.0,
        ohem_ratio=3.0,
        reduction=tf.keras.losses.Reduction.AUTO,
        name="combined_craft_loss",
    ):
        """
        Initialize Combined CRAFT Loss

        Args:
            char_weight: Weight for character region loss
            affinity_weight: Weight for affinity loss
            ohem_ratio: OHEM ratio for hard example mining
            reduction: Type of reduction to apply to loss
            name: Name of the loss function
        """
        super(CombinedCRAFTLoss, self).__init__(reduction=reduction, name=name)
        self.char_weight = char_weight
        self.affinity_weight = affinity_weight
        self.ohem_ratio = ohem_ratio

        # Create separate OHEM losses for character and affinity
        self.char_loss_fn = MSE_OHEM_Loss(ratio=ohem_ratio, name="char_ohem_loss")
        self.affinity_loss_fn = MSE_OHEM_Loss(
            ratio=ohem_ratio, name="affinity_ohem_loss"
        )

    def call(self, y_true, y_pred):
        """
        Compute combined CRAFT loss

        Args:
            y_true: Ground truth labels [batch_size, height, width, 2]
            y_pred: Predicted labels [batch_size, height, width, 2]

        Returns:
            Combined loss value
        """
        # Split into character and affinity channels
        char_true = y_true[..., 0:1]
        char_pred = y_pred[..., 0:1]

        affinity_true = y_true[..., 1:2]
        affinity_pred = y_pred[..., 1:2]

        # Compute individual losses
        char_loss = self.char_loss_fn(char_true, char_pred)
        affinity_loss = self.affinity_loss_fn(affinity_true, affinity_pred)

        # Combine losses with weights
        total_loss = self.char_weight * char_loss + self.affinity_weight * affinity_loss

        return total_loss

    def get_config(self):
        """Get configuration for serialization"""
        config = super(CombinedCRAFTLoss, self).get_config()
        config.update(
            {
                "char_weight": self.char_weight,
                "affinity_weight": self.affinity_weight,
                "ohem_ratio": self.ohem_ratio,
            }
        )
        return config


# Legacy function for backward compatibility
def MSE_OHEM_Loss_legacy(y_pred, y_true, ratio=3.0):
    """Legacy OHEM loss function for backward compatibility"""
    loss_fn = MSE_OHEM_Loss(ratio=ratio)
    return loss_fn(y_true, y_pred)


# Factory functions
def create_ohem_loss(ratio=3.0):
    """Factory function to create OHEM loss"""
    return MSE_OHEM_Loss(ratio=ratio)


def create_weighted_mse_loss(pos_weight=1.0, neg_weight=1.0):
    """Factory function to create weighted MSE loss"""
    return WeightedMSELoss(pos_weight=pos_weight, neg_weight=neg_weight)


def create_craft_loss(char_weight=1.0, affinity_weight=1.0, ohem_ratio=3.0):
    """Factory function to create combined CRAFT loss"""
    return CombinedCRAFTLoss(
        char_weight=char_weight, affinity_weight=affinity_weight, ohem_ratio=ohem_ratio
    )
