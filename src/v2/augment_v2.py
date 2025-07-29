import tensorflow as tf
import numpy as np
import cv2
import random
from typing import Tuple, Optional
import albumentations as A


class CRAFTAugmentation:
    """Data augmentation pipeline for CRAFT text detection"""

    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        label_size: Tuple[int, int] = (256, 256),
        augment_prob: float = 0.5,
    ):
        """
        Initialize augmentation pipeline

        Args:
            image_size: Target image size
            label_size: Target label size
            augment_prob: Probability of applying augmentations
        """
        self.image_size = image_size
        self.label_size = label_size
        self.augment_prob = augment_prob

        # Define augmentation pipeline using Albumentations
        self.transform = A.Compose(
            [
                A.OneOf(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.1),
                        A.RandomRotate90(p=0.2),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2, contrast_limit=0.2, p=0.5
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=10,
                            sat_shift_limit=20,
                            val_shift_limit=20,
                            p=0.3,
                        ),
                        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                    ],
                    p=0.4,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                        A.MotionBlur(blur_limit=7, p=0.2),
                    ],
                    p=0.2,
                ),
                A.OneOf(
                    [
                        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
                        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
                        A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.2),
                    ],
                    p=0.1,
                ),
                A.Resize(height=self.image_size[1], width=self.image_size[0], p=1.0),
            ],
            additional_targets={"mask": "mask"},
        )

        # Simple transform for labels (resize only)
        self.label_transform = A.Compose(
            [
                A.Resize(height=self.label_size[1], width=self.label_size[0], p=1.0),
            ]
        )

    def __call__(
        self, image: np.ndarray, label: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply augmentations to image and label

        Args:
            image: Input image
            label: Input label (optional)

        Returns:
            Augmented image and label
        """
        if random.random() > self.augment_prob:
            # No augmentation, just resize
            image_resized = cv2.resize(image, self.image_size)
            if label is not None:
                label_resized = cv2.resize(label, self.label_size)
                return image_resized, label_resized
            return image_resized, None

        # Apply augmentations
        if label is not None:
            # Apply same geometric transforms to both image and label
            transformed = self.transform(image=image, mask=label)
            aug_image = transformed["image"]
            aug_label = transformed["mask"]

            # Ensure label has correct size
            if aug_label.shape[:2] != self.label_size:
                aug_label = cv2.resize(aug_label, self.label_size)

            return aug_image, aug_label
        else:
            # Apply transforms to image only
            transformed = self.transform(image=image)
            return transformed["image"], None


class TensorFlowAugmentation:
    """TensorFlow-based augmentation pipeline"""

    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        label_size: Tuple[int, int] = (256, 256),
    ):
        """
        Initialize TensorFlow augmentation pipeline

        Args:
            image_size: Target image size
            label_size: Target label size
        """
        self.image_size = image_size
        self.label_size = label_size

    @tf.function
    def random_flip(
        self, image: tf.Tensor, label: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply random horizontal flip"""
        if tf.random.uniform([]) > 0.5:
            image = tf.image.flip_left_right(image)
            label = tf.image.flip_left_right(label)
        return image, label

    @tf.function
    def random_brightness_contrast(self, image: tf.Tensor) -> tf.Tensor:
        """Apply random brightness and contrast"""
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        return tf.clip_by_value(image, 0.0, 1.0)

    @tf.function
    def random_hue_saturation(self, image: tf.Tensor) -> tf.Tensor:
        """Apply random hue and saturation"""
        image = tf.image.random_hue(image, max_delta=0.1)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        return tf.clip_by_value(image, 0.0, 1.0)

    @tf.function
    def resize_images(
        self, image: tf.Tensor, label: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Resize image and label to target sizes"""
        image = tf.image.resize(image, self.image_size)
        label = tf.image.resize(label, self.label_size)
        return image, label

    def __call__(
        self, image: tf.Tensor, label: tf.Tensor, training: bool = True
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply TensorFlow augmentations

        Args:
            image: Input image tensor
            label: Input label tensor
            training: Whether in training mode

        Returns:
            Augmented image and label tensors
        """
        # Resize first
        image, label = self.resize_images(image, label)

        if training:
            # Apply augmentations during training
            image, label = self.random_flip(image, label)
            image = self.random_brightness_contrast(image)
            image = self.random_hue_saturation(image)

        return image, label


def geometric_transform(
    image: np.ndarray,
    label: np.ndarray,
    angle_range: Tuple[float, float] = (-10, 10),
    scale_range: Tuple[float, float] = (0.9, 1.1),
    translate_range: Tuple[float, float] = (-0.1, 0.1),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply geometric transformations to image and label

    Args:
        image: Input image
        label: Input label
        angle_range: Range of rotation angles
        scale_range: Range of scaling factors
        translate_range: Range of translation factors

    Returns:
        Transformed image and label
    """
    h, w = image.shape[:2]

    # Random parameters
    angle = random.uniform(*angle_range)
    scale = random.uniform(*scale_range)
    tx = random.uniform(*translate_range) * w
    ty = random.uniform(*translate_range) * h

    # Create transformation matrix
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    # Apply transformation
    transformed_image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    transformed_label = cv2.warpAffine(
        label, M, (label.shape[1], label.shape[0]), borderMode=cv2.BORDER_REFLECT
    )

    return transformed_image, transformed_label


def color_jitter(
    image: np.ndarray,
    brightness_range: Tuple[float, float] = (0.8, 1.2),
    contrast_range: Tuple[float, float] = (0.8, 1.2),
    saturation_range: Tuple[float, float] = (0.8, 1.2),
    hue_range: Tuple[float, float] = (-0.1, 0.1),
) -> np.ndarray:
    """
    Apply color jittering to image

    Args:
        image: Input image
        brightness_range: Range of brightness factors
        contrast_range: Range of contrast factors
        saturation_range: Range of saturation factors
        hue_range: Range of hue shifts

    Returns:
        Color-jittered image
    """
    # Convert to float
    image = image.astype(np.float32) / 255.0

    # Brightness
    brightness_factor = random.uniform(*brightness_range)
    image = image * brightness_factor

    # Contrast
    contrast_factor = random.uniform(*contrast_range)
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    image = (image - mean) * contrast_factor + mean

    # Convert to HSV for saturation and hue adjustments
    image_hsv = cv2.cvtColor(np.clip(image, 0, 1), cv2.COLOR_RGB2HSV)

    # Saturation
    saturation_factor = random.uniform(*saturation_range)
    image_hsv[:, :, 1] *= saturation_factor

    # Hue
    hue_shift = random.uniform(*hue_range)
    image_hsv[:, :, 0] += hue_shift
    image_hsv[:, :, 0] = np.clip(image_hsv[:, :, 0], 0, 1)

    # Convert back to RGB
    image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)

    # Clip and convert back to uint8
    image = np.clip(image * 255, 0, 255).astype(np.uint8)

    return image


def add_noise(
    image: np.ndarray, noise_type: str = "gaussian", noise_params: dict = None
) -> np.ndarray:
    """
    Add noise to image

    Args:
        image: Input image
        noise_type: Type of noise ('gaussian', 'salt_pepper', 'speckle')
        noise_params: Parameters for noise generation

    Returns:
        Noisy image
    """
    if noise_params is None:
        noise_params = {}

    image = image.astype(np.float32)

    if noise_type == "gaussian":
        mean = noise_params.get("mean", 0)
        std = noise_params.get("std", 25)
        noise = np.random.normal(mean, std, image.shape)
        noisy_image = image + noise

    elif noise_type == "salt_pepper":
        prob = noise_params.get("prob", 0.05)
        noisy_image = image.copy()

        # Salt noise
        salt_mask = np.random.random(image.shape[:2]) < prob / 2
        noisy_image[salt_mask] = 255

        # Pepper noise
        pepper_mask = np.random.random(image.shape[:2]) < prob / 2
        noisy_image[pepper_mask] = 0

    elif noise_type == "speckle":
        noise = np.random.randn(*image.shape)
        noisy_image = image + image * noise * 0.1

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def create_augmentation_pipeline(
    image_size: Tuple[int, int] = (512, 512),
    label_size: Tuple[int, int] = (256, 256),
    use_albumentations: bool = True,
) -> callable:
    """
    Create augmentation pipeline

    Args:
        image_size: Target image size
        label_size: Target label size
        use_albumentations: Whether to use Albumentations library

    Returns:
        Augmentation function
    """
    if use_albumentations:
        return CRAFTAugmentation(image_size, label_size)
    else:
        return TensorFlowAugmentation(image_size, label_size)


# Legacy functions for backward compatibility
def augment_data(image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy augmentation function"""
    augmenter = CRAFTAugmentation()
    return augmenter(image, label)


def random_crop(
    image: np.ndarray, label: np.ndarray, crop_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Random crop for image and label"""
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size

    if h <= crop_h or w <= crop_w:
        return image, label

    top = random.randint(0, h - crop_h)
    left = random.randint(0, w - crop_w)

    cropped_image = image[top : top + crop_h, left : left + crop_w]
    cropped_label = label[top : top + crop_h, left : left + crop_w]

    return cropped_image, cropped_label
