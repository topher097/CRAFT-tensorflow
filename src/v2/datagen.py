import tensorflow as tf
import numpy as np
import cv2
import os
import random
from typing import Tuple, List
import matplotlib.image as Image


def normalizeMeanVariance(
    image, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)
):
    """
    Normalize image with ImageNet mean and variance

    Args:
        image: Input image array
        mean: Mean values for normalization
        variance: Variance values for normalization

    Returns:
        Normalized image
    """
    image = image.astype(np.float32) / 255.0

    # Convert to tensor for easier processing
    if isinstance(image, np.ndarray):
        image = tf.convert_to_tensor(image, dtype=tf.float32)

    # Normalize
    mean = tf.constant(mean, dtype=tf.float32)
    std = tf.constant(variance, dtype=tf.float32)

    normalized = (image - mean) / std

    return normalized.numpy() if isinstance(normalized, tf.Tensor) else normalized


def denormalizeMeanVariance(
    image, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)
):
    """
    Denormalize image back to original range

    Args:
        image: Normalized image array
        mean: Mean values used for normalization
        variance: Variance values used for normalization

    Returns:
        Denormalized image
    """
    if isinstance(image, np.ndarray):
        image = tf.convert_to_tensor(image, dtype=tf.float32)

    mean = tf.constant(mean, dtype=tf.float32)
    std = tf.constant(variance, dtype=tf.float32)

    denormalized = (image * std) + mean
    denormalized = tf.clip_by_value(denormalized, 0.0, 1.0)

    return (denormalized.numpy() * 255.0).astype(np.uint8)


class DataGenerator:
    """TensorFlow v2 compatible data generator for CRAFT training"""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 2,
        image_size: Tuple[int, int] = (512, 512),
        label_size: Tuple[int, int] = (256, 256),
        shuffle: bool = True,
        augment: bool = True,
    ):
        """
        Initialize data generator

        Args:
            data_dir: Directory containing training data
            batch_size: Batch size for training
            image_size: Size to resize input images
            label_size: Size of output labels
            shuffle: Whether to shuffle data
            augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.label_size = label_size
        self.shuffle = shuffle
        self.augment = augment

        # Load file paths
        self.image_paths = self._load_image_paths()
        self.label_paths = self._load_label_paths()

        # Ensure same number of images and labels
        assert len(self.image_paths) == len(self.label_paths), (
            "Number of images and labels must match"
        )

        self.num_samples = len(self.image_paths)
        self.indices = list(range(self.num_samples))

        if self.shuffle:
            random.shuffle(self.indices)

    def _load_image_paths(self) -> List[str]:
        """Load image file paths"""
        image_dir = os.path.join(self.data_dir, "images")
        if not os.path.exists(image_dir):
            # Fallback to data_dir if images subdirectory doesn't exist
            image_dir = self.data_dir

        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_paths.extend(tf.io.gfile.glob(os.path.join(image_dir, ext)))

        return sorted(image_paths)

    def _load_label_paths(self) -> List[str]:
        """Load label file paths"""
        label_dir = os.path.join(self.data_dir, "labels")
        if not os.path.exists(label_dir):
            # Fallback to data_dir if labels subdirectory doesn't exist
            label_dir = self.data_dir

        label_paths = []
        for ext in ["*.npy", "*.npz"]:
            label_paths.extend(tf.io.gfile.glob(os.path.join(label_dir, ext)))

        return sorted(label_paths)

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            # Fallback to matplotlib
            image = Image.imread(image_path)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Resize image
        image = cv2.resize(image, self.image_size)

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def _load_label(self, label_path: str) -> np.ndarray:
        """Load and preprocess label"""
        if label_path.endswith(".npy"):
            label = np.load(label_path)
        elif label_path.endswith(".npz"):
            label_data = np.load(label_path)
            # Assume the label is stored under 'arr_0' or similar key
            label = label_data[list(label_data.keys())[0]]
        else:
            raise ValueError(f"Unsupported label format: {label_path}")

        # Resize label to target size
        if len(label.shape) == 3:
            label = cv2.resize(label, self.label_size)
        elif len(label.shape) == 2:
            label = cv2.resize(label, self.label_size)
            label = np.expand_dims(label, axis=-1)

        # Ensure label has 2 channels (character + affinity)
        if label.shape[-1] == 1:
            # Duplicate channel if only one channel
            label = np.concatenate([label, label], axis=-1)
        elif label.shape[-1] > 2:
            # Take first 2 channels if more than 2
            label = label[..., :2]

        return label.astype(np.float32)

    def _augment_data(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation"""
        if not self.augment:
            return image, label

        # Random horizontal flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)

        # Random rotation (small angles)
        if random.random() > 0.7:
            angle = random.uniform(-10, 10)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)

            # Rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Apply rotation
            image = cv2.warpAffine(image, M, (w, h))
            label = cv2.warpAffine(label, M, (self.label_size[0], self.label_size[1]))

        # Random brightness and contrast
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)  # Contrast
            beta = random.uniform(-20, 20)  # Brightness
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        return image, label

    def __len__(self) -> int:
        """Get number of batches per epoch"""
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """Make generator iterable"""
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get next batch"""
        if not hasattr(self, "_current_idx"):
            self._current_idx = 0

        if self._current_idx >= self.num_samples:
            # Reset for next epoch
            self._current_idx = 0
            if self.shuffle:
                random.shuffle(self.indices)
            raise StopIteration

        # Get batch indices
        batch_indices = self.indices[
            self._current_idx : self._current_idx + self.batch_size
        ]
        self._current_idx += self.batch_size

        # Load batch data
        batch_images = []
        batch_labels = []

        for idx in batch_indices:
            # Load image and label
            image = self._load_image(self.image_paths[idx])
            label = self._load_label(self.label_paths[idx])

            # Apply augmentation
            image, label = self._augment_data(image, label)

            # Normalize image
            image = normalizeMeanVariance(image)

            batch_images.append(image)
            batch_labels.append(label)

        return np.array(batch_images), np.array(batch_labels)


def create_tf_dataset(
    data_dir: str,
    batch_size: int = 2,
    image_size: Tuple[int, int] = (512, 512),
    label_size: Tuple[int, int] = (256, 256),
    shuffle: bool = True,
    augment: bool = True,
    prefetch_buffer: int = tf.data.AUTOTUNE,
) -> tf.data.Dataset:
    """
    Create TensorFlow Dataset for training

    Args:
        data_dir: Directory containing training data
        batch_size: Batch size for training
        image_size: Size to resize input images
        label_size: Size of output labels
        shuffle: Whether to shuffle data
        augment: Whether to apply data augmentation
        prefetch_buffer: Prefetch buffer size

    Returns:
        tf.data.Dataset object
    """
    generator = DataGenerator(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        label_size=label_size,
        shuffle=shuffle,
        augment=augment,
    )

    # Create dataset from generator
    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=(
            tf.TensorSpec(shape=(None, *image_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, *label_size, 2), dtype=tf.float32),
        ),
    )

    if prefetch_buffer:
        dataset = dataset.prefetch(prefetch_buffer)

    return dataset


# Legacy functions for backward compatibility
def generator(data_dir="/path/to/syntext/dataset", shuffle=True, batch_size=2):
    """Legacy generator function for backward compatibility"""
    gen = DataGenerator(data_dir=data_dir, batch_size=batch_size, shuffle=shuffle)

    while True:
        try:
            yield next(gen)
        except StopIteration:
            # Reset generator
            gen = DataGenerator(
                data_dir=data_dir, batch_size=batch_size, shuffle=shuffle
            )


def procces_function(image, label):
    """Legacy process function for backward compatibility"""
    # Apply normalization
    processed_image = normalizeMeanVariance(image)
    return processed_image, label


# Utility functions
def resize_image_and_label(
    image: np.ndarray,
    label: np.ndarray,
    image_size: Tuple[int, int] = (512, 512),
    label_size: Tuple[int, int] = (256, 256),
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize image and label to target sizes"""
    resized_image = cv2.resize(image, image_size)
    resized_label = cv2.resize(label, label_size)

    return resized_image, resized_label


def load_syntext_data(data_dir: str) -> Tuple[List[str], List[str]]:
    """Load SynthText dataset file paths"""
    image_paths = []
    label_paths = []

    # Add your SynthText loading logic here
    # This is a placeholder implementation

    return image_paths, label_paths
