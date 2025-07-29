import numpy as np
import cv2
import random
from typing import Tuple, Optional

# This file appears to be a variant of augment.py, so we'll provide a simple augmentation pipeline


def random_flip(
    image: np.ndarray, label: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Random horizontal flip for image and label"""
    if random.random() > 0.5:
        image = np.fliplr(image)
        if label is not None:
            label = np.fliplr(label)
    return image, label


def random_rotate(
    image: np.ndarray,
    label: Optional[np.ndarray] = None,
    angle_range: Tuple[int, int] = (-10, 10),
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Random rotation for image and label"""
    angle = random.uniform(*angle_range)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if label is not None:
        label = cv2.warpAffine(label, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return image, label


def random_brightness(
    image: np.ndarray, brightness_range: Tuple[float, float] = (0.8, 1.2)
) -> np.ndarray:
    """Random brightness adjustment"""
    factor = random.uniform(*brightness_range)
    image = np.clip(image * factor, 0, 255).astype(np.uint8)
    return image


def augment_n(
    image: np.ndarray, label: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Simple augmentation pipeline for CRAFT (variant N)"""
    image, label = random_flip(image, label)
    image, label = random_rotate(image, label)
    image = random_brightness(image)
    return image, label


# Legacy function for backward compatibility
def augment_n_legacy(
    image: np.ndarray, label: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    return augment_n(image, label)
