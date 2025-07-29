import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


def get_result_img(
    src_img: np.ndarray,
    text_score: np.ndarray,
    link_score: np.ndarray,
    text_threshold: float = 0.7,
    link_threshold: float = 0.4,
    low_text: float = 0.4,
) -> np.ndarray:
    """
    Generate result image with detected text regions

    Args:
        src_img: Source image
        text_score: Text region score map
        link_score: Link/affinity score map
        text_threshold: Threshold for text regions
        link_threshold: Threshold for link regions
        low_text: Low threshold for text regions

    Returns:
        Result image with text regions highlighted
    """
    # Ensure inputs are numpy arrays
    if isinstance(text_score, tf.Tensor):
        text_score = text_score.numpy()
    if isinstance(link_score, tf.Tensor):
        link_score = link_score.numpy()

    # Resize score maps to match source image size
    img_h, img_w = src_img.shape[:2]
    text_score = cv2.resize(text_score, (img_w, img_h))
    link_score = cv2.resize(link_score, (img_w, img_h))

    # Create binary masks
    text_score_comb = np.clip(text_score + link_score, 0, 1)

    # Apply thresholds
    ret, text_mask = cv2.threshold(text_score_comb, low_text, 1, cv2.THRESH_BINARY)
    text_mask = text_mask.astype(np.uint8)

    # Find connected components
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text_mask, connectivity=4, ltype=cv2.CV_32S
    )

    # Create result image
    result_img = src_img.copy()

    # Draw bounding boxes for each component
    for k in range(1, nLabels):
        # Size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # Get bounding box
        x = stats[k, cv2.CC_STAT_LEFT]
        y = stats[k, cv2.CC_STAT_TOP]
        w = stats[k, cv2.CC_STAT_WIDTH]
        h = stats[k, cv2.CC_STAT_HEIGHT]

        # Draw rectangle
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return result_img


def get_text_boxes(
    text_score: np.ndarray,
    link_score: np.ndarray,
    text_threshold: float = 0.7,
    link_threshold: float = 0.4,
    low_text: float = 0.4,
    poly: bool = False,
) -> List[np.ndarray]:
    """
    Extract text boxes from score maps

    Args:
        text_score: Text region score map
        link_score: Link/affinity score map
        text_threshold: Threshold for text regions
        link_threshold: Threshold for link regions
        low_text: Low threshold for text regions
        poly: Whether to return polygons or rectangles

    Returns:
        List of text boxes (either rectangles or polygons)
    """
    # Ensure inputs are numpy arrays
    if isinstance(text_score, tf.Tensor):
        text_score = text_score.numpy()
    if isinstance(link_score, tf.Tensor):
        link_score = link_score.numpy()

    # Combine text and link scores
    text_score_comb = np.clip(text_score + link_score, 0, 1)

    # Apply threshold
    ret, text_mask = cv2.threshold(text_score_comb, low_text, 1, cv2.THRESH_BINARY)
    text_mask = text_mask.astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(text_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        # Filter small contours
        if cv2.contourArea(contour) < 10:
            continue

        if poly:
            # Return polygon
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            boxes.append(approx.reshape(-1, 2))
        else:
            # Return bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]))

    return boxes


def adjust_result_coordinates(
    boxes: List[np.ndarray], ratio_w: float, ratio_h: float
) -> List[np.ndarray]:
    """
    Adjust box coordinates based on image resize ratios

    Args:
        boxes: List of text boxes
        ratio_w: Width ratio (original_width / resized_width)
        ratio_h: Height ratio (original_height / resized_height)

    Returns:
        Adjusted boxes
    """
    adjusted_boxes = []
    for box in boxes:
        adjusted_box = box.copy()
        adjusted_box[:, 0] *= ratio_w
        adjusted_box[:, 1] *= ratio_h
        adjusted_boxes.append(adjusted_box)

    return adjusted_boxes


def visualize_score_maps(
    text_score: np.ndarray, link_score: np.ndarray, save_path: Optional[str] = None
) -> None:
    """
    Visualize text and link score maps

    Args:
        text_score: Text region score map
        link_score: Link/affinity score map
        save_path: Path to save visualization (optional)
    """
    # Ensure inputs are numpy arrays
    if isinstance(text_score, tf.Tensor):
        text_score = text_score.numpy()
    if isinstance(link_score, tf.Tensor):
        link_score = link_score.numpy()

    # Create subplot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot text score
    axes[0].imshow(text_score, cmap="jet")
    axes[0].set_title("Text Score Map")
    axes[0].axis("off")

    # Plot link score
    axes[1].imshow(link_score, cmap="jet")
    axes[1].set_title("Link Score Map")
    axes[1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def draw_text_boxes(
    image: np.ndarray,
    boxes: List[np.ndarray],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw text boxes on image

    Args:
        image: Input image
        boxes: List of text boxes
        color: Box color (BGR format)
        thickness: Line thickness

    Returns:
        Image with drawn boxes
    """
    result_img = image.copy()

    for box in boxes:
        # Convert to integer coordinates
        box = box.astype(np.int32)

        # Draw polygon
        cv2.polylines(result_img, [box], True, color, thickness)

    return result_img


def create_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Create heatmap overlay on image

    Args:
        image: Base image
        heatmap: Heatmap to overlay
        alpha: Transparency factor
        colormap: OpenCV colormap

    Returns:
        Image with heatmap overlay
    """
    # Ensure inputs are numpy arrays
    if isinstance(heatmap, tf.Tensor):
        heatmap = heatmap.numpy()

    # Resize heatmap to match image size
    img_h, img_w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (img_w, img_h))

    # Normalize heatmap to 0-255 range
    heatmap_norm = (
        (heatmap_resized - heatmap_resized.min())
        / (heatmap_resized.max() - heatmap_resized.min())
        * 255
    ).astype(np.uint8)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_norm, colormap)

    # Blend with original image
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay


def post_process_predictions(
    text_score: np.ndarray,
    link_score: np.ndarray,
    image_shape: Tuple[int, int],
    text_threshold: float = 0.7,
    link_threshold: float = 0.4,
    low_text: float = 0.4,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Post-process model predictions to extract text boxes

    Args:
        text_score: Text region score map
        link_score: Link/affinity score map
        image_shape: Original image shape (height, width)
        text_threshold: Threshold for text regions
        link_threshold: Threshold for link regions
        low_text: Low threshold for text regions

    Returns:
        Tuple of (text_boxes, combined_score_map)
    """
    # Get text boxes
    boxes = get_text_boxes(
        text_score, link_score, text_threshold, link_threshold, low_text
    )

    # Calculate resize ratios
    score_h, score_w = text_score.shape[:2]
    ratio_h = image_shape[0] / score_h
    ratio_w = image_shape[1] / score_w

    # Adjust coordinates
    adjusted_boxes = adjust_result_coordinates(boxes, ratio_w, ratio_h)

    # Create combined score map
    combined_score = np.clip(text_score + link_score, 0, 1)

    return adjusted_boxes, combined_score


def save_detection_results(
    image: np.ndarray,
    boxes: List[np.ndarray],
    text_score: np.ndarray,
    link_score: np.ndarray,
    output_dir: str,
    filename: str,
) -> None:
    """
    Save detection results including image with boxes and score maps

    Args:
        image: Original image
        boxes: Detected text boxes
        text_score: Text score map
        link_score: Link score map
        output_dir: Output directory
        filename: Base filename
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Save image with boxes
    result_img = draw_text_boxes(image, boxes)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_result.jpg"), result_img)

    # Save score maps
    text_heatmap = create_heatmap_overlay(image, text_score)
    link_heatmap = create_heatmap_overlay(image, link_score)

    cv2.imwrite(os.path.join(output_dir, f"{filename}_text_score.jpg"), text_heatmap)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_link_score.jpg"), link_heatmap)

    # Save raw score maps
    plt.imsave(
        os.path.join(output_dir, f"{filename}_text_raw.jpg"), text_score, cmap="jet"
    )
    plt.imsave(
        os.path.join(output_dir, f"{filename}_link_raw.jpg"), link_score, cmap="jet"
    )


# Legacy functions for backward compatibility
def get_res_hmp(predictions):
    """Legacy function for backward compatibility"""
    if len(predictions.shape) == 4:
        text_score = predictions[0, :, :, 0]
        link_score = predictions[0, :, :, 1]
    else:
        text_score = predictions[:, :, 0]
        link_score = predictions[:, :, 1]

    return text_score, link_score
