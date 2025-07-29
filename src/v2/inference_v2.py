#!/usr/bin/env python3
"""
CRAFT Text Detection Inference Script - TensorFlow v2
Usage: python inference_v2.py --checkpoint_dir /path/to/checkpoint --input_image /path/to/image.jpg
"""

import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

# Import our converted TF2 modules
from net_v2 import CRAFTNet
from text_utils_v2 import get_text_boxes, draw_text_boxes, save_detection_results
from datagen_v2 import normalizeMeanVariance


class CRAFTInference:
    """CRAFT Text Detection Inference Class"""

    def __init__(self, checkpoint_dir: str, input_size: tuple = (512, 512)):
        """
        Initialize CRAFT inference

        Args:
            checkpoint_dir: Directory containing checkpoint files
            input_size: Input image size (width, height)
        """
        self.checkpoint_dir = checkpoint_dir
        self.input_size = input_size
        self.model = None
        self.checkpoint = None

        # Initialize model
        self._load_model()

    def _load_model(self):
        """Load CRAFT model and checkpoint"""
        print("Initializing CRAFT model...")

        # Create model
        self.model = CRAFTNet()

        # Build model by running a dummy forward pass
        dummy_input = tf.zeros((1, self.input_size[1], self.input_size[0], 3))
        _ = self.model(dummy_input, training=False)

        # Setup checkpoint
        self.checkpoint = tf.train.Checkpoint(model=self.model)

        # Find latest checkpoint
        checkpoint_path = self._find_checkpoint()

        if checkpoint_path:
            print(f"Loading checkpoint from: {checkpoint_path}")
            self.checkpoint.restore(checkpoint_path)
            print("Checkpoint loaded successfully!")
        else:
            print("Warning: No checkpoint found. Using randomly initialized weights.")

    def _find_checkpoint(self):
        """Find the latest checkpoint in the directory"""
        checkpoint_dir = Path(self.checkpoint_dir)

        # Look for TF2 checkpoint files
        checkpoint_files = list(checkpoint_dir.glob("*.index"))

        if checkpoint_files:
            # Get the latest checkpoint
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            # Remove .index extension to get checkpoint prefix
            return str(latest_checkpoint).replace(".index", "")

        # Look for legacy TF1 checkpoint files (.ckpt)
        legacy_files = list(checkpoint_dir.glob("*.ckpt.index"))
        if legacy_files:
            latest_checkpoint = max(legacy_files, key=os.path.getctime)
            return str(latest_checkpoint).replace(".index", "")

        # Try checkpoint manager format
        manager_checkpoint = tf.train.latest_checkpoint(str(checkpoint_dir))
        if manager_checkpoint:
            return manager_checkpoint

        return None

    def preprocess_image(self, image_path: str):
        """
        Preprocess input image for inference

        Args:
            image_path: Path to input image

        Returns:
            Preprocessed image tensor and original image
        """
        # Load image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Convert BGR to RGB
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Resize image
        resized_image = cv2.resize(original_image, self.input_size)

        # Normalize image
        normalized_image = normalizeMeanVariance(resized_image)

        # Add batch dimension
        input_tensor = tf.expand_dims(normalized_image, axis=0)

        return input_tensor, original_image

    def run_inference(
        self,
        image_path: str,
        text_threshold: float = 0.7,
        link_threshold: float = 0.4,
        low_text: float = 0.4,
    ):
        """
        Run inference on input image

        Args:
            image_path: Path to input image
            text_threshold: Threshold for text regions
            link_threshold: Threshold for link regions
            low_text: Low threshold for text regions

        Returns:
            Dictionary containing detection results
        """
        print(f"Running inference on: {image_path}")

        # Preprocess image
        input_tensor, original_image = self.preprocess_image(image_path)

        # Run inference
        predictions = self.model(input_tensor, training=False)

        # Extract predictions
        predictions_np = predictions.numpy()
        text_score = predictions_np[0, :, :, 0]  # Character region score
        link_score = predictions_np[0, :, :, 1]  # Affinity score

        # Get text boxes
        text_boxes = get_text_boxes(
            text_score, link_score, text_threshold, link_threshold, low_text
        )

        # Adjust coordinates to original image size
        original_h, original_w = original_image.shape[:2]
        score_h, score_w = text_score.shape

        ratio_w = original_w / score_w
        ratio_h = original_h / score_h

        adjusted_boxes = []
        for box in text_boxes:
            adjusted_box = box.copy()
            adjusted_box[:, 0] *= ratio_w
            adjusted_box[:, 1] *= ratio_h
            adjusted_boxes.append(adjusted_box.astype(np.int32))

        # Create result image
        result_image = draw_text_boxes(original_image, adjusted_boxes)

        return {
            "original_image": original_image,
            "result_image": result_image,
            "text_boxes": adjusted_boxes,
            "text_score": text_score,
            "link_score": link_score,
            "num_detections": len(adjusted_boxes),
        }

    def save_results(self, results: dict, output_path: str):
        """
        Save inference results

        Args:
            results: Results dictionary from run_inference
            output_path: Path to save results
        """
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save result image
        result_bgr = cv2.cvtColor(results["result_image"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), result_bgr)

        # Save detailed results
        base_name = Path(output_path).stem
        output_dir_str = str(output_dir)

        save_detection_results(
            results["original_image"],
            results["text_boxes"],
            results["text_score"],
            results["link_score"],
            output_dir_str,
            base_name,
        )

        print(f"Results saved to: {output_path}")
        print(f"Detected {results['num_detections']} text regions")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="CRAFT Text Detection Inference")
    parser.add_argument(
        "--checkpoint_dir", required=True, help="Directory containing checkpoint files"
    )
    parser.add_argument("--input_image", required=True, help="Path to input image")
    parser.add_argument(
        "--output_path", default="result.jpg", help="Path to save result image"
    )
    parser.add_argument(
        "--text_threshold", type=float, default=0.7, help="Text region threshold"
    )
    parser.add_argument(
        "--link_threshold", type=float, default=0.4, help="Link region threshold"
    )
    parser.add_argument(
        "--low_text", type=float, default=0.4, help="Low text threshold"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Input image size (width height)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.checkpoint_dir):
        raise ValueError(f"Checkpoint directory not found: {args.checkpoint_dir}")

    if not os.path.exists(args.input_image):
        raise ValueError(f"Input image not found: {args.input_image}")

    # Initialize inference
    craft_inference = CRAFTInference(
        checkpoint_dir=args.checkpoint_dir, input_size=tuple(args.input_size)
    )

    # Run inference
    results = craft_inference.run_inference(
        image_path=args.input_image,
        text_threshold=args.text_threshold,
        link_threshold=args.link_threshold,
        low_text=args.low_text,
    )

    # Save results
    craft_inference.save_results(results, args.output_path)


if __name__ == "__main__":
    main()

# Example usage:
# python inference_v2.py --checkpoint_dir ./model --input_image ./test_image.jpg --output_path ./result.jpg
