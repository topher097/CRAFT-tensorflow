#!/usr/bin/env python3
"""
CRAFT Text Detection Inference Script - Using TensorFlow SavedModel (Improved)
Usage: python savedmodel_inference_fixed.py --model_path /path/to/saved_model --input_image /path/to/image.jpg
"""

import argparse
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class CRAFTSavedModelInference:
    """CRAFT Text Detection using SavedModel format"""

    def __init__(self, model_path: str, input_size: tuple = (512, 512)):
        """
        Initialize CRAFT inference with SavedModel

        Args:
            model_path: Path to SavedModel directory
            input_size: Input image size (width, height)
        """
        self.model_path = model_path
        self.input_size = input_size
        self.model = None
        self.infer = None
        self.input_key = None

        # Load model
        self._load_model()

    def _load_model(self):
        """Load SavedModel and inspect its signature"""
        print(f"Loading SavedModel from: {self.model_path}")

        try:
            self.model = tf.saved_model.load(self.model_path)
            print("SavedModel loaded successfully!")

            # Print available signatures
            if hasattr(self.model, "signatures"):
                print("Available signatures:")
                for sig_name in self.model.signatures.keys():
                    print(f"  - {sig_name}")

                # Get the serving signature
                if "serving_default" in self.model.signatures:
                    self.infer = self.model.signatures["serving_default"]
                else:
                    sig_name = list(self.model.signatures.keys())[0]
                    self.infer = self.model.signatures[sig_name]
                    print(f"Using signature: {sig_name}")

                # Inspect the signature to get input/output info
                self._inspect_signature()
            else:
                # If no signatures, try to use the model directly
                self.infer = self.model
                print("No signatures found, using model directly")

        except Exception as e:
            raise ValueError(f"Failed to load SavedModel: {e}")

    def _inspect_signature(self):
        """Inspect the model signature to understand input/output format"""
        if hasattr(self.infer, "structured_input_signature"):
            input_signature = self.infer.structured_input_signature[1]
            output_signature = self.infer.structured_outputs

            print("\nModel signature details:")
            print("Inputs:")
            for key, spec in input_signature.items():
                print(f"  {key}: {spec}")
                self.input_key = key  # Store the input key

            print("Outputs:")
            if isinstance(output_signature, dict):
                for key, spec in output_signature.items():
                    print(f"  {key}: {spec}")
            else:
                print(f"  Output: {output_signature}")

            # Check expected input shape
            if self.input_key:
                expected_shape = input_signature[self.input_key].shape
                print(f"\nExpected input shape: {expected_shape}")

                # Update input size if model expects different dimensions
                if len(expected_shape) >= 3:
                    model_h, model_w = expected_shape[1], expected_shape[2]
                    if model_h is not None and model_w is not None:
                        if (model_h, model_w) != (
                            self.input_size[1],
                            self.input_size[0],
                        ):
                            print(
                                f"Updating input size from {self.input_size} to ({model_w}, {model_h})"
                            )
                            self.input_size = (model_w, model_h)

    def normalizeMeanVariance(
        self, image, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)
    ):
        """
        Normalize image with ImageNet mean and variance
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

    def preprocess_image(self, image_path: str):
        """
        Preprocess input image for inference
        """
        # Load image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Convert BGR to RGB
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Resize image to model's expected size
        resized_image = cv2.resize(original_image, self.input_size)

        # Normalize image
        normalized_image = self.normalizeMeanVariance(resized_image)

        # Add batch dimension
        input_tensor = tf.expand_dims(normalized_image, axis=0)

        print(f"Input tensor shape: {input_tensor.shape}")

        return input_tensor, original_image

    def get_text_boxes(
        self,
        text_score: np.ndarray,
        link_score: np.ndarray,
        text_threshold: float = 0.7,
        link_threshold: float = 0.4,
        low_text: float = 0.4,
    ) -> list[np.ndarray]:
        """Extract text boxes from score maps using CRAFT's original algorithm"""
        import math

        # Make copies to avoid modifying original arrays
        textmap = text_score.copy()
        linkmap = link_score.copy()
        img_h, img_w = textmap.shape

        # Binary thresholding
        ret, text_score_bin = cv2.threshold(textmap, low_text, 1, 0)
        ret, link_score_bin = cv2.threshold(linkmap, link_threshold, 1, 0)

        # Combine text and link scores
        text_score_comb = np.clip(text_score_bin + link_score_bin, 0, 1)

        # Connected component analysis
        nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            text_score_comb.astype(np.uint8), connectivity=4
        )

        det = []
        mapper = []

        for k in range(1, nLabels):  # Skip background (label 0)
            # Size filtering - remove small components
            size = stats[k, cv2.CC_STAT_AREA]
            if size < 10:
                continue

            # Threshold filtering - suppress regions with low text confidence
            if np.max(textmap[labels == k]) < text_threshold:
                continue

            # Create segmentation map for current component
            segmap = np.zeros(textmap.shape, dtype=np.uint8)
            segmap[labels == k] = 255
            # Remove link areas that don't have text
            segmap[np.logical_and(link_score_bin == 1, text_score_bin == 0)] = 0

            # Get bounding box of component
            x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
            w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]

            # Calculate dilation iterations based on component size
            niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
            sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1

            # Boundary check
            sx = max(0, sx)
            sy = max(0, sy)
            ex = min(img_w, ex)
            ey = min(img_h, ey)

            # Morphological dilation to connect nearby text regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
            segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

            # Extract contour points and create minimum area rectangle
            np_contours = (
                np.roll(np.array(np.where(segmap != 0)), 1, axis=0)
                .transpose()
                .reshape(-1, 2)
            )

            if len(np_contours) < 4:  # Need at least 4 points for a rectangle
                continue

            rectangle = cv2.minAreaRect(np_contours)
            box = cv2.boxPoints(rectangle)

            # Handle diamond-shaped boxes (nearly square aspect ratio)
            w_box, h_box = (
                np.linalg.norm(box[0] - box[1]),
                np.linalg.norm(box[1] - box[2]),
            )
            box_ratio = max(w_box, h_box) / (min(w_box, h_box) + 1e-5)

            if abs(1 - box_ratio) <= 0.1:  # Nearly square
                # Use axis-aligned bounding box instead
                l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
                t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
                box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

            # Make clock-wise order starting from top-left
            startidx = box.sum(axis=1).argmin()  # Point with minimum x+y (top-left)
            box = np.roll(box, 4 - startidx, 0)
            box = np.array(box)

            det.append(box)
            mapper.append(k)

        return det

    def draw_text_boxes(
        self,
        image: np.ndarray,
        boxes: list,
        color: tuple = (0, 255, 0),
        thickness: int = 2,
    ):
        """Draw text boxes on image"""
        result_img = image.copy()

        for box in boxes:
            # Convert to integer coordinates
            box = box.astype(np.int32)

            # Draw polygon
            cv2.polylines(result_img, [box], True, color, thickness)

        return result_img

    def run_inference(
        self,
        image_path: str,
        text_threshold: float = 0.7,
        link_threshold: float = 0.4,
        low_text: float = 0.4,
    ):
        """Run inference on input image"""
        print(f"Running inference on: {image_path}")

        # Preprocess image
        input_tensor, original_image = self.preprocess_image(image_path)

        # Run inference with proper input format
        try:
            if self.input_key:
                # Use the correct input key name
                print(f"Using input key: {self.input_key}")
                predictions = self.infer(**{self.input_key: input_tensor})
            else:
                # Try common input names
                for input_name in ["input", "inputs", "input_1", "x"]:
                    try:
                        print(f"Trying input name: {input_name}")
                        predictions = self.infer(**{input_name: input_tensor})
                        print(f"Success with input name: {input_name}")
                        break
                    except Exception as e:
                        print(f"Failed with {input_name}: {str(e)[:100]}...")
                        continue
                else:
                    # Last resort: try positional argument
                    print("Trying positional argument...")
                    predictions = self.infer(input_tensor)

        except Exception as e:
            print(f"Error during inference: {e}")
            raise

        # Handle different output formats
        if isinstance(predictions, dict):
            print(f"Output keys: {list(predictions.keys())}")
            # Try to find the main output
            for key in ["output", "predictions", "logits", "output_0"]:
                if key in predictions:
                    predictions = predictions[key]
                    print(f"Using output key: {key}")
                    break
            else:
                # Use first available output
                output_key = list(predictions.keys())[0]
                predictions = predictions[output_key]
                print(f"Using first output key: {output_key}")

        # Convert to numpy
        if isinstance(predictions, tf.Tensor):
            predictions_np = predictions.numpy()
        else:
            predictions_np = predictions

        print(f"Predictions shape: {predictions_np.shape}")

        # Handle different output shapes
        if len(predictions_np.shape) == 4:
            # Shape: [batch, height, width, channels]
            if predictions_np.shape[-1] >= 2:
                text_score = predictions_np[0, :, :, 0]
                link_score = predictions_np[0, :, :, 1]
            else:
                text_score = predictions_np[0, :, :, 0]
                link_score = predictions_np[0, :, :, 0]  # Use same for both
        elif len(predictions_np.shape) == 3:
            # Shape: [height, width, channels]
            if predictions_np.shape[-1] >= 2:
                text_score = predictions_np[:, :, 0]
                link_score = predictions_np[:, :, 1]
            else:
                text_score = predictions_np[:, :, 0]
                link_score = predictions_np[:, :, 0]
        else:
            raise ValueError(f"Unexpected prediction shape: {predictions_np.shape}")

        print(
            f"Text score shape: {text_score.shape}, range: [{text_score.min():.3f}, {text_score.max():.3f}]"
        )
        print(
            f"Link score shape: {link_score.shape}, range: [{link_score.min():.3f}, {link_score.max():.3f}]"
        )

        # Get text boxes
        text_boxes = self.get_text_boxes(
            text_score, link_score, text_threshold, link_threshold, low_text
        )

        # Adjust coordinates to original image size
        original_h, original_w = original_image.shape[:2]
        score_h, score_w = text_score.shape

        ratio_w = original_w / score_w
        ratio_h = original_h / score_h

        print(f"Scaling ratios: w={ratio_w:.2f}, h={ratio_h:.2f}")

        adjusted_boxes = []
        for box in text_boxes:
            # Scale coordinates to original image size
            adjusted_box = np.round(box.astype(np.float32) * [ratio_w, ratio_h]).astype(
                np.int32
            )
            adjusted_boxes.append(adjusted_box)

        # Create result image
        result_image = self.draw_text_boxes(original_image, adjusted_boxes)

        return {
            "original_image": original_image,
            "result_image": result_image,
            "text_boxes": adjusted_boxes,
            "text_score": text_score,
            "link_score": link_score,
            "num_detections": len(adjusted_boxes),
        }

    def save_results(self, results: dict, output_path: str):
        """Save inference results"""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save result image
        result_bgr = cv2.cvtColor(results["result_image"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), result_bgr)

        # Save score maps
        base_name = Path(output_path).stem
        output_dir_str = str(output_dir)

        plt.imsave(
            os.path.join(output_dir_str, f"{base_name}_text_score.jpg"),
            results["text_score"],
            cmap="jet",
        )
        plt.imsave(
            os.path.join(output_dir_str, f"{base_name}_link_score.jpg"),
            results["link_score"],
            cmap="jet",
        )

        print(f"Results saved to: {output_path}")
        print(f"Detected {results['num_detections']} text regions")
        print(f"Score maps saved to: {output_dir_str}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="CRAFT Text Detection Inference using SavedModel (Fixed)"
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to SavedModel directory"
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
        help="Input image size (width height) - will be overridden by model requirements",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.model_path):
        raise ValueError(f"SavedModel path not found: {args.model_path}")

    if not os.path.exists(args.input_image):
        raise ValueError(f"Input image not found: {args.input_image}")

    # Initialize inference
    craft_inference = CRAFTSavedModelInference(
        model_path=args.model_path, input_size=tuple(args.input_size)
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
