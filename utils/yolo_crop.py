import os
import cv2
import torch
import numpy as np
from PIL import Image

from ultralytics import YOLO

class YoloPredictor:
    def __init__(self,
                 yolo_model_path: str,
                 classes_dict: dict = None,
                 device: str = None,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 max_detections: int = 10):
        """
        Initializes the YoloSAM predictor.

        Args:
            yolo_model_path (str): Path to the YOLO model weights file (e.g., 'yolov8n.pt').
            classes_dict (dict, optional): A dictionary mapping class IDs to class names.
                                           If None, uses YOLO's default class names.
                                           Example: {0: 'person', 1: 'bicycle', ...}
            device (str, optional): Device to run models on ('cuda' or 'cpu').
                                    Autodetects if None.
            conf_threshold (float): Confidence threshold for YOLO detections.
            iou_threshold (float): IOU threshold for NMS in YOLO.
            max_detections (int): Maximum number of detections from YOLO.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device


        # Load YOLO model
        print(f"Loading YOLO model from: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.to(self.device)
        self.yolo_model.eval()
        print("YOLO model loaded.")


        self.classes_dict = classes_dict if classes_dict is not None else self.yolo_model.names
        self.yolo_conf = conf_threshold
        self.yolo_iou = iou_threshold
        self.yolo_max_det = max_detections

        print(f"YOLO parameters: conf={self.yolo_conf}, iou={self.yolo_iou}, max_det={self.yolo_max_det}")
        print(f"Class mapping: {self.classes_dict}")


    def _load_image(self, image_path_or_array):
        """Loads an image from path or uses numpy array directly."""
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            if image is None:
                raise FileNotFoundError(f"Image not found at {image_path_or_array}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # SAM expects RGB
        elif isinstance(image_path_or_array, np.ndarray):
            if image_path_or_array.ndim == 3 and image_path_or_array.shape[2] == 3:
                # Assuming BGR if it's a typical OpenCV image, convert to RGB
                # If it's already RGB, this might be redundant but usually harmless.
                # For more robustness, you might add a parameter to specify input format.
                image = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2RGB)
            elif image_path_or_array.ndim == 2: # Grayscale, convert to RGB
                image = cv2.cvtColor(image_path_or_array, cv2.COLOR_GRAY2RGB)
            else:
                image = image_path_or_array # Assume it's already RGB
        else:
            raise TypeError("image_path_or_array must be a path string or a NumPy array.")
        return image

    def predict(self, image_path_or_array):
        """
        Performs object detection with YOLO and then segmentation with SAM.

        Args:
            image_path_or_array (str or np.ndarray): Path to the image or a NumPy array (RGB or BGR).

        Returns:
            list: A list of dictionaries, where each dictionary contains:
                  'box': [x1, y1, x2, y2] bounding box from YOLO.
                  'class_id': Detected class ID.
                  'class_name': Detected class name.
                  'confidence': YOLO detection confidence.
        """
        original_image_rgb = self._load_image(image_path_or_array)

        # YOLO Detection
        yolo_results = self.yolo_model.predict(
            source=original_image_rgb, # Ultralytics YOLO handles RGB/BGR internally
            conf=self.yolo_conf,
            iou=self.yolo_iou,
            max_det=self.yolo_max_det,
            verbose=False # Set to True for debugging YOLO
        )

        detected_objects = []
        if not yolo_results or not yolo_results[0].boxes:
            print("No objects detected by YOLO.")
            return detected_objects

        # Get boxes in xyxy format and other info
        boxes_xyxy = yolo_results[0].boxes.xyxy.cpu().numpy()
        confidences = yolo_results[0].boxes.conf.cpu().numpy()
        class_ids = yolo_results[0].boxes.cls.cpu().numpy().astype(int)

        if boxes_xyxy.shape[0] == 0:
            print("No objects detected by YOLO after filtering.")
            return detected_objects

        for i in range(boxes_xyxy.shape[0]):
            box = boxes_xyxy[i]
            class_id = class_ids[i]
            confidence = confidences[i]

            class_name = self.classes_dict.get(class_id, f"unknown_class_{class_id}")

            detected_objects.append({
                'box': box.astype(np.int32).tolist(),  # Convert to int32 for consistency
                'class_id': class_id,
                'class_name': class_name,
                'confidence': float(confidence),
            })
        
        return detected_objects

    def remove_models(self):
        """
        Cleans up resources, if necessary.
        """
        del self.yolo_model

        # clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared.")

        print("YoloSAM predictor resources cleaned up.")

def crop_iris_using_yolo(
        image_dir: str,
        yolo_model: YoloPredictor,
        save_path: str,
        min_crop_area: int = 100,
    ):
        """
        Processes images in a directory, detects iris using a Yolo model,
        and saves cropped images of detected iris.

        Args:
            image_dir (str): Path to the directory containing images.
            yolo_model (YoloPredictor): An instance of the YoloPredictor class.
            save_path (str): Base path to save the cropped images.
                                Subdirectories will be created for each original image.
            min_crop_area (int): Minimum area (pixels) of the cropped object to be saved.
                                Helps avoid saving tiny or empty crops.
        """
        if not os.path.isdir(image_dir):
            print(f"Error: Image directory '{image_dir}' not found.")
            return

        os.makedirs(save_path, exist_ok=True)
        print(f"Saving cropped objects to subdirectories under: {save_path}")

        supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        
        # recursively find all image files in the directory
        image_files = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    image_files.append(os.path.relpath(os.path.join(root, file), image_dir))

        if not image_files:
            print(f"No supported image files found in '{image_dir}'.")
            return


        classes_to_crop = {'iris'} # only crop iris

        for image_path in image_files:
            #split into directory and filename
            image_subdir, image_filename = os.path.split(image_path)
            image_path = os.path.join(image_dir, image_path) # full path to the image

            print(f"\nProcessing image: {image_path}")

            try:
                # Load the original image using OpenCV (typically BGR)
                # This is the image we will crop from.
                # YoloSAM's predict method will handle its own loading and conversion for the models.
                original_image_bgr = cv2.imread(image_path)
                if original_image_bgr is None:
                    print(f"  Could not read image: {image_path}. Skipping.")
                    continue

                # Get predictions (YOLO boxes)
                predictions = yolo_model.predict(image_path)

                # filter only iris predictions
                if classes_to_crop is not None:
                    predictions = [pred for pred in predictions if pred['class_name'] in classes_to_crop]

                if not predictions:
                    print(f"  No iris detected in {image_filename}.")
                elif len(predictions) > 1:
                    print(f"  Warning: Found multiple iris in {image_filename}. Only the first will be processed.")

                    # sort by confidence score 
                    predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
                    # only keep the first one
                    # predictions = predictions[:1]


                # Create a subdirectory for this image's crops
                image_crop_save_dir = os.path.join(save_path, image_subdir)
                os.makedirs(image_crop_save_dir, exist_ok=True)

                image_cropped = original_image_bgr.copy()  # Default to the original image

                for i, pred in enumerate(predictions[:1]): # Only process the first prediction
                    yolo_box = pred['box']  # YOLO box
                    x1, y1, x2, y2 = yolo_box


                    # Crop the mask to the bounding rectangle
                    image_cropped = original_image_bgr[y1:y2, x1:x2]

                    # Check if the cropped area is large enough
                    if image_cropped.size < min_crop_area:
                        print(f"  Warning: Cropped area is too small ({image_cropped.size} pixels). Skipping.")
                        image_cropped = original_image_bgr
                
                # save the cropped image
                save_filename = os.path.join(image_crop_save_dir, image_filename)
                cv2.imwrite(save_filename, image_cropped)

            except Exception as e:
                print(f"  An error occurred while processing {image_path}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Crop iris regions from images using a YOLO detection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Directory of input images to process (searched recursively).",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Directory where cropped images will be saved.",
    )
    parser.add_argument(
        "--yolo_model", required=True,
        help="Path to the YOLO model weights file (e.g. pretrained_models/yolo_best.pt).",
    )
    parser.add_argument(
        "--conf", type=float, default=0.4,
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--iou", type=float, default=0.5,
        help="YOLO NMS IoU threshold.",
    )
    parser.add_argument(
        "--min_area", type=int, default=100,
        help="Minimum crop area in pixels; smaller crops fall back to the full image.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device string (e.g. 'cuda:0'). Auto-detected when omitted.",
    )

    args = parser.parse_args()

    CLASSES = {
        0: 'iris',
        1: 'pupil',
        2: 'sclera',
        3: 'upper_lashes',
        4: 'lower_lashes',
    }

    predictor = YoloPredictor(
        yolo_model_path=args.yolo_model,
        classes_dict=CLASSES,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )

    crop_iris_using_yolo(
        image_dir=args.input,
        yolo_model=predictor,
        save_path=args.output,
        min_crop_area=args.min_area,
    )

    predictor.remove_models()

