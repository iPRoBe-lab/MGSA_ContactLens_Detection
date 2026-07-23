# this code implements  Automatic Detection of Non-Cosmetic Soft Contact Lenses in Ocular Images

import cv2
import numpy as np
import os
import yaml
import logging
from logging import getLogger
from tqdm import tqdm


# add path ofthe directory containing iris_seg module to the python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import random
import numpy as np
np.random.seed(42)
random.seed(42)

# from iris_seg.segment import segment
from utils.iris_seg import get_iris_boundaries


logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)

def plot_cricle(image, cx, cy, radius, color):
    image = cv2.circle(image, (cx, cy), radius, color, 2)
    return image

def draw_points(image, points, color, size=1):
    for point in points:
        image = cv2.circle(image, point, size, color, -1)
    return image

def gaussian_filter(image, kernel_size=5, sigma=1.0):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def get_circle_mask(circle, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    mask = cv2.circle(mask, (circle[0], circle[1]), circle[2], 1, -1)
    return mask.astype(bool)

def get_arc_mask(image, circle, angle_range, step, min_pix_offset, max_pix_offset):
    cx, cy, radius = circle
    h, w = image.shape
    
    # Create an empty mask of the same shape as the image
    mask = np.zeros((h, w), dtype=np.uint8)

    # Convert degrees to radians
    start_angle = np.deg2rad(angle_range[0])
    end_angle = np.deg2rad(angle_range[1])
    step = np.deg2rad(step)

    # Generate the angles over which we will iterate
    angles = np.arange(start_angle, end_angle, step)

    # Traverse radially between the min_pix_offset and max_pix_offset
    for theta in angles:
        for r in range(radius + min_pix_offset, radius + max_pix_offset, 1):
            # Calculate x, y for each point in the radial line
            x = int(cx + r * np.sin(theta))  # x corresponds to column
            y = int(cy + r * np.cos(theta))  # y corresponds to row
            

            # Check if the calculated coordinates are within the image bounds
            if 0 <= x < w and 0 <= y < h:
                mask[y, x] = True  # Mark the pixel as part of the mask

    # slightly dialte the mask to fill the gasp
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), 1)
    
    return mask.astype(bool)

# Function to overlay the class mask on top of an image
def overlay_class_mask(image, mask, class_colors, class_idx, alpha=0.5):
    # Get the height and width of the mask and image
    h, w = mask.shape
    
    # Ensure the image has 3 channels (RGB) if it is grayscale
    if len(image.shape) == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Create an empty color mask (h, w, 3)
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Map each class index to the corresponding color
    for cls, idx in class_idx.items():
        color_mask[mask == idx] = class_colors[cls]
    
    # Blend the original image and the colored mask
    blended = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
    
    return blended
    

def get_roi_mask(image, configs, debug=False, return_success=False):
    """
    Get the region of interest mask for the given eye image.
    This includes the iris, pupil and arc regions around the iris.

    Args:
        image:          Grayscale image as a NumPy array.
        configs:        Data-prep config dict.
        debug:          If True, also return a visualisation image.
        return_success: If True, append a boolean flag indicating whether
                        iris boundaries were found successfully.

    Returns:
        (mask, vis_image)                   when return_success=False
        (mask, vis_image, success: bool)    when return_success=True
    """
    success = True
    try:
        logger.debug('Finding iris boundaries')
        cirpupil, ciriris = get_iris_boundaries(image)
        logger.debug('Iris boundaries found')
        logger.debug(f'pupil: {cirpupil}, iris: {ciriris}')
        
        iris_mask = get_circle_mask(ciriris, image.shape)
        pupil_mask = get_circle_mask(cirpupil, image.shape)
        
        iris_mask = iris_mask & ~pupil_mask
    
    
        image = gaussian_filter(image, kernel_size=configs['gaus_kernel_size'], sigma=configs['gaus_sigma'])
        
        
        configs_tr1 = configs['traversal']
        
        logger.debug('Traversing radially')
        left_arc_mask = get_arc_mask(image, ciriris, configs_tr1['left_interval'], configs_tr1['angle_step'],
                                    configs_tr1['min_offset'], configs_tr1['max_offset'])
        
        right_arc_mask = get_arc_mask(image, ciriris, configs_tr1['right_interval'], configs_tr1['angle_step'],
                                    configs_tr1['min_offset'], configs_tr1['max_offset'])
        arc_mask = left_arc_mask | right_arc_mask
    except:
        logger.warning('Failed to find iris boundaries, going with empty masks')
        iris_mask = np.zeros(image.shape, dtype=bool)
        pupil_mask = np.zeros(image.shape, dtype=bool)
        arc_mask = np.zeros(image.shape, dtype=bool)
        success = False
        
    
    # combine the two arc masks and pupil and iris masks
    classes = configs['seg_classes']
    classes_idx = {c: i+1 for i, c in enumerate(classes)}
    
    combined_mask = np.zeros(image.shape, dtype=np.uint8)
    combined_mask[iris_mask] = classes_idx['iris']
    combined_mask[pupil_mask] = classes_idx['pupil']
    combined_mask[arc_mask] = classes_idx['outer_arc']
    
    
    # draw points on debug image
    if debug:
        class_colors = configs['class_colors']
        vis_image = overlay_class_mask(image, combined_mask, class_colors, classes_idx, 0.4)
    else:
        vis_image = None

    if return_success:
        return combined_mask, vis_image, success
    return combined_mask, vis_image

def prepare_masks(
        image_dir: str,
        configs: dict,
        save_path: str
    ):

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

        for image_path in image_files:
            #split into directory and filename
            image_subdir, image_filename = os.path.split(image_path)
            image_path = os.path.join(image_dir, image_path) # full path to the image

            print(f"\nProcessing image: {image_path}")

            try:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                mask, vis_image = get_roi_mask(img, configs, debug=True)
            
                mask_save_dir = os.path.join(save_path, image_subdir)
                os.makedirs(mask_save_dir, exist_ok=True)
                
                mask_path = os.path.join(mask_save_dir, image_filename)
                cv2.imwrite(mask_path, mask)

                if vis_image is not None:
                    vis_save_dir = os.path.join(mask_save_dir, 'visualizations')
                    os.makedirs(vis_save_dir, exist_ok=True)
                    vis_path = os.path.join(vis_save_dir, image_filename)
                    cv2.imwrite(vis_path, vis_image)
            except Exception as e:
                print(f"Error processing image '{image_path}': {e}")
                continue

    

def split_train_val(image_files, split_ratios):
    """
    split the image files into training and validation sets
    args:
        image_files: list of image files
        labels: list of labels
        split_ratios: dict of splits, key: split name, value: split ratio
    """
    
    spilt_sum = sum(split_ratios.values())
    assert spilt_sum == 1, 'Sum of split ratios should be 1'
    
    num_samples = len(image_files)
    split_names = list(split_ratios.keys())
    split_sizes = [int(num_samples * split_ratios[k]) for k in split_names]
    
    indices = np.random.permutation(num_samples)
    data_splits = {}
    cur_idx = 0
    for i, size in enumerate(split_sizes):
        if i == len(split_sizes) - 1:
            split_idx = indices[cur_idx:]
        else:
            split_idx = indices[cur_idx: cur_idx + size]
        
        data_splits[split_names[i]] = [image_files[idx] for idx in split_idx]
        
        cur_idx += size
    
    return data_splits


if __name__ == '__main__':

    # IMAGE_DIR = "data/Iris/Test" # Replace with your image directory

    # SAVE_PATH = "data/Iris/Test_masks"  # Directory to save masks

    # CONFIGS_PATH = 'configs/data_prep.yaml'  # Path to the config file

    # with open(CONFIGS_PATH, 'r') as f:
    #     configs = yaml.safe_load(f)
    
    # # Prepare masks for the images in the directory
    # prepare_masks(IMAGE_DIR, configs, SAVE_PATH)
    
    import argparse
    parser = argparse.ArgumentParser(description='Detects contact lenses in eye images')
    parser.add_argument('-i', '--input', type=str, help='Path to input directory containing images')
    parser.add_argument('-o', '--output', type=str, help='Path to output directory')
    parser.add_argument('-c', '--config', type=str, default='configs/data_prep.yaml',
                        help='Path to config file')
    parser.add_argument('-d', '--debug', action='store_true', help='Show debug image')
    
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        configs = yaml.safe_load(f)
    
    all_images = []
    
    if os.path.isdir(args.input):
        all_images = [img.path for img in os.scandir(args.input) if img.is_file()]
    else:
        print('Input should be a directory containing images')
        exit(1)
    
    assert all_images, 'No images found in the input directory'
    
    # split data into training and validation sets
    data_splits = split_train_val(all_images, configs['data_splits'])
    root_dir = os.path.dirname(os.path.normpath(args.input))
    for split_name, split_data in data_splits.items():
    
        split_dir = os.path.join(args.output, split_name)
        images_dir = os.path.join(split_dir, 'images')
        masks_dir = os.path.join(split_dir, 'masks')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        
        if args.debug:
            vis_dir = os.path.join(split_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
        
        for image in tqdm(split_data):
            
            logger.info(f'Processing {image}')
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            
            mask, vis_image = get_roi_mask(img, configs, debug=args.debug)
            # prev_img_name = os.path.basename(image).replace('.tiff', '.png')
            # mask = cv2.imread(os.path.join(root_dir, 'masks', prev_img_name), cv2.IMREAD_GRAYSCALE)
            # vis_image = cv2.imread(os.path.join(root_dir, 'visualizations', prev_img_name), cv2.IMREAD_COLOR)
        
            img_name = os.path.splitext(os.path.basename(image))[0]
            
            img_path = os.path.join(images_dir, f'{img_name}.png')
            cv2.imwrite(img_path, img)
            
            mask_path = os.path.join(masks_dir, f'{img_name}.png')
            cv2.imwrite(mask_path, mask)
            
            if args.debug:    
                vis_path = os.path.join(vis_dir, f'{img_name}.png')
                cv2.imwrite(vis_path, vis_image)
            
        
    
            
        
        
        
        
    
    
    
    
    