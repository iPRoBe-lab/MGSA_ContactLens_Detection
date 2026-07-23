import os
from tqdm import tqdm
import cv2
import numpy as np

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='crop images to region of interest')
    parser.add_argument('-i', '--data', type=str, help='Path to input directory containing images')
    parser.add_argument('-o', '--output', type=str, help='Path to output directory')
    
    
    args = parser.parse_args()
    

    splits = ['train', 'val']

    for split in splits:
        split_dir = os.path.join(args.data, split)
        images_dir = os.path.join(split_dir, 'images')
        masks_dir = os.path.join(split_dir, 'masks')
        
        all_images = [img.path for img in os.scandir(images_dir) if img.is_file()]
        
        output_dir = os.path.join(args.output, split)
        cropped_dir = os.path.join(output_dir, 'cropped')
        out_images_dir = os.path.join(output_dir, 'images')
        out_masks_dir = os.path.join(output_dir, 'masks')

        os.makedirs(cropped_dir, exist_ok=True)
        os.makedirs(out_images_dir, exist_ok=True)
        os.makedirs(out_masks_dir, exist_ok=True)

        for image in tqdm(all_images):
            
            img_name = os.path.basename(image)

            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(os.path.join(masks_dir, img_name), cv2.IMREAD_GRAYSCALE)
            
            # crop the image to the region of interest  
            mask_fg = mask > 0
            bbox = cv2.boundingRect(mask_fg.astype(np.uint8))
            x, y, w, h = bbox
            
            center = (x + w // 2, y + h // 2)
            new_w = new_h = max(w, h)
            x1 = max(center[0] - new_w // 2, 0)
            y1 = max(center[1] - new_h // 2, 0)
            x2 = min(x1 + new_w, img.shape[1])
            y2 = min(y1 + new_h, img.shape[0])

            cropped_img = img[y1:y2, x1:x2]


            os.system(f'cp {image} {out_images_dir}')
            os.system(f'cp {os.path.join(masks_dir, img_name)} {out_masks_dir}')
            
            cv2.imwrite(os.path.join(cropped_dir, img_name), cropped_img)