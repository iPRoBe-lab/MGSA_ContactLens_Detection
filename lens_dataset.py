import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from logging import getLogger

logger = getLogger(__name__)

class LensDatasetWithMask(Dataset):
    def __init__(self, root, class_ids, roi_ids=None, inp_dim=256, train=True):
        """
        Custom dataset loader

        Args:
            - root:         path to the dataset folder
            - class_ids:   dictionary of class names and their corresponding ids
            - roi_ids: list of ids for the region of interest (default: None)
            - inp_dim:      input dimension for the images
            - train:        whether to use the train or test transformations
        """
        self.root = root
        
        images_subdir = os.path.join(root, 'images')
        masks_subdir = os.path.join(root, 'masks')
        # cropped_subdir = 'cropped'
        
        self.class_ids = class_ids
        self.roi_ids = roi_ids

        self.data = self._load_data(images_subdir, masks_subdir)

        # Image augmentations
        image_resize = [transforms.Resize(inp_dim)]
        mask_resize = [transforms.Resize(inp_dim, interpolation=Image.NEAREST)]
        
        # Augmentation list for images
        # Note: keep spatial augmentations mild — Normal vs Clear differences
        # are subtle texture/contrast cues that strong distortions destroy.
        image_aug_list = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            # Subtle colour jitter preserves lens-induced contrast differences
            transforms.ColorJitter(brightness=0.15, contrast=0.15,
                                   saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Random erasing simulates occlusions (eyelashes, reflections)
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.0)),
        ]

        # Augmentation list for masks (spatial only, no colour)
        mask_aug_list = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor()
        ]
        
        if train:
            self.image_transform = transforms.Compose(image_resize + image_aug_list)
            self.mask_transform = transforms.Compose(mask_resize + mask_aug_list)
        else:
            tensor_transform = [transforms.ToTensor()]
            self.image_transform = transforms.Compose(image_resize + tensor_transform + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.mask_transform = transforms.Compose(mask_resize + tensor_transform)
    
    def _load_data(self, images_dir, masks_dir):
        """ Load data by pairing the image and mask paths """
        data = []

        images = [img for img in os.scandir(images_dir) if img.is_file()]

        img_names = [os.path.splitext(os.path.basename(img))[0] for img in images]
        
        masks = [os.path.join(masks_dir, f'{img_name}.png') for img_name in img_names]
        
        # remove images without corresponding masks
        for img, mask in zip(images, masks):
            if os.path.exists(mask):
                
                for class_name in self.class_ids.keys():
                    # check if the class name is in the image name without case sensitivity
                    if class_name.lower() in img.name.lower():
                        cls = self.class_ids[class_name]
                        data.append((img.path, mask, cls))
                        break        
        return data
    
    def __getitem__(self, index):
        
        img_path, mask_path, label = self.data[index]
        img = Image.open(img_path).convert('RGB') # 3-channel image
        
        mask = Image.open(mask_path).convert('L') # 1-channel image

        # only keep the region of interest
        if self.roi_ids is not None:
            mask = mask.point(lambda p: p in self.roi_ids, '1')
        else:
            mask = mask.point(lambda p: p > 0, '1')
        
        # transform the images
        
        # Use a fresh random seed per sample so augmentation varies across epochs.
        # Both image and mask use the SAME seed so spatial transforms stay in sync.
        seed = torch.randint(0, 2**31 - 1, (1,)).item()
        torch.manual_seed(seed)
        img  = self.image_transform(img)
        torch.manual_seed(seed)
        mask = self.mask_transform(mask)

        return img, mask, label
    
    def __len__(self):
        return len(self.data)
    
    def _get_labels(self):
        return [label for _, _, label in self.data]

class LensDatasetCropped(Dataset):
    """
    Dataset of pre-cropped iris images for Stage-2 (Normal vs Clear).

    Images are grayscale iris crops stored in ``root/cropped/``.
    They are converted to 3-channel (by replicating the gray channel) so
    ImageNet-pretrained backbones can be used without modification.

    Returns ``(img, dummy_mask, label)`` where ``dummy_mask`` is a single-pixel
    zero tensor — this keeps the interface identical to LensDatasetWithMask so
    the same training loop and identity_cross_validate() work unchanged.

    The ``data`` attribute holds ``(img_path, label)`` tuples; identity is
    extracted from the leading numeric part of the filename (e.g.
    ``10_20120913_L_Normal.bmp`` → identity ``"10"``).
    """

    def __init__(self, root, class_ids, inp_dim=224, train=True,
                 extra_dirs: list | None = None):
        """
        Args:
            extra_dirs: list of additional flat image directories to include
                        (e.g. UND cropped folders). Identity is extracted as
                        the part before the first 'd' in the filename, prefixed
                        with 'und_' to avoid collisions with IIITD subject IDs.
        """
        self.root      = root
        self.class_ids = class_ids
        self.data      = self._load_data(os.path.join(root, 'cropped'))
        if extra_dirs:
            for d in extra_dirs:
                self.data += self._load_flat_dir(d)

        # Grayscale iris → 3-channel + ImageNet normalisation
        to_3ch = transforms.Grayscale(num_output_channels=3)
        resize = transforms.Resize((inp_dim, inp_dim) if isinstance(inp_dim, int)
                                   else tuple(inp_dim))
        norm   = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

        if train:
            self.image_transform = transforms.Compose([
                to_3ch,
                resize,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08),
                                        scale=(0.9, 1.1)),
                # Brightness/contrast only — hue/saturation meaningless for 3=gray
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
                transforms.RandomAutocontrast(p=0.3),
                transforms.ToTensor(),
                norm,
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.08)),
            ])
        else:
            self.image_transform = transforms.Compose([
                to_3ch, resize, transforms.ToTensor(), norm,
            ])

    def _load_data(self, images_dir):
        data = []
        for img in os.scandir(images_dir):
            if not img.is_file():
                continue
            for class_name in self.class_ids.keys():
                if class_name.lower() in img.name.lower():
                    data.append((img.path, self.class_ids[class_name]))
                    break
        return data

    def _load_flat_dir(self, images_dir: str) -> list:
        """Load a flat directory of images (e.g. UND dataset)."""
        data = []
        for img in os.scandir(images_dir):
            if not img.is_file():
                continue
            for class_name in self.class_ids.keys():
                if class_name.lower() in img.name.lower():
                    data.append((img.path, self.class_ids[class_name]))
                    break
        return data

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = Image.open(img_path).convert('L')  # keep as grayscale; to_3ch handles it
        img = self.image_transform(img)
        # Dummy single-pixel mask keeps (img, mask, label) interface intact
        dummy_mask = torch.zeros(1, 1, 1)
        return img, dummy_mask, label

    def __len__(self):
        return len(self.data)

    def _get_labels(self):
        return [lbl for _, lbl in self.data]

class LensDatasetCombined(Dataset):
    
    def __init__(self, root, class_ids, roi_ids=None, inp_dim=256, cropped_inp_dim=256):
        """
        Combined dataset loader for images, masks, and cropped images.

        Args:
            - root:         path to the dataset folder
            - class_ids:   dictionary of class names and their corresponding ids
            - roi_ids:     list of ids for the region of interest (default: None)
            - inp_dim:     input dimension for the images
            - cropped_inp_dim: input dimension for the cropped images
        """
        self.root = root
        self.class_ids = class_ids
        self.roi_ids = roi_ids
        self.inp_dim = inp_dim

        images_dir = os.path.join(self.root, 'images')
        masks_dir = os.path.join(self.root, 'masks')
        cropped_dir = os.path.join(self.root, 'cropped')

        self.data = self._load_data(images_dir, masks_dir, cropped_dir)

        tensor_transform = [transforms.ToTensor()]
        image_resize = [transforms.Resize(inp_dim)]
        cropped_resize = [transforms.Resize(cropped_inp_dim)]
        mask_resize = [transforms.Resize(inp_dim, interpolation=Image.NEAREST)]

        self.image_transform = transforms.Compose(image_resize + tensor_transform + [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose(mask_resize + tensor_transform)
        self.cropped_transform = transforms.Compose(cropped_resize + tensor_transform + [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_data(self, images_dir, masks_dir, cropped_dir):

        data = []

        images = [img for img in os.scandir(images_dir) if img.is_file()]
        cropped_dict = {os.path.splitext(img.name)[0]: img.path 
                   for img in os.scandir(cropped_dir) if img.is_file()
                }
        mask_dict = {os.path.splitext(img.name)[0]: img.path
                     for img in os.scandir(masks_dir) if img.is_file()
                }


        for img in images:
            img_path = img.path
            img_name = os.path.splitext(img.name)[0]
            mask_path = mask_dict.get(img_name, None)
            cropped_path = cropped_dict.get(img_name, None)

            if mask_path is not None and cropped_path is not None \
                  and os.path.exists(mask_path) and os.path.exists(cropped_path):
                
                for class_name in self.class_ids.keys():
                    if class_name.lower() in img_name.lower():
                        cls = self.class_ids[class_name]
                        data.append((img_path, mask_path, cropped_path, cls))
                        break

        return data

    def __getitem__(self, index):
        img_path, mask_path, cropped_path, img_label = self.data[index]

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        cropped_img = Image.open(cropped_path).convert('RGB')

        if self.roi_ids is not None:
            mask = mask.point(lambda p: p in self.roi_ids, '1')
        else:
            mask = mask.point(lambda p: p > 0, '1')

        img = self.image_transform(img)
        mask = self.mask_transform(mask)
        cropped_img = self.cropped_transform(cropped_img)

        img_name = os.path.basename(img_path)

        return img, mask, cropped_img, img_label, img_name

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":

    # Test the dataset creation & dataloader
    data_path = 'data/selected_data/train'
    import yaml
    
    cfg = yaml.load(open('configs/eval_two_stage.yaml', 'r'), Loader=yaml.SafeLoader)
    print(cfg)

    seg_classes = cfg['seg_classes']
    roi_classes = cfg['roi_classes']
    roi_ids = [seg_classes.index(roi) + 1 for roi in roi_classes] if roi_classes else None
    print(f"Region of Interest: {roi_classes}, ROI Index: {roi_ids}")

    # custom_dataset = LensDatasetWithMask(data_path, cfg['class_ids'], roi_ids=roi_ids, inp_dim=cfg['inp_dim'], train=False)
    # print(f"Number of samples: {len(custom_dataset)}")

    custom_dataset = LensDatasetCombined(data_path, cfg['class_ids'], roi_ids=roi_ids, inp_dim=cfg['inp_dim'], cropped_inp_dim=cfg['cropped_inp_dim'])

    class_lsit = []
    for i in range(len(custom_dataset)):
        _, _, _, label, _ = custom_dataset[i]
        class_lsit.append(label)
    import numpy as np
    print(np.unique(class_lsit, return_counts=True))


    # batch = custom_dataset[0]
    # inputs, masks, labels = batch
    # print(custom_dataset.data[0])
    # print(f"Batch Input Shape: {inputs.shape}, Batch Label Shape: {labels}, Batch Mask Shape: {masks.shape}")
    
    # print(torch.sum(masks))
    
    # import cv2
    # cv2.imwrite('mask.png', masks.squeeze().numpy() * 255)
