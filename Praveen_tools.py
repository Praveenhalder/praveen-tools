import os
import torch
import folder_paths
from PIL import Image, ImageEnhance
import numpy as np
from nodes import *

class SelectLastImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "images": ("IMAGE", {"forceInput": True}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "select_last"
    CATEGORY = "image/utility"

    def select_last(self, images=None):
        if images is None:
            raise ValueError("No images provided to SelectLastImage node")
            
        # Ensure we have a batch of images (add batch dimension if single image)
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
            
        if images.shape[0] == 0:
            raise ValueError("Empty image list provided to SelectLastImage node")
            
        # Select the last image in the batch
        last_image = images[-1].unsqueeze(0)
        return (last_image,)




class SplitImageList:
    """
    Split one image list into three separate image lists.
    Outputs are in the order: [first third, middle third, last third]
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "split_ratio_1": ("FLOAT", {"default": 0.33, "min": 0.0, "max": 1.0, "step": 0.01}),
                "split_ratio_2": ("FLOAT", {"default": 0.33, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("images_1", "images_2", "images_3")
    FUNCTION = "split_images"
    CATEGORY = "image/list"

    def split_images(self, images, split_ratio_1=0.33, split_ratio_2=0.33):
        # Calculate split points based on ratios
        total = images.shape[0]
        split1 = max(1, min(total - 2, round(total * split_ratio_1)))
        remaining = total - split1
        split2 = max(1, min(remaining - 1, round(remaining * (split_ratio_2 / (1 - split_ratio_1)))))
        
        # Split the tensor
        images1 = images[:split1]
        images2 = images[split1:split1+split2]
        images3 = images[split1+split2:]
        
        return (images1, images2, images3)
  
    
    
class MergeImageLists:
    """
    Merge three image lists back into one combined list.
    Maintains the original order: first list + second list + third list.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_1": ("IMAGE",),
                "images_2": ("IMAGE",),
                "images_3": ("IMAGE",),
            },
            "optional": {
                "merge_order": (["1-2-3", "1-3-2", "2-1-3", "2-3-1", "3-1-2", "3-2-1"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("merged_images",)
    FUNCTION = "merge_images"
    CATEGORY = "image/list"

    def merge_images(self, images_1, images_2, images_3, merge_order="1-2-3"):
        # Convert all inputs to tensors if they aren't already
        img_tensors = {
            "1": images_1,
            "2": images_2,
            "3": images_3
        }
        
        # Get the merge order
        order = merge_order.split('-')
        
        # Concatenate in specified order
        merged = torch.cat([img_tensors[o] for o in order], dim=0)
        
        return (merged,)
        

class AdjustBrightnessContrast:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01
                }),
                "contrast": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01
                }),
                "saturation": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01
                }),
                "red_gain": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01
                }),
                "green_gain": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01
                }),
                "blue_gain": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust"
    CATEGORY = "image/postprocessing"

    def adjust(self, image, brightness=1.0, contrast=1.0, saturation=1.0, red_gain=1.0, green_gain=1.0, blue_gain=1.0):
        batch_size, height, width, channels = image.shape
        result = torch.zeros_like(image)

        for b in range(batch_size):
            img_tensor = image[b]
            img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np).convert("RGB")

            # Apply brightness, contrast, saturation
            if brightness != 1.0:
                img_pil = ImageEnhance.Brightness(img_pil).enhance(brightness)
            if contrast != 1.0:
                img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast)
            if saturation != 1.0:
                img_pil = ImageEnhance.Color(img_pil).enhance(saturation)

            # Convert to numpy for RGB gain adjustment
            img_np = np.array(img_pil).astype(np.float32) / 255.0

            # Apply RGB gains per channel
            img_np[..., 0] *= red_gain   # R
            img_np[..., 1] *= green_gain # G
            img_np[..., 2] *= blue_gain  # B

            # Clip to valid range
            img_np = np.clip(img_np, 0.0, 1.0)

            result[b] = torch.from_numpy(img_np)

        return (result,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "SelectLastImage": SelectLastImage,
    "SplitImageList": SplitImageList,
    "MergeImageLists": MergeImageLists,
    "AdjustBrightnessContrast": AdjustBrightnessContrast,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelectLastImage": "Select Last Image",
    "SplitImageList": "Split Image List (3-Way)",
    "MergeImageLists": "Merge Image Lists (3-Way)",
    "AdjustBrightnessContrast": "Image Brightness/Contrast/Saturation/RGB",
}
