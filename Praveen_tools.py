import os
import torch
import nodes
import folder_paths
from PIL import Image, ImageEnhance
import numpy as np
from comfy.sd import VAE
from comfy.utils import common_upscale
from nodes import common_ksampler
from comfy_extras import nodes_upscale_model
from comfy import model_management
from nodes import MAX_RESOLUTION
import math
from typing import List
from comfy.utils import ProgressBar
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
        
class ImageDimensions16:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 832, 
                    "min": 16, 
                    "max": nodes.MAX_RESOLUTION, 
                    "step": 16
                }),
                "height": ("INT", {
                    "default": 480, 
                    "min": 16, 
                    "max": nodes.MAX_RESOLUTION, 
                    "step": 16
                }),
                "length": ("INT", {
                    "default": 81, 
                    "min": 1, 
                    "max": nodes.MAX_RESOLUTION, 
                    "step": 4,
                    "tooltip": "Length for WanVace."
                }),
            },
        }
    
    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "length")
    FUNCTION = "get_dimensions"
    CATEGORY = "image/dimensions"

    def get_dimensions(self, width, height, length):
        return (width, height, length)
        


class OverlappingImageListSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "overlap_count": ("INT", {
                    "default": 6, 
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "segment_size": ("INT", {
                    "default": 40, 
                    "min": 1,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = tuple(["IMAGE"] * 6)  # Always returns 6 segments
    RETURN_NAMES = tuple([f"segment_{i+1}" for i in range(6)])
    FUNCTION = "split_images"
    CATEGORY = "image/list"

    def split_images(self, images: torch.Tensor, overlap_count: int = 6, segment_size: int = 40):
        total_images = images.shape[0]
        
        # Calculate step size (how much to move forward for each segment)
        step = segment_size - overlap_count
        
        # Calculate how many segments we can actually make
        max_possible_segments = math.ceil((total_images - overlap_count) / step)
        actual_segments = min(6, max_possible_segments)  # We'll return up to 6 segments
        
        # Prepare the output segments
        segments = []
        progress_bar = ProgressBar(actual_segments)
        
        for i in range(actual_segments):
            start = i * step
            end = start + segment_size
            
            # Adjust the last segment to not go beyond the image count
            if end > total_images:
                end = total_images
                start = max(0, end - segment_size)
            
            segment = images[start:end]
            
            # Pad with last frame if segment is smaller than segment_size
            if segment.shape[0] < segment_size:
                padding = segment[-1:].repeat(segment_size - segment.shape[0], 1, 1, 1)
                segment = torch.cat([segment, padding], dim=0)
            
            segments.append(segment)
            progress_bar.update(1)
        
        # If we have fewer than 6 segments, pad with empty tensors
        while len(segments) < 6:
            segments.append(torch.zeros_like(images[0:1]).repeat(segment_size, 1, 1, 1))
        
        return tuple(segments)
        
        

class OverlappingImageListMerger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "overlap_count": ("INT", {
                    "default": 6, 
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
            },
            "optional": {
                "segment_1": ("IMAGE",),
                "segment_2": ("IMAGE",),
                "segment_3": ("IMAGE",),
                "segment_4": ("IMAGE",),
                "segment_5": ("IMAGE",),
                "segment_6": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("merged_images",)
    FUNCTION = "merge_images"
    CATEGORY = "image/list"

    def merge_images(self, overlap_count: int = 6, segment_1=None, segment_2=None, segment_3=None, segment_4=None, segment_5=None, segment_6=None):
        # Collect all non-None segments
        segments = [seg for seg in [segment_1, segment_2, segment_3, segment_4, segment_5, segment_6] if seg is not None]
        
        if not segments:
            # Return empty tensor with correct dimensions if no inputs
            if segment_1 is not None:
                return (torch.zeros((0, *segment_1.shape[1:])),)
            return (torch.zeros((0, 1, 1, 3)),)  # Default empty tensor shape
        
        # Filter out empty segments (all zeros)
        non_empty_segments = []
        for seg in segments:
            if torch.any(seg != 0):  # Check if segment contains non-zero values
                non_empty_segments.append(seg)
        
        if not non_empty_segments:
            return (torch.zeros((0, *segments[0].shape[1:])),)
        
        # Start with the first segment
        merged = non_empty_segments[0]
        progress_bar = ProgressBar(len(non_empty_segments) - 1)
        
        # Merge subsequent segments, skipping the overlapping part
        for i in range(1, len(non_empty_segments)):
            current_segment = non_empty_segments[i]
            
            # Remove the overlapping frames from the start of current segment
            # But only if there's actually enough frames to remove
            if overlap_count < current_segment.shape[0]:
                current_segment = current_segment[overlap_count:]
            else:
                current_segment = current_segment[0:0]  # empty tensor
            
            if current_segment.shape[0] > 0:
                merged = torch.cat([merged, current_segment], dim=0)
            
            progress_bar.update(1)
        
        return (merged,)
 
 
class RGBChannelMerger:
   
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "red_channel": ("IMAGE",),
                "green_channel": ("IMAGE",),
                "blue_channel": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("merged_rgb",)
    FUNCTION = "merge_channels"
    CATEGORY = "image/channels"
    
    def merge_channels(self, red_channel, green_channel, blue_channel):
       
        
        # Ensure all inputs have the same spatial dimensions
        if (red_channel.shape[:3] != green_channel.shape[:3] or 
            red_channel.shape[:3] != blue_channel.shape[:3]):
            raise ValueError("All input images must have the same batch size, height, and width")
        
        # Get the batch size and spatial dimensions
        batch_size, height, width = red_channel.shape[:3]
        
        # Extract red channel (channel 0) from red_channel input
        if red_channel.shape[3] >= 1:
            red = red_channel[:, :, :, 0:1]  # Extract red channel
        else:
            raise ValueError("Red channel input must have at least 1 channel")
        
        # Extract green channel (channel 1) from green_channel input
        if green_channel.shape[3] >= 2:
            green = green_channel[:, :, :, 1:2]  # Extract green channel
        elif green_channel.shape[3] == 1:
            green = green_channel[:, :, :, 0:1]  # Use single channel as green
        else:
            raise ValueError("Green channel input must have at least 1 channel")
        
        # Extract blue channel (channel 2) from blue_channel input
        if blue_channel.shape[3] >= 3:
            blue = blue_channel[:, :, :, 2:3]  # Extract blue channel
        elif blue_channel.shape[3] >= 1:
            blue = blue_channel[:, :, :, 0:1]  # Use first available channel as blue
        else:
            raise ValueError("Blue channel input must have at least 1 channel")
        
        # Merge the extracted channels into a single RGB image
        merged_rgb = torch.cat([red, green, blue], dim=3)
        
        return (merged_rgb,) 



class BlackImageGenerator:
    """
    A ComfyUI node that generates a black image with controllable width and height
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {
                    "default": 512, 
                    "min": 64, 
                    "max": 8192, 
                    "step": 8,
                    "display": "number"
                }),
                "height": ("INT", {
                    "default": 512, 
                    "min": 64, 
                    "max": 8192, 
                    "step": 8,
                    "display": "number"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "display": "number"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_black_image"
    CATEGORY = "image/generate"
    
    def generate_black_image(self, width, height, batch_size):
        """
        Generate a black image with specified dimensions
        
        Args:
            width (int): Width of the image in pixels
            height (int): Height of the image in pixels  
            batch_size (int): Number of images to generate
            
        Returns:
            tuple: Tuple containing the generated black image tensor
        """
        # Create a black image tensor with shape [batch_size, height, width, channels]
        # ComfyUI expects images in BHWC format with values in range [0, 1]
        black_image = torch.zeros((batch_size, height, width, 3), dtype=torch.float32)
        
        return (black_image,)



class SkipFirstImage:
    """
    A ComfyUI node that skips the first image from an image list/batch.
    Useful for removing the first image from a batch of images.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "skip_first"
    CATEGORY = "image/batch"
    
    def skip_first(self, images):
        """
        Skip the first image from the input batch.
        
        Args:
            images: Input image tensor with shape (batch, height, width, channels)
            
        Returns:
            Tensor with the first image removed
        """
        # Check if there are any images
        if images.shape[0] == 0:
            # Return empty tensor if no images
            return (images,)
        
        # Check if there's only one image
        if images.shape[0] == 1:
            # Return empty tensor if only one image (since we're skipping it)
            empty_tensor = torch.zeros((0, images.shape[1], images.shape[2], images.shape[3]), 
                                     dtype=images.dtype, device=images.device)
            return (empty_tensor,)
        
        # Skip the first image and return the rest
        remaining_images = images[1:]
        return (remaining_images,)

class ResolutionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
            }
        }
    
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_resolution"
    CATEGORY = "utils"
    
    def get_resolution(self, width, height):
        return (width, height)




# Node registration
NODE_CLASS_MAPPINGS = {
    "SelectLastImage": SelectLastImage,
    "SplitImageList": SplitImageList,
    "MergeImageLists": MergeImageLists,
    "AdjustBrightnessContrast": AdjustBrightnessContrast,
    "ImageDimensions16": ImageDimensions16,
    "OverlappingImageListSplitter": OverlappingImageListSplitter,
    "OverlappingImageListMerger": OverlappingImageListMerger,
    "RGBChannelMerger": RGBChannelMerger,
    "BlackImageGenerator": BlackImageGenerator,
    "SkipFirstImage": SkipFirstImage,
    "ResolutionNode": ResolutionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelectLastImage": "Select Last Image",
    "SplitImageList": "Split Image List (3-Way)",
    "MergeImageLists": "Merge Image Lists (3-Way)",
    "AdjustBrightnessContrast": "Image Brightness/Contrast/Saturation/RGB",
    "ImageDimensions16": "Image Dimensions (Multiple of 16)",
    "OverlappingImageListSplitter": "Overlapping Image List Splitter (Auto)",
    "OverlappingImageListMerger": "Overlapping Image List Merger (Auto)",
    "RGBChannelMerger": "RGB Channel Merger",
    "BlackImageGenerator": "Black Image Generator",
    "SkipFirstImage": "Skip First Image",
    "ResolutionNode": "Resolution Wan",
}
