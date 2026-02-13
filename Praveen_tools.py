import os
import torch
import torch.nn.functional as F
import nodes
import folder_paths
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import numpy as np
from comfy.sd import VAE
from comfy.utils import common_upscale
from nodes import common_ksampler
from comfy_extras import nodes_upscale_model
from comfy import model_management
from nodes import MAX_RESOLUTION
import math
from typing import List, Tuple
from comfy.utils import ProgressBar
import folder_paths
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


class LoadImageWithFilename:
    """
    A ComfyUI node that loads an image and outputs both the image tensor and filename.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True})
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "filename")
    FUNCTION = "load_image"
    CATEGORY = "image"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        
        # Load the image
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode == 'I':
            img = img.point(lambda i: i * (1 / 255))
        img = img.convert("RGB")
        
        # Convert to numpy array and normalize
        image_np = np.array(img).astype(np.float32) / 255.0
        
        # Convert to torch tensor with shape [1, H, W, C]
        image_tensor = torch.from_numpy(image_np)[None,]
        
        # Get just the filename without path
        filename = os.path.basename(image)
        
        return (image_tensor, filename)

    @classmethod
    def IS_CHANGED(cls, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True


class DiagonalTextWatermark:
    """
    A ComfyUI node that adds a semi-transparent diagonal text watermark to images
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {
                    "default": "WATERMARK",
                    "multiline": False
                }),
                "font_size": ("INT", {
                    "default": 48,
                    "min": 10,
                    "max": 500,
                    "step": 1
                }),
                "opacity": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "angle": ("FLOAT", {
                    "default": -45.0,
                    "min": -180.0,
                    "max": 180.0,
                    "step": 1.0
                }),
                "color": (["white", "black", "red", "blue", "green"],),
                "repeat": ("BOOLEAN", {
                    "default": True
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_watermark"
    CATEGORY = "image/watermark"
    
    def add_watermark(self, image, text, font_size, opacity, angle, color, repeat):
        # Convert from ComfyUI tensor format (B, H, W, C) to PIL
        batch_size = image.shape[0]
        result = []
        
        # Color mapping
        color_map = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "blue": (0, 0, 255),
            "green": (0, 255, 0)
        }
        text_color = color_map[color]
        
        for i in range(batch_size):
            # Convert tensor to PIL Image
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            
            # Create watermark layer
            watermark = self.create_watermark(
                img_pil.size, text, font_size, text_color, angle, repeat
            )
            
            # Apply opacity to just the text (not the transparent background)
            alpha = watermark.split()[3]
            alpha = alpha.point(lambda p: int(p * opacity))
            watermark.putalpha(alpha)
            
            # Composite watermark onto image
            img_rgba = img_pil.convert('RGBA')
            img_with_watermark = Image.alpha_composite(img_rgba, watermark).convert('RGB')
            
            # Convert back to tensor
            result_np = np.array(img_with_watermark).astype(np.float32) / 255.0
            result.append(torch.from_numpy(result_np))
        
        return (torch.stack(result),)
    
    def create_watermark(self, size, text, font_size, color, angle, repeat):
        """Create the watermark layer"""
        width, height = size
        
        # Create transparent overlay
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Try to use default font, fallback to PIL default
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        if repeat:
            # Calculate diagonal spacing for repeated watermarks
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Create a larger canvas for rotation
            diagonal = int(math.sqrt(width**2 + height**2))
            temp_img = Image.new('RGBA', (diagonal * 2, diagonal * 2), (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            
            # Spacing between repeated text
            spacing_x = text_width + 100
            spacing_y = text_height + 100
            
            # Draw repeated text
            for y in range(-diagonal, diagonal * 3, spacing_y):
                for x in range(-diagonal, diagonal * 3, spacing_x):
                    temp_draw.text((x, y), text, font=font, fill=color + (255,))
            
            # Rotate the entire pattern
            rotated = temp_img.rotate(angle, expand=False, resample=Image.BICUBIC)
            
            # Crop to original size (centered)
            left = (rotated.width - width) // 2
            top = (rotated.height - height) // 2
            overlay = rotated.crop((left, top, left + width, top + height))
        else:
            # Single centered watermark
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Create temporary image for text
            temp_img = Image.new('RGBA', (text_width + 20, text_height + 20), (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            temp_draw.text((10, 10), text, font=font, fill=color + (255,))
            
            # Rotate text
            rotated_text = temp_img.rotate(angle, expand=True, resample=Image.BICUBIC)
            
            # Calculate position to center
            x = (width - rotated_text.width) // 2
            y = (height - rotated_text.height) // 2
            
            # Paste onto overlay
            overlay.paste(rotated_text, (x, y), rotated_text)
        
        return overlay

class ImageTileSplit:
    """
    Split an image into tiles with configurable overlap.
    Outputs tiles in row-major order for reliable merging.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "grid_size": (["2x2", "3x3", "4x4", "5x5", "6x6", "8x8", "10x10", "12x12", "16x16"], {
                    "default": "4x4"
                }),
                "overlap_percent": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.5,
                    "display": "slider"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "TILE_INFO")
    RETURN_NAMES = ("tiles", "tile_info")
    FUNCTION = "split_image"
    CATEGORY = "image/tile"
    
    def split_image(self, image: torch.Tensor, grid_size: str, overlap_percent: float) -> Tuple[torch.Tensor, dict]:
        """
        Split image into tiles with overlap.
        
        Args:
            image: Input tensor [B, H, W, C]
            grid_size: Grid configuration string (e.g., "4x4")
            overlap_percent: Overlap as percentage of tile size (0-50)
        
        Returns:
            tiles: Stacked tile tensor [B*N, tile_h, tile_w, C]
            tile_info: Dictionary with reconstruction metadata
        """
        # Parse grid size
        grid_n = int(grid_size.split("x")[0])
        
        B, H, W, C = image.shape
        
        # Calculate base tile size (without overlap)
        base_tile_h = H / grid_n
        base_tile_w = W / grid_n
        
        # Calculate overlap in pixels (applied to each side)
        overlap_h = int(math.ceil(base_tile_h * overlap_percent / 100.0))
        overlap_w = int(math.ceil(base_tile_w * overlap_percent / 100.0))
        
        # Calculate tile dimensions with overlap
        tile_h = int(math.ceil(base_tile_h)) + 2 * overlap_h
        tile_w = int(math.ceil(base_tile_w)) + 2 * overlap_w
        
        # Pad image to handle edge tiles
        padded_image = F.pad(
            image.permute(0, 3, 1, 2),  # [B, C, H, W]
            (overlap_w, overlap_w + (grid_n * int(math.ceil(base_tile_w)) - W),
             overlap_h, overlap_h + (grid_n * int(math.ceil(base_tile_h)) - H)),
            mode='reflect'
        ).permute(0, 2, 3, 1)  # [B, H, W, C]
        
        tiles = []
        tile_coords = []
        
        for row in range(grid_n):
            for col in range(grid_n):
                # Calculate tile boundaries (pixel-accurate)
                y_start = int(row * base_tile_h)
                x_start = int(col * base_tile_w)
                
                # Extract tile with overlap from padded image
                tile = padded_image[
                    :,
                    y_start:y_start + tile_h,
                    x_start:x_start + tile_w,
                    :
                ]
                
                tiles.append(tile)
                tile_coords.append({
                    'row': row,
                    'col': col,
                    'y_start': y_start,
                    'x_start': x_start,
                    'base_h': int(math.ceil(base_tile_h)),
                    'base_w': int(math.ceil(base_tile_w))
                })
        
        # Stack tiles: [B*N, tile_h, tile_w, C]
        stacked_tiles = torch.cat(tiles, dim=0)
        
        # Store metadata for reconstruction
        tile_info = {
            'grid_n': grid_n,
            'original_h': H,
            'original_w': W,
            'tile_h': tile_h,
            'tile_w': tile_w,
            'overlap_h': overlap_h,
            'overlap_w': overlap_w,
            'overlap_percent': overlap_percent,
            'base_tile_h': int(math.ceil(base_tile_h)),
            'base_tile_w': int(math.ceil(base_tile_w)),
            'batch_size': B,
            'channels': C,
            'tile_coords': tile_coords
        }
        
        return (stacked_tiles, tile_info)


class ImageTileMerge:
    """
    Merge tiles back into a single image with seamless blending.
    Supports upscaled tiles and independent crop/fade controls.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "tile_info": ("TILE_INFO",),
                "crop_percent": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.5,
                    "display": "slider"
                }),
                "fade_percent": ("FLOAT", {
                    "default": 100.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "display": "slider"
                }),
                "blend_mode": (["linear", "cosine"], {
                    "default": "cosine"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "merge_tiles"
    CATEGORY = "image/tile"
    
    def create_blend_mask(self, size: int, fade_pixels: int, blend_mode: str) -> torch.Tensor:
        """Create a 1D blend ramp for seamless transitions."""
        if fade_pixels <= 0:
            return torch.ones(size)
        
        mask = torch.ones(size)
        ramp = torch.linspace(0, 1, fade_pixels)
        
        if blend_mode == "cosine":
            # Smoother cosine interpolation
            ramp = (1 - torch.cos(ramp * math.pi)) / 2
        
        # Apply ramp to start
        mask[:fade_pixels] = ramp
        # Apply ramp to end
        mask[-fade_pixels:] = ramp.flip(0)
        
        return mask
    
    def merge_tiles(self, tiles: torch.Tensor, tile_info: dict, 
                    crop_percent: float, fade_percent: float, 
                    blend_mode: str) -> Tuple[torch.Tensor]:
        """
        Merge tiles back into original image with blending.
        
        Args:
            tiles: Tile tensor [B*N, tile_h, tile_w, C]
            tile_info: Metadata from split operation
            crop_percent: How much of overlap to crop (0-50)
            fade_percent: How much of remaining overlap to fade (0-100)
            blend_mode: "linear" or "cosine" blending
        
        Returns:
            Reconstructed image tensor [B, H, W, C]
        """
        grid_n = tile_info['grid_n']
        original_h = tile_info['original_h']
        original_w = tile_info['original_w']
        B = tile_info['batch_size']
        C = tile_info['channels']
        orig_overlap_h = tile_info['overlap_h']
        orig_overlap_w = tile_info['overlap_w']
        orig_tile_h = tile_info['tile_h']
        orig_tile_w = tile_info['tile_w']
        orig_base_h = tile_info['base_tile_h']
        orig_base_w = tile_info['base_tile_w']
        
        # Get current tile dimensions (may be upscaled)
        _, curr_tile_h, curr_tile_w, _ = tiles.shape
        
        # Calculate scale factor (for upscaled tiles)
        scale_h = curr_tile_h / orig_tile_h
        scale_w = curr_tile_w / orig_tile_w
        
        # Scale all dimensions
        overlap_h = int(round(orig_overlap_h * scale_h))
        overlap_w = int(round(orig_overlap_w * scale_w))
        base_h = int(round(orig_base_h * scale_h))
        base_w = int(round(orig_base_w * scale_w))
        out_h = int(round(original_h * scale_h))
        out_w = int(round(original_w * scale_w))
        
        # Calculate crop amounts
        crop_h = int(overlap_h * crop_percent / 100.0)
        crop_w = int(overlap_w * crop_percent / 100.0)
        
        # Remaining overlap after crop
        remaining_overlap_h = overlap_h - crop_h
        remaining_overlap_w = overlap_w - crop_w
        
        # Fade zone within remaining overlap
        fade_h = int(remaining_overlap_h * fade_percent / 100.0)
        fade_w = int(remaining_overlap_w * fade_percent / 100.0)
        
        # Initialize output tensors
        output = torch.zeros(B, out_h, out_w, C, dtype=tiles.dtype, device=tiles.device)
        weight_sum = torch.zeros(B, out_h, out_w, 1, dtype=tiles.dtype, device=tiles.device)
        
        num_tiles = grid_n * grid_n
        
        for batch_idx in range(B):
            for tile_idx in range(num_tiles):
                # Get tile for this batch
                flat_idx = batch_idx * num_tiles + tile_idx
                tile = tiles[flat_idx]
                
                row = tile_idx // grid_n
                col = tile_idx % grid_n
                
                # Crop the tile
                cropped_tile = tile[crop_h:curr_tile_h - crop_h, 
                                   crop_w:curr_tile_w - crop_w, :]
                
                cropped_h, cropped_w, _ = cropped_tile.shape
                
                # Calculate position in output (accounting for crop)
                y_pos = int(round(row * base_h * scale_h / scale_h)) 
                x_pos = int(round(col * base_w * scale_w / scale_w))
                y_pos = int(row * base_h) - remaining_overlap_h
                x_pos = int(col * base_w) - remaining_overlap_w
                
                # Create 2D weight mask for this tile
                mask_h = self.create_blend_mask(cropped_h, fade_h, blend_mode)
                mask_w = self.create_blend_mask(cropped_w, fade_w, blend_mode)
                
                # Handle edge tiles (no fade on outer edges)
                if row == 0:
                    mask_h[:remaining_overlap_h + fade_h] = 1.0
                if row == grid_n - 1:
                    mask_h[-(remaining_overlap_h + fade_h):] = 1.0
                if col == 0:
                    mask_w[:remaining_overlap_w + fade_w] = 1.0
                if col == grid_n - 1:
                    mask_w[-(remaining_overlap_w + fade_w):] = 1.0
                
                # Create 2D mask
                weight_mask = mask_h.unsqueeze(1) * mask_w.unsqueeze(0)
                weight_mask = weight_mask.unsqueeze(-1).to(tiles.device)
                
                # Calculate valid region for placement
                src_y_start = max(0, -y_pos)
                src_x_start = max(0, -x_pos)
                dst_y_start = max(0, y_pos)
                dst_x_start = max(0, x_pos)
                
                src_y_end = min(cropped_h, out_h - y_pos)
                src_x_end = min(cropped_w, out_w - x_pos)
                dst_y_end = min(out_h, y_pos + cropped_h)
                dst_x_end = min(out_w, x_pos + cropped_w)
                
                if src_y_end <= src_y_start or src_x_end <= src_x_start:
                    continue
                
                # Extract valid regions
                tile_region = cropped_tile[src_y_start:src_y_end, src_x_start:src_x_end, :]
                mask_region = weight_mask[src_y_start:src_y_end, src_x_start:src_x_end, :]
                
                # Accumulate weighted pixels
                output[batch_idx, dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] += \
                    tile_region * mask_region
                weight_sum[batch_idx, dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] += \
                    mask_region
        
        # Normalize by weights
        weight_sum = torch.clamp(weight_sum, min=1e-8)
        output = output / weight_sum
        
        # Crop to original (scaled) dimensions
        output = output[:, :out_h, :out_w, :]
        
        return (output,)


class TileInfoDisplay:
    """
    Utility node to display tile information for debugging.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_info": ("TILE_INFO",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info_text",)
    FUNCTION = "display_info"
    CATEGORY = "image/tile"
    OUTPUT_NODE = True
    
    def display_info(self, tile_info: dict) -> Tuple[str]:
        info_lines = [
            f"Grid: {tile_info['grid_n']}x{tile_info['grid_n']}",
            f"Original Size: {tile_info['original_w']}x{tile_info['original_h']}",
            f"Tile Size: {tile_info['tile_w']}x{tile_info['tile_h']}",
            f"Base Tile: {tile_info['base_tile_w']}x{tile_info['base_tile_h']}",
            f"Overlap: {tile_info['overlap_w']}x{tile_info['overlap_h']} px",
            f"Overlap %: {tile_info['overlap_percent']:.1f}%",
            f"Batch Size: {tile_info['batch_size']}",
            f"Total Tiles: {tile_info['grid_n'] ** 2}",
        ]
        return ("\n".join(info_lines),)

class FilenameExtractor:
    """
    A ComfyUI node that extracts the filename without extension from a file path.
    Example: D:\folder\file.jpg -> file
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "extract_filename"
    CATEGORY = "utils"

    def extract_filename(self, file_path):
        """
        Extract filename without extension from a file path.
        
        Args:
            file_path: Full file path string
            
        Returns:
            Tuple containing the filename without extension
        """
        # Get the base filename with extension
        basename = os.path.basename(file_path)
        
        # Remove the extension
        filename_without_ext = os.path.splitext(basename)[0]
        
        return (filename_without_ext,)




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
    "LoadImageWithFilename": LoadImageWithFilename,
    "DiagonalTextWatermark": DiagonalTextWatermark,
    "ImageTileSplit": ImageTileSplit,
    "ImageTileMerge": ImageTileMerge,
    "TileInfoDisplay": TileInfoDisplay,
    "FilenameExtractor": FilenameExtractor,
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
    "LoadImageWithFilename": "Load Image (with filename)",
    "DiagonalTextWatermark": "Diagonal Text Watermark",
    "ImageTileSplit": "Image Tile Split",
    "ImageTileMerge": "Image Tile Merge", 
    "TileInfoDisplay": "Tile Info Display",
    "FilenameExtractor": "Extract Filename",
}
