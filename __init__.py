"""
Praveen's ComfyUI Custom Nodes
A collection of utility nodes for image manipulation, tiling, and batch processing.
"""

from .Praveen_tools import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Package metadata
__version__ = "1.0.0"
__author__ = "Praveen"
__description__ = "Custom nodes for ComfyUI focusing on image manipulation, tiling, and workflow optimization"

# Web directory for any future UI components
WEB_DIRECTORY = "./web"
