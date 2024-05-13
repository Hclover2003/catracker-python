import numpy as np
import cv2

""" 
alignment.py: A module for image alignment and transformation

This module provides functions for adding padding to images, translating images, rotating images, and applying transformations to images.
It is used in the image alignment step of the neuron tracking algorithm.
"""

def add_padding(data: np.ndarray, pad_width: int, pad_height: int) -> np.ndarray:
    """ Adds padding to a 2D image of shape (height, width, channels)
    
    Args:
        data (np.ndarray): Input image (height, width, channels)
        pad_width (int): Width of the padding (new width = width + 2 * pad_width)
        pad_height (int): Height of the padding (new height = height + 2 * pad_height)
    
    Returns:
        np.ndarray: Padded image (new_height, new_width, channels)
    """
    if len(data.shape) < 3 or data.shape[2] not in [1, 3, 4]:
        raise ValueError("Invalid image dimensions")

    height, width, channels = data.shape
    new_height = height + 2 * pad_height
    new_width = width + 2 * pad_width

    # Create an array with zeros and shape (new_height, new_width, channels)
    padded_image = np.ones((new_height, new_width, channels), dtype=data.dtype)

    # Copy the original image data into the center of the padded image
    padded_image[pad_height:pad_height + height, pad_width:pad_width + width] = data

    return padded_image

def get_border_value(image: np.ndarray) -> int:
    """ Get the border value for the image based on the number of channels
    
    The border value of an image is the value used to fill the border when applying transformations like translation or rotation
    
    Args:
        image (np.ndarray): Input image
    
    Returns:
        int: Border value for the image
    """
    # Check the number of channels in the image and set border value (0s) accordingly
    if len(image.shape) == 2:
        # Grayscale image
        return 0
    elif len(image.shape) == 3:
        # Color image (RGB or RGBA)
        return (0, 0, 0)[:image.shape[2]]  # Handles both RGB and RGBA
    else:
        raise ValueError("Unsupported image format")

def translate_image(image: np.ndarray, tx: int, ty: int) -> np.ndarray:
    """
    Translate an image by a given amount in the x and y directions
    
    Args:
        image (np.ndarray): Input image
        tx (int): Translation in the x-direction
        ty (int): Translation in the y-direction
    
    Returns:
        np.ndarray: Translated image
    """
    border_value = get_border_value(image)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    result = cv2.warpAffine(image, M, image.shape[1::-1], borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
    return result

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """ 
    Rotate an image by a given angle
    
    Args:
        image (np.ndarray): Input image
        angle (float): Rotation angle in degrees
        
    Returns:
        np.ndarray: Rotated image
    """
    border_value = get_border_value(image)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
    return result

def apply_transformation(image: np.ndarray, tx: int, ty: int, rotate: int) -> np.ndarray:
    """ Apply translation and rotation to an image
    
    Args:
        image (np.ndarray): Input image
        tx (int): Translation in the x-direction
        ty (int): Translation in the y-direction
        rotate (int): Rotation angle
    
    Returns:
        np.ndarray: Transformed image
    """
    # Apply translation
    transformed_image = translate_image(image, tx, ty)

    # Apply rotation
    if rotate != 0:
        transformed_image = rotate_image(transformed_image, rotate)

    return transformed_image
        