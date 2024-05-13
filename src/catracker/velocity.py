import re
import cv2
import numpy as np
from typing import Tuple

"""velocity.py: A module for calculating the velocity of a worm from its position data.

This module provides functions for calculating the velocity of a worm from its position data.
"""

def extract_positions(string_label: str, pattern: str=r'xpos=(-?\d+\.\d+),ypos=(-?\d+\.\d+)') -> Tuple[float, float]:
    """
    Extract the x and y positions from a string label

    Args:
        string_label : string label from the stage position data (x, y positions are in this string)
        pattern : regular expression pattern to match the x and y positions

    Returns:
        xpos : x position
        ypos : y position
    """
    match = re.search(pattern, string_label)

    if match:
        xpos = float(match.group(1))
        ypos = float(match.group(2))
        return xpos, ypos
    else:
        print("No match found")


def array_to_video(arr, output_avi_path: str = "output.avi"):
    """
    Converts a 3D NumPy array to an AVI video file.

    Args:
        arr : ndarray of type np.uint8 from 0-255 (frames, height, width)
        output_avi_path : path to output AVI video file
    """
    num_frames, height, width = arr.shape

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = 30  # Adjust the frames per second as needed
    out = cv2.VideoWriter(output_avi_path, fourcc, fps, (width, height))

    # Write each frame to the AVI video
    for frame_idx in range(num_frames):
        frame = arr[frame_idx].astype("uint8")
        # Convert grayscale to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)

    # Release the VideoWriter
    out.release()


def exponential_moving_average(data: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute the exponential moving average of a 1D array.

    Args:
        data : array of shape (num_frames,)
        alpha : smoothing factor (0 < alpha < 1, smaller alpha = more smoothing)
    """
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


def calculate_forward_reverse_signs(abs_anterior_positions: np.ndarray, abs_posterior_positions: np.ndarray) -> np.ndarray:
    """ 
    Calculate the forward/reverse signs for each frame of the video.

    Args:
        abs_anterior_positions : absolute positions (x, y) of the anterior neuron of the worm (shape: (num_frames, 2))
        abs_posterior_positions : absolute positions (x, y) of the posterior neuron of the worm (shape: (num_frames, 2))

    Returns:
        forward_reverse_sign : 1 if the worm is moving forward, -1 if the worm is moving backward (shape: (num_frames,))
    """
    # AP axis direction: anterior - posterior
    ap_direction_vectors = (abs_anterior_positions -
                            abs_posterior_positions)[1:]
    anterior_direction_vectors = np.diff(abs_anterior_positions, axis=0)
    ap_angles = np.arctan2(
        ap_direction_vectors[:, 1], ap_direction_vectors[:, 0])
    anterior_angles = np.arctan2(
        anterior_direction_vectors[:, 1], anterior_direction_vectors[:, 0])

    # Calculate the angle differences and wrap them to [-pi, pi]
    angle_differences = abs(ap_angles - anterior_angles)
    angle_differences = np.degrees(angle_differences)
    angle_differences = np.where(
        angle_differences > 180, 360 - angle_differences, angle_differences)

    # Create an array where the value is 1 if the angle difference is within 90 degrees (pi/2 radians) and -1 otherwise
    forward_reverse_signs = np.where(angle_differences <= 90, 1, -1)
    return forward_reverse_signs


def calculate_displacements(neuron_abs_positions: np.ndarray) -> np.ndarray:
    """ 
    Calculate the magnitude of the displacements between each frame of the video.
    
    Args:
        neuron_abs_positions : absolute positions (x, y) of the neuron of the worm (shape: (num_frames, 2))
    """
    neuron_displacements = np.diff(neuron_abs_positions, axis=0)
    euclidean_distances = np.linalg.norm(neuron_displacements, axis=1)
    return euclidean_distances


def get_abs_positions(anterior_positions: np.ndarray, posterior_positions: np.ndarray, stage_positions: np.ndarray, scale_factor: float) -> Tuple[np.ndarray, np.ndarray]:
    """ 
    Get the absolute positions of the anterior and posterior neurons of the worm from the raw positions and stage positions.

    Args:
        anterior_positions : raw positions (x, y) of the anterior neuron of the worm (shape: (num_frames, 2))
        posterior_positions : raw positions (x, y) of the posterior neuron of the worm (shape: (num_frames, 2))
        stage_positions : raw positions (x, y) of the stage (shape: (num_frames, 2))
        scale_factor : scale factor for the neuron positions (ie. 1 pixel = 2.75 micrometers)
    
    Returns:
        abs_anterior_positions : absolute positions (x, y) of the anterior neuron of the worm (shape: (num_frames, 2))
        abs_posterior_positions : absolute positions (x, y) of the posterior neuron of the worm (shape: (num_frames, 2))
    """
    abs_anterior_positions = (
        anterior_positions * scale_factor) - stage_positions[:, :2]
    abs_posterior_positions = (
        posterior_positions * scale_factor) - stage_positions[:, :2]
    return abs_anterior_positions, abs_posterior_positions


def calculate_velocities(anterior_positions: np.ndarray, posterior_positions: np.ndarray, stage_positions: np.ndarray, scale_factor: float):
    """
    Calculate the velocities of the worm from the absolute positions of the anterior and posterior neurons.

    Args:
        abs_anterior_positions : absolute positions (x, y) of the anterior neuron of the worm (shape: (num_frames, 2))
        abs_posterior_positions : absolute positions (x, y) of the posterior neuron of the worm (shape: (num_frames, 2))

    Returns:
        forward_reverse_signs : forward/reverse signs for each frame (1 if the worm is moving forward, -1 if the worm is moving backward) (shape: (num_frames,))
        velocities : velocities of the worm (shape: (num_frames,))
    """
    abs_anterior_positions, abs_posterior_positions = get_abs_positions(anterior_positions, posterior_positions,
                                                                        stage_positions, scale_factor)
    displacements = calculate_displacements(abs_anterior_positions)
    forward_reverse_signs = calculate_forward_reverse_signs(
        abs_anterior_positions, abs_posterior_positions)
    velocities = forward_reverse_signs * displacements
    return forward_reverse_signs, velocities

