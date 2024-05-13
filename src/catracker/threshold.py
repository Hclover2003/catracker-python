import numpy as np
import cv2
import random
import time
from typing import Tuple, List, Optional
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist

"""
threshold.py: A module for thresholding and blob detection in images.

This module provides functions for thresholding images, detecting blobs, and finding centroids in images.
"""

def find_mean_contour_intensity(contour : List, image : np.ndarray) -> float:
    """Compute the mean intensity of a contour within a given image.

    Args:
        contour : list of np.ndarray
            A list of (x, y) coordinates that define the contour.
        image : ndarray
            The 2D image (e.g., grayscale) in which to compute the mean intensity of the contour.
            shape (height, width)

    Returns:
        float
            The mean intensity value within the specified contour in the given image.
    """
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.drawContours(mask, contour, -1, (255,255,255), thickness = cv2.FILLED)
    return np.mean(image[mask==255])

def return_bounding_boxes(image_data: np.array, coords_list: List[Tuple[int, int]], crop_x: int = 20, crop_y: int = 20) -> np.array:
    """Crop images from the original image at specified coordinates and flatten them.

    Args:
        image_data : np.array
            The original image data.
        coords_list : list of tuple
            List of x, y coordinates at which to crop the image.
        crop_x : int
            Width of the cropped images.
        crop_y : int
            Height of the cropped images.

    Returns:
        np.array
            Array of flattened cropped images with their coordinates, 
            each row represents a cropped image. Has shape (len(coords_list), 402)
    """
    data = np.zeros((len(coords_list),402))
    for i, coords in enumerate(coords_list):
        x,y = coords
        data[i][0] = x
        data[i][1] = y
        data[i][2:] = np.ndarray.flatten(preprocess_img(crop_img(image_data, [x,y], crop_x, crop_y)))
    return data #shape ((len(coords_list), 402))

def preprocess_img(img: np.array, weights: tuple = (0.299, 0.587, 0.114)) -> np.array:
    """Convert image to grayscale and normalize to uint8. Supports different data types and custom weights.

    Args:
        img : np.array
            Image to convert.
        weights : tuple, optional
            Weights for the RGB channels when converting to grayscale (default is (0.299, 0.587, 0.114)).

    Returns:
        np.array
            Converted image.
    """
    
    #Normalization/conversion to uint8
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img/np.max(img) * 255).astype(np.uint8)  # If the image is float, normalize it to [0,1] and scale to 255

    elif img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)  # If the image is 16-bit, scale it down to 8-bit

    # Convert to grayscale using custom weights if it's an RGB or RGBA image
    if len(img.shape) < 3: # If the image is already grayscale, do nothing
        img = img[:, :, np.newaxis]
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        b, g, r = cv2.split(img)
        img = (weights[0] * r + weights[1] * g + weights[2] * b).astype(np.uint8)

    elif img.shape[2] == 3:  # RGB
        b, g, r = cv2.split(img)
        img = (weights[0] * r + weights[1] * g + weights[2] * b).astype(np.uint8)

    return img

def crop_img(img: np.array, coordinate: Tuple[int, int], crop_width: int = 20, crop_height: int = 20) -> np.array:
    """ Crop given image centered around a specified point.

    Args:
        img : np.array
            Image to crop.
        coordinate : tuple of int
            Point to crop around.
        crop_width : int
            Width of crop.
        crop_height : int
            Height of crop.

    Returns:
        np.array
            Cropped image of shape (crop_width, crop_height).
    """
    x = round(coordinate[0])
    y = round(coordinate[1])
    cropped_img = img[max(0, y-(crop_height//2)): min(img.shape[0], y+(crop_height//2)),
                      max(0, x-(crop_width//2)): min(img.shape[1], x+(crop_width//2))]
    if cropped_img.shape != (crop_width, crop_height):
        cropped_img = cv2.resize(cropped_img, (crop_width, crop_height))
    return cropped_img               
       
def color_blobs(img: np.array, bound_size: int, min_brightness_const: int, min_area: int) -> Tuple[np.array, List[Tuple[int, int]], List[np.array], int]:
    """Detects blobs and finds their centroids in the image and colors them with distinctive colors.

    Args:
        img : np.array
            Input image.
        bound_size : int
            Parameter for adaptive thresholding.
        min_brightness_const : int
            Parameter for adaptive thresholding.
        min_area : int
            Minimum area of the detected blob.

    Returns:
        tuple
            new_img : np.array
                Output image where each blob is colored with a distinctive color.
            centroids : list of tuple
                Centroids of each blob in the image.
            contours : list of np.array
                Contours of each blob in the image.
            num_contours : int
                Number of detected blobs in the image.
    """
    
    # Make sure bound_size is odd (should be already)
    if bound_size % 2 == 0:
        bound_size = bound_size + 1

    # Get thresholded and new image from get_blobs_adaptive
    thresh, new_img = get_blobs_adaptive(preprocess_img(img), bound_size, min_brightness_const, min_area)

    # Find centroids and contours
    centroids, contours = find_centroids(new_img)

    # Generate distinctive colors for each centroid
    colors = [(random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)) for _ in centroids]

    # Draw contours and centroids on the new_img
    for i, c in enumerate(contours):
        color = colors[i]
        cv2.drawContours(new_img, [c], -1, color, -1)
        cv2.circle(new_img, centroids[i], 3, color, -1)

    return new_img, centroids, contours, len(contours)

def get_blobs_adaptive(img: np.array, bound_size: int, min_brightness_const: int, min_area: int) -> Tuple[np.array, np.array]:
    """
    Performs adaptive thresholding on the image to get blobs and return the thresholded image and a new image.

    Args:
        img : np.array
            Original image.
        bound_size : int
            Parameter for adaptive thresholding.
        min_brightness_const : int
            Parameter for adaptive thresholding.
        min_area : int
            Minimum area of the detected blob.

    Returns
    -------
    tuple
        thresh : np.array
            Thresholded image.
        new_img : np.array
            New image where the blobs are filled.
    """
    # "smoothing" the image with Gaussian Blur
    im_gauss = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(im_gauss, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, bound_size, (min_brightness_const))
    # Find contours
    cont, hierarchy = cv2.findContours(thresh,
                                       cv2.RETR_EXTERNAL,
                                       # only stores necessary points to define contour (avoids redundancy, saves memory)
                                       cv2.CHAIN_APPROX_SIMPLE
                                       )
    cont_filtered = []
    for con in cont:
        # calculate area, filter for contours above certain size
        area = cv2.contourArea(con)
        perimeter = cv2.arcLength(con, True)
        if area > min_area:  # chosen by trial/error
            cont_filtered.append(con)

    # Draw + fill contours
    new_img = np.full_like(img, 0)  # image has black background
    for c in cont_filtered:
        cv2.drawContours(new_img,  # image to draw on
                         [c],  # contours to draw
                         -1,  # contouridx: since negative, all contours are drawn
                         255,  # colour of contours: white
                         -1  # thickness: since negative, fill in the shape
                         )
    return thresh, new_img

def find_centroids(segmented_img: np.ndarray) -> Tuple[List[Tuple[int, int]], List[np.array]]:
    """
    Finds centroids and contours of segmented image.

    Args:
        segmented_img : a 2D image of shape (height, width) with values of 0 and 255.

    Returns:
        tuple : (centroids, contours)
            centroids : list of tuple
                List of centroid coordinates.
            contours : list of np.array
                List of contours.
    """
    centroids = []
    # get contours
    contours, hierarchy = cv2.findContours(segmented_img,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)

    # compute the centroid of each contour
    for c in contours:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroids.append((cX, cY))

    return centroids, contours    

def distance(point1, point2):
    """
    Computes the distance between two points.
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def find_closest_centroid_idx(x: int, y: int, centroids: List[Tuple[int, int]]) -> Tuple[int, float]:
    """
    Finds the index of the closest centroid from a list of centroids to a given point.
    
    A centroid is a tuple of (x, y) coordinates, representing the center of a blob.
    
    Args:
        x (int): x-coordinate of the point.
        y (int): y-coordinate of the point.
        centroids (List[Tuple[int, int]]): A list of centroid coordinates.
    
    Returns:
        Tuple[int, float]: The index of the closest centroid and the distance to that centroid.
    """
    min_distance = float('inf')
    closest_centroid_idx = None
    for idx, centroid in enumerate(centroids):
        dist = distance((x,y), centroid)
        if dist < min_distance:
            min_distance = dist
            closest_centroid_idx = idx
    return closest_centroid_idx, min_distance

def match_closest_contours(coordinates: np.ndarray, 
                           centroids: List[Tuple[int, int]], 
                           max_distance: float = np.inf) -> List[Optional[int]]:
    """
    Match each coordinate to the closest centroid, each centroid can only be matched once. 
    If the closest unmatched centroid is too far away, or there are no more centroids, return None for that coordinate.

    Args:
        coordinates (np.ndarray): An array of coordinates.
        centroids (List[Tuple[int, int]]): A list of centroid coordinates.
        max_distance (float): Maximum allowable distance for a valid match.

    Returns:
        List[Optional[int]]: A list of indices of the matched centroid or None.
    """
    coordinates = np.array(coordinates)
    centroids = np.array(centroids)
    print("COORDINATES", coordinates)
    print("CENTROIDS", centroids)
    
    num_coords = len(coordinates)
    num_centroids = len(centroids)

    if num_centroids == 0:
        return [None] * num_coords

    # Create a cost matrix using Euclidean distance
    cost_matrix = np.linalg.norm(coordinates[:, np.newaxis] - centroids, axis=2)

    # If there are fewer centroids than coordinates, add dummy rows with a large cost
    if num_coords > num_centroids:
        dummy_rows = np.full((num_coords - num_centroids, num_centroids), float('inf'))
        cost_matrix = np.vstack([cost_matrix, dummy_rows])

    # Apply the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Prepare the result
    matches = [None] * num_coords
    for i, j in zip(row_ind, col_ind):
        if i < num_coords and cost_matrix[i, j] <= max_distance:
            matches[i] = j
            
    print("MATCHES", matches)

    return matches

def calculate_average_centroid_distance(red_object_data, num_frames, sample_fraction=0.1):
    """
    Calculate the average distance between centroids over a sample of frames.

    Args:
        red_object_data: Data containing centroids for each frame.
        num_frames (int): Total number of frames.
        sample_fraction (float): Fraction of frames to sample for calculation.

    Returns:
        float: The average distance between centroids across the sampled frames.
    """
    # Determine the number of frames to sample
    num_sampled_frames = int(num_frames * sample_fraction)
    
    # Randomly select frames
    sampled_frame_indices = random.sample(range(num_frames), num_sampled_frames)

    # Initialize a list to store distances from each sampled frame
    all_frame_distances = []

    # Calculate distances in each sampled frame
    for frame_idx in sampled_frame_indices:
        try: #in case that frame was not tracked
            centroids = red_object_data[frame_idx][0]
            if len(centroids) > 1:
                # Calculate pairwise distances between centroids and add to the list
                distances = pdist(centroids)
                all_frame_distances.extend(distances)
        except IndexError:
            continue

    # Compute the overall average distance
    if all_frame_distances:
        avg_distance = np.mean(all_frame_distances)
    else:
        avg_distance = 100

    return avg_distance

def find_matching_contour(GFP_box : np.ndarray, 
                          RFP_contour : np.ndarray, 
                          GFP_threshold_params: Tuple[int, int, int]) -> np.ndarray:
    """Finds the GFP contour within GFP_box that best matches the RFP contour. 
    
    Only searches within +/- 10 of the threshold params.

    Parameters
    ----------
    GFP_box : np.array
        Image of GFP channel (Dimension: crop_y x crop_x).
        
    RFP_contour : np.array
        Corresponding contour in RFP channel.
        
    GFP_threshold_params : tuple
        Parameters for adaptive thresholding. (bound_size, min_brightness_const, min_area)
        
    Returns
    -------
    GFP_contour : np.array or None
    """
    
    RFP_contour_area = cv2.contourArea(RFP_contour)
    
    #create three values varying between +/- 10 for each of the given threshold parameters
    param1 = [max(3, GFP_threshold_params[0] - i) for i in (-10, 0, 10)] #boundsize
    param2 = [GFP_threshold_params[1] - 10, GFP_threshold_params[1], GFP_threshold_params[1] + 10] #minbrightness
    param3 = [max(0, GFP_threshold_params[2] - i) for i in (-10, 0, 10)] #minarea
    
    #run through all 3^3 combinations of threshold params and record the contour size for each
    total_time = 0
    contour_sizes = []
    corresp_contours = []
    
    for p1 in param1:
        for p2 in param2:
            for p3 in param3:
                # Get thresholded image
                start_time = time.time()
                thresh, new_img = get_blobs_adaptive(preprocess_img(GFP_box), p1, p2, p3)
                
                centroids, contours = find_centroids(new_img)
                if len(contours) >= 1:
                    largest_contour = max(contours, key=cv2.contourArea)
                    largest_contour_size = cv2.contourArea(largest_contour)
                else:
                    largest_contour = None
                    largest_contour_size = 0  

                # Append the size of the largest contour to the list
                contour_sizes.append(largest_contour_size)
                corresp_contours.append(largest_contour)

                elapsed_time = time.time() - start_time
                total_time += elapsed_time
    
    #print the total time taken (remove later)
    print("TIME TAKEN", total_time)
    
    #find the contour size closest to RFP_contour_area
    GFP_contour = corresp_contours[np.argmin(np.abs(np.array(contour_sizes) - RFP_contour_area))]
    
    #return the intensity of the contour with the best threshold params
    return GFP_contour #np.ndarray or None
    