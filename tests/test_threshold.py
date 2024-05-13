import unittest
import numpy as np
from catracker.threshold import *

""" 
test_threshold.py: A module for testing the threshold module.

This module provides tests for the functions in the threshold module.
"""

class TestThreshold(unittest.TestCase):
    
    def setUp(self):
        """ Set up variables for testing"""
        self.image_data = np.random.randint(0, 255, size=(100, 100)).astype(np.uint8)
        self.coords_list = [(10, 10), (20, 20), (30, 30)]
        self.crop_x = 20
        self.crop_y = 20
        self.gfp_box = np.random.randint(0, 255, size=(self.crop_y, self.crop_x)).astype(np.uint8)
        self.rfp_contour = np.array([[10, 10], [20, 20], [30, 30]])
        self.gfp_threshold_params = (5, 100, 100)
    
    def test_find_mean_contour_intensity(self):
        """Test find_mean_contour_intensity function"""
        contour = np.array([[10, 10], [20, 20], [30, 30]])
        mean_intensity = find_mean_contour_intensity(contour, self.image_data)
        self.assertIsInstance(mean_intensity, float)
    
    def test_return_bounding_boxes(self):
        """Test return_bounding_boxes function"""
        bounding_boxes = return_bounding_boxes(self.image_data, self.coords_list, self.crop_x, self.crop_y)
        self.assertIsInstance(bounding_boxes, np.ndarray)
        self.assertEqual(bounding_boxes.shape, (len(self.coords_list), 402))
    
    def test_preprocess_img(self):
        """ Test preprocess_img function"""
        processed_img = preprocess_img(self.image_data)
        self.assertIsInstance(processed_img, np.ndarray)
        self.assertEqual(processed_img.shape, self.image_data.shape)
    
    def test_crop_img(self):
        """ Test crop_img function"""
        cropped_img = crop_img(self.image_data, (50, 50), self.crop_x, self.crop_y)
        self.assertIsInstance(cropped_img, np.ndarray)
        self.assertEqual(cropped_img.shape, (self.crop_y, self.crop_x))
    
    def test_color_blobs(self):
        """ Test color_blobs function"""
        new_img, centroids, contours, num_contours = color_blobs(self.image_data, 5, 100, 100)
        self.assertIsInstance(new_img, np.ndarray)
        self.assertIsInstance(centroids, list)
        self.assertIsInstance(contours, list)
        self.assertIsInstance(num_contours, int)
    
    def test_get_blobs_adaptive(self):
        """ Test get_blobs_adaptive function"""
        thresh, new_img = get_blobs_adaptive(self.image_data, 5, 100, 100)
        self.assertIsInstance(thresh, np.ndarray)
        self.assertIsInstance(new_img, np.ndarray)
    
    def test_find_centroids(self):
        """ Test find_centroids function"""
        centroids, contours = find_centroids(self.image_data)
        self.assertIsInstance(centroids, list)
        self.assertIsInstance(contours, list)
    
    def test_distance(self):
        """ Test distance function"""
        dist = distance((0, 0), (3, 4))
        self.assertIsInstance(dist, float)
    
    def test_find_closest_centroid_idx(self):
        """ Test find_closest_centroid_idx function"""
        closest_centroid_idx, min_distance = find_closest_centroid_idx(0, 0, [(1, 1), (2, 2), (3, 3)])
        self.assertIsInstance(closest_centroid_idx, int)
        self.assertIsInstance(min_distance, float)
    
    def test_match_closest_contours(self):
        """ Test match_closest_contours function"""
        coordinates = np.array([(0, 0), (1, 1), (2, 2)])
        centroids = [(1, 1), (2, 2), (3, 3)]
        matches = match_closest_contours(coordinates, centroids)
        self.assertIsInstance(matches, list)
    
    def test_calculate_average_centroid_distance(self):
        """ Test calculate_average_centroid_distance function"""
        red_object_data = [[[(1, 1), (2, 2)], []], [[(3, 3), (4, 4)], []]]
        num_frames = 2
        avg_distance = calculate_average_centroid_distance(red_object_data, num_frames)
        self.assertIsInstance(avg_distance, float)
    
    def test_find_matching_contour(self):
        """ Test find_matching_contour function"""
        GFP_contour = find_matching_contour(self.gfp_box, self.rfp_contour, self.gfp_threshold_params)
        self.assertIsInstance(GFP_contour, np.ndarray)
    
if __name__ == '__main__':
    unittest.main()