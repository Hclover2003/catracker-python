import unittest
import numpy as np
import h5py
from catracker.importer import *

"""
test_import.py: A module for testing the import module.

This module provides tests for the functions in the import module.
"""

class TestImport(unittest.TestCase):
    def test_convert_to_hdf5(self):
        # Test converting an avi file
        avi_file = "/path/to/video.avi"
        converted_file = convert_to_hdf5(avi_file)
        self.assertEqual(converted_file, "/path/to/video.hdf5")

        # Test converting a tiff file
        tiff_file = "/path/to/image.tiff"
        converted_file = convert_to_hdf5(tiff_file)
        self.assertEqual(converted_file, "/path/to/image.hdf5")

        # Test converting an nd2 file
        nd2_file = "/path/to/image.nd2"
        converted_file = convert_to_hdf5(nd2_file)
        self.assertEqual(converted_file, "/path/to/image.hdf5")

        # Test converting an npy file
        npy_file = "/path/to/data.npy"
        converted_file = convert_to_hdf5(npy_file)
        self.assertEqual(converted_file, "/path/to/data.hdf5")

        # Test converting a folder
        folder_path = "/path/to/images"
        converted_file = convert_to_hdf5(folder_path, folder=True)
        self.assertEqual(converted_file, "/path/to/images.hdf5")

    def test_create_hdf5(self):
        output_filepath = "/path/to/output.hdf5"
        dataset_name = "data"
        data = np.random.rand(10, 10)
        create_hdf5(output_filepath, dataset_name, data)

        with h5py.File(output_filepath, 'r') as f:
            self.assertTrue(dataset_name in f)
            self.assertTrue(np.array_equal(data, f[dataset_name][:]))

    def test_convert_to_grayscale(self):
        data = np.random.rand(10, 10)
        grayscale_data = convert_to_grayscale(data)
        self.assertEqual(grayscale_data.shape, data.shape)
        self.assertEqual(grayscale_data.dtype, np.uint8)

    def test_video_capture_import(self):
        video_file = "/path/to/video.avi"
        output_filepath = "/path/to/output.hdf5"
        video_capture_import(video_file, output_filepath)

        with h5py.File(output_filepath, 'r') as f:
            self.assertTrue("video" in f)
            self.assertEqual(f["video"].shape[0], 100)  # Assuming 100 frames in the video

    def test_tiff_file_import(self):
        tiff_file = "/path/to/image.tiff"
        output_filepath = "/path/to/output.hdf5"
        tiff_file_import(tiff_file, output_filepath)

        with h5py.File(output_filepath, 'r') as f:
            self.assertTrue("image" in f)
            self.assertEqual(f["image"].shape, (1, 512, 512))  # Assuming the image is 512x512

    def test_nd2_file_import(self):
        nd2_file = "/path/to/image.nd2"
        output_filepath = "/path/to/output.hdf5"
        nd2_file_import(nd2_file, output_filepath)

        with h5py.File(output_filepath, 'r') as f:
            self.assertTrue("image" in f)
            self.assertEqual(f["image"].shape, (10, 512, 512))  # Assuming 10 frames in the nd2 file

    def test_npy_file_import(self):
        npy_file = "/path/to/data.npy"
        output_filepath = "/path/to/output.hdf5"
        npy_file_import(npy_file, output_filepath)

        with h5py.File(output_filepath, 'r') as f:
            self.assertTrue("data" in f)
            self.assertEqual(f["data"].shape, (10, 10))  # Assuming the data is a 10x10 array

    def test_folder_to_hdf5(self):
        folder_path = "/path/to/images"
        output_hdf5_path = "/path/to/output.hdf5"
        folder_to_hdf5(folder_path, output_hdf5_path)

        with h5py.File(output_hdf5_path, 'r') as f:
            self.assertTrue("images" in f)
            self.assertEqual(f["images"].shape[0], 10)  # Assuming 10 images in the folder

    def test_get_number_of_frames(self):
        hdf5_path = "/path/to/data.hdf5"
        num_frames = get_number_of_frames(hdf5_path)
        self.assertEqual(num_frames, 100)  # Assuming 100 frames in the dataset

    def test_get_index_dataset_name(self):
        hdf5_path = "/path/to/data.hdf5"
        dataset_name = get_index_dataset_name(hdf5_path, 0)
        self.assertEqual(dataset_name, "data")

if __name__ == "__main__":
    unittest.main()