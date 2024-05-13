import os
import re
import numpy as np
import h5py
import cv2  # For avi files
import nd2reader  # For nd2 files
import tifffile
from PIL import Image  # For tiff files
from pathlib import Path

""" import.py: A module for importing various file types and converting them to HDF5 format.

This module provides functions to convert various file types to HDF5 format and save them to disk.
The supported file types are .avi, .tiff, .tif, .nd2, and .npy.
"""

def convert_to_hdf5(file_path: str, folder: str = False) -> str:
        """
        Converts given filepath to hdf5 filepath

        Args:
            file_path (String): local filepath
            folder (bool, optional): whether the given filepath corresponds to a folder/directory.
            
        Returns:
            local filepath of the converted hdf5 file
            
        Raises:
            Exception: Unsupported file type. Please try again
        """
        
        fp = Path(file_path)
        
        # check if file is already an .hdf5 or .h5
        if fp.suffix == '.hdf5' or fp.suffix == '.h5':
            return file_path
        
        # if file is a folder, convert all images in folder to an hdf5
        if folder:
            output_hdf5_path = f"{fp}.hdf5"
            folder_to_hdf5(file_path, output_hdf5_path)
            return output_hdf5_path
        
        # check for other common extensions
        if fp.suffix == '.avi':
            output_hdf5_path = f"{fp}.hdf5"
            video_capture_import(fp, output_hdf5_path)
            return output_hdf5_path
        elif fp.suffix == '.tiff' or fp.suffix == ".tif":
            output_hdf5_path = f"{fp}.hdf5"
            tiff_file_import(fp, output_hdf5_path)
            return output_hdf5_path
        elif fp.suffix == '.nd2':
            output_hdf5_path = f"{fp}.hdf5"
            nd2_file_import(fp, output_hdf5_path)
            return output_hdf5_path
        elif fp.suffix == ".npy":
            output_hdf5_path = f"{fp}.hdf5"
            npy_file_import(fp, output_hdf5_path)
            return output_hdf5_path
        else:
            raise Exception("Unsupported file type. Please try again")   
        
def create_hdf5(output_filepath: str, dataset_name: str, data: np.ndarray) -> None:
    """
    Creates an HDF5 file and saves the data in a dataset.
    
    Args:
        output_filepath (str): Path to the HDF5 file.
        dataset_name (str): Name of the dataset to create or append.
        data (np.ndarray): Data to save in the dataset.
    """
    with h5py.File(output_filepath, 'w') as f:
        f.create_dataset(dataset_name, data=data, compression="gzip")
        
def convert_to_grayscale(data: np.ndarray) -> np.ndarray:
    """
    Normalizes and converts the image data to uint8
    
    Args:
        data (np.ndarray): Image data to convert.
    """
    return ((data / data.max()) * 255).astype(np.uint8)

def video_capture_import(fp: str, output_filepath: str) -> None:
    """
    Import an avi file and save it into an HDF5 file.
    
    Args:
        fp (Path): Path to the video file.
        output_filepath (str): Path to the output HDF5 file.
    """
    cap = cv2.VideoCapture(str(fp))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    data = np.stack(frames, axis=0)
    data = convert_to_grayscale(frames)
    create_hdf5(output_filepath, fp.stem, data)

def tiff_file_import(fp: str, output_filepath: str) -> None:
    """
    Import a tiff file and save it into an HDF5 file.
    
    Args:
        fp (Path): Path to the tiff file.
        output_filepath (str): Path to the output HDF5 file.
    """
    data = tifffile.imread(fp)
    data = convert_to_grayscale(data)
    create_hdf5(output_filepath, fp.stem, data)

def nd2_file_import(fp: str, output_filepath: str) -> None:
    """
    Import an nd2 file and save it into an HDF5 file.
    
    Args:
        fp (Path): Path to the nd2 file.
        output_filepath (str): Path to the output HDF5 file.
    """
    with nd2reader.ND2Reader(fp) as images:
        data = np.stack([img for img in images], axis=0)
    data = convert_to_grayscale(data)
    create_hdf5(output_filepath, fp.stem, data)

def npy_file_import(fp, output_filepath):
    """ 
    Import an npy file and save it into an HDF5 file.
    
    Args:
        fp (Path): Path to the npy file.
        output_filepath (str): Path to the output HDF5 file.
    """
    data = np.load(fp)
    data = convert_to_grayscale(data)
    create_hdf5(output_filepath, fp.stem, data)

def extract_number(filename: str) -> int:
    """ 
    Extracts the number from a filename (used for sorting files by number)
    
    Args:
        filename (str): Name of the file.
    
    Returns:
        int: Number extracted from the filename.
    """
    number = re.findall(r'\d+', str(filename))
    return int(number[-1]) if number else 0
    
def folder_to_hdf5(dir_path: str, output_hdf5_path: str)-> None:
    """ 
    Import all image files in a directory and save them into an HDF5 file.
    
    Args:
        dir_path (Path): Path to the directory containing the image files.
        output_hdf5_path (str): Path to the output HDF5 file.
    """
    fp = Path(dir_path)
    print("Uploading...")

    # List all files in the directory
    all_files = [fp / file for file in os.listdir(dir_path) if file.is_file()]

    # Filter image files
    image_files = [file for file in all_files if file.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')]

    if not image_files:
        print(f"No file chosen or no valid image files found in the directory: {fp}. Images should have .jpg, .jpeg, .png, .bmp extensions.")
        return None

    # Sort files based on integers if present, otherwise by filename
    sorted_files = sorted(image_files, key=extract_number)

    with h5py.File(output_hdf5_path, 'w') as hdf5_file:
        for i, img_path in enumerate(sorted_files):
            # Load image
            img = np.array(Image.open(img_path))
            # Initialize dataset or resize if it already exists
            if i == 0:
                maxshape = (None,) + img.shape
                dataset = hdf5_file.create_dataset("images", data=img[np.newaxis, ...], maxshape=maxshape, chunks=True, compression="gzip")
            else:
                dataset.resize(dataset.shape[0] + 1, axis=0)
                dataset[-1] = img

def get_number_of_frames(hdf5_path: str) -> int:
    """
    Returns the number of frames in the specified dataset within an HDF5 file.

    Args:
        hdf5_path (str): Path to the HDF5 file.

    Returns:
        int: Number of frames in the dataset.
    """
    # Using get_index_dataset_name to get the name of the first dataset
    first_dataset_name = get_index_dataset_name(hdf5_path, 0)
    
    with h5py.File(hdf5_path, 'r') as file:
        # Access the dataset by name to ensure we don't load it into memory
        dataset = file[first_dataset_name]
        num_frames = dataset.shape[0]  # Assuming the first dimension represents frames
        return num_frames
    
def get_index_dataset_name(file_path: str, index: int = 0) -> str:
    """
    Get the name of the [index] dataset in the HDF5 file.
    
    Args:
        file_path (str): Path to the HDF5 file.
        index (int): Index of the dataset to get the name of.
    """
    with h5py.File(file_path, 'r') as file:
        index_dataset_name = list(file.keys())[index]  # Get the dataset name at index
        return index_dataset_name