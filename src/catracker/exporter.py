import pandas as pd
import numpy as np
from threshold import *

""" export.py: A module for generating results dataframes.

This module provides a function to generate a dataframe with the results of the tracking and object detection algorithms. 
"""

def generateResultsDataframe(self,red_xy_data:list,green_xy_data:list,red_object_data:list,green_object_data:list,red_img_data:list,green_img_data:list,file_name:str,match_distance_threshold:int=10):
    """ Takes in the data from the tracking and object detection algorithms and generates a dataframe with the results
    
    Args:
        red_xy_data (list): list of tuples with the x and y coordinates of the red neurons
        green_xy_data (list): list of tuples with the x and y coordinates of the green neurons
        red_object_data (list): list of tuples with the centroids, contours and areas of the red neurons
        green_object_data (list): list of tuples with the centroids, contours and areas of the green neurons
        red_img_data (list): list of red images
        green_img_data (list): list of green images
        file_name (str): name of the file to save the results
        match_distance_threshold (int): maximum distance between a centroid and a coordinate to consider them a match
        
    Returns:
        DataFrame: dataframe with 4n columns, where n is the number of neurons. Each column contains the following information:
            N{i}_RFP_coordinates: x and y coordinates of the red neuron i
            N{i}_RFP_intensity: mean intensity of the red neuron i
            N{i}_GFP_coordinates: x and y coordinates of the green neuron i
            N{i}_GFP_intensity: mean intensity of the green neuron i
        list: list of tuples with the neurons that were not matched (shape: (neuron, frame))
    """
    num_frames = self.ImageData.get_number_frames()
    num_neurons = len(self.neurons)
    green_unmatched = []
    
    # initialize empty dataframe to store results
    df_column_names = []
    for i in range(num_neurons):
        df_column_names.extend([f"N{i+1}_RFP_coordinates", 
                                f"N{i+1}_RFP_intensity", 
                                f"N{i+1}_GFP_coordinates", 
                                f"N{i+1}_GFP_intensity"])
    self.save_df = pd.DataFrame(columns = df_column_names, index = np.arange(0, num_frames, 1))
    
    # loop through frames and find matching centroids for each neuron
    for frame in range(num_frames):
        for n, neuron in enumerate(self.neurons):
            red_x, red_y = red_xy_data[n][frame]
            
            # skip: no neuron detected in this frame
            if red_x==0 and red_y == 0:
                self.save_df.at[frame, f"N{n+1}_RFP_coordinates"] = "nan"
                self.save_df.at[frame, f"N{n+1}_RFP_intensity"] = "nan"
                self.save_df.at[frame, f"N{n+1}_GFP_coordinates"] = "nan"
                self.save_df.at[frame, f"N{n+1}_GFP_intensity"] = "nan"
                continue
            
            # generate results for red data
            red_centroids, red_contours, _ = red_object_data[frame]
            closest_centroid_index, dist = find_closest_centroid_idx(red_x,red_y,red_centroids[0])
            if dist <= match_distance_threshold:
                red_contour = red_contours[0][closest_centroid_index]
                red_intensity = find_mean_contour_intensity(red_contour, red_img_data[frame])
            else:
                print("NO MATCHING RFP CONTOUR FOUND FOR NEURON", n, "FRAME", frame, "WITH CLOSEST CENTROID:", dist)
                red_contour = "nan"
                red_intensity = "nan"
                
            # generate results for green data
            green_x, green_y = green_xy_data[n][frame] #exists because for every red coordinate there is a green coordinate
            green_centroids, green_contours, _ = green_object_data[frame]
            closest_centroid_index, dist = find_closest_centroid_idx(green_x,green_y,green_centroids[0])
            if dist <= match_distance_threshold:
                green_contour = green_contours[0][closest_centroid_index]
                green_intensity = find_mean_contour_intensity(green_contour, green_img_data[frame])
            else:
                print("NO MATCHING GFP CONTOUR FOUND FOR NEURON", n, "FRAME", frame, "WITH CLOSEST CENTROID:", dist)
                green_contour = "nan"
                green_intensity = "nan"
                green_unmatched.append((n,frame))
                
            # add to dataframe
            self.save_df.at[frame, f"N{n+1}_RFP_coordinates"] = (red_x, red_y)
            self.save_df.at[frame, f"N{n+1}_RFP_intensity"] = red_intensity
            self.save_df.at[frame, f"N{n+1}_GFP_coordinates"] = (green_x, green_y)
            self.save_df.at[frame, f"N{n+1}_GFP_intensity"] = green_intensity
                
    # return dataframe and unmatched neurons
    return self.save_df, green_unmatched