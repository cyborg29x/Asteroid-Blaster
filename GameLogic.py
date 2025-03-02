import numpy as np
import ctypes
import time
from sdl3 import *

def pixel_alpha_channel_extraction(surface):
    SDL_LockSurface(surface)
    pixels_pointer = surface.contents.pixels
    pitch = surface.contents.pitch
    height = surface.contents.h
    width = surface.contents.w
    
    raw_data = ctypes.string_at(pixels_pointer, pitch * height)
    #print(raw_data)
    pixel_array = np.frombuffer(raw_data, dtype = Uint32).copy()
    #print(pixel_array)
    pixel_array = pixel_array.reshape((height, width))
    #print(pixel_array)
    alpha_channel_array = pixel_array & 0xFF > 0
    alpha_channel_array = np.where(alpha_channel_array)
    alpha_channel_array = np.column_stack((alpha_channel_array[1], alpha_channel_array[0]))
    alpha_channel_array = alpha_channel_array.astype(np.int32)
    SDL_UnlockSurface(surface)
    return alpha_channel_array

# Calculate the angle between the y-axis and the vector pointing from point1 to point2
def angle_y_axis_two_points(point1, point2):
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]
    return (
        180
        - np.arccos(delta_y / np.sqrt(delta_x**2 + delta_y**2)) * 180 / np.pi
        if delta_x > 0
        else 180
        + np.arccos(delta_y / np.sqrt(delta_x**2 + delta_y**2)) * 180 / np.pi
    )

def circle_collision(position_1, radius_1, position_2, radius_2):
    distance = np.sqrt((position_1[0] - position_2[0]) ** 2 + (position_1[1] - position_2[1]) ** 2)
    return distance < radius_1 + radius_2

def pixel_perfect_collision(object_1, object_2):
    # Get non-transparent pixel arrays
    pixel_array_1 = object_1.alpha_channel_array.astype(float).copy()
    pixel_array_2 = object_2.alpha_channel_array.astype(float).copy()
    #print(pixel_array_1)

    # Transform both arrays to global coordinates
    angle_1 = np.radians(object_1.angle)
    angle_2 = np.radians(object_2.angle)
    
    #print(angle_1, object_1.x_pos)
    #print(pixel_array_1[:, 0])
    pixel_array_1[:, 0] = pixel_array_1[:, 0] * np.cos(angle_1) + object_1.x_pos
    pixel_array_1[:, 1] = pixel_array_1[:, 1] * np.sin(angle_1) + object_1.y_pos
    
    pixel_array_2[:, 0] = pixel_array_2[:, 0] * np.cos(angle_2) + object_2.x_pos
    pixel_array_2[:, 1] = pixel_array_2[:, 1] * np.sin(angle_2) + object_2.y_pos
    
    # Merge arrays into one
    merged_array = np.vstack((pixel_array_1, pixel_array_2))
    #print(merged_array)
    
    # Check if inside radius of object 1
    object_1_position = np.array([object_1.x_pos, object_1.y_pos])
    distances_squared = (merged_array[:, 0] - object_1_position[0]) ** 2 + (merged_array[:, 1] - object_1_position[1]) ** 2
    merged_array = merged_array[distances_squared <= object_1.collision_radius ** 2]
    
    # Repeat for object 2
    object_2_position = np.array([object_2.x_pos, object_2.y_pos])
    distances_squared = (merged_array[:, 0] - object_2_position[0]) ** 2 + (merged_array[:, 1] - object_2_position[1]) ** 2
    merged_array = merged_array[distances_squared <= object_2.collision_radius ** 2]
    
    #print(merged_array)
    return merged_array

def coordinate_conversion(coordinates_array, translation_coordinate, angle = 0):
    # Make sure to copy instead of referencing
    coordinates_array = coordinates_array.astype(float).copy()
    translation_coordinate = translation_coordinate.astype(float).copy()
    
    # Transform angle from degrees to radians
    angle = np.radians(angle)
    
    # Apply coordinate conversion
    coordinates_array[:, 0] = coordinates_array[:, 0] * np.cos(angle) + translation_coordinate[0]
    coordinates_array[:, 1] = coordinates_array[:, 1] * np.sin(angle) + translation_coordinate[1]

    return coordinates_array
    
def collision_velocity_update(object_1, object_2, pixel_collision_array):
    # Ensure array is copied
    pixel_collision_array = pixel_collision_array.copy()
    
    # Determine the coordinates of the collision's centroid
    centroid_x_position = np.mean(pixel_collision_array[:, 0])
    centroid_y_position = np.mean(pixel_collision_array[:, 1])
    centroid = np.array([centroid_x_position, centroid_y_position])
    
    # Determine the 4 pixels nearest to the centroid
    # For object 1
    neighbours = 4
    alpha_channel_array = coordinate_conversion(object_1.alpha_channel_array,
                                                np.array([object_1.x_pos, object_1.y_pos]),
                                                object_1.angle).copy()
    distances = np.sum((alpha_channel_array - centroid) ** 2, axis = 1)
    closest_indices = np.argpartition(distances, neighbours)[:neighbours]
    neighbours_array_1 = alpha_channel_array[closest_indices]
    neighbours_center_1 = np.array([np.mean(neighbours_array_1[:, 0]), np.mean(neighbours_array_1[:, 1])])
    #print(neighbours_center_1)
    
    
    # For object 2
    alpha_channel_array = coordinate_conversion(object_2.alpha_channel_array,
                                                np.array([object_2.x_pos, object_2.y_pos]),
                                                object_2.angle).copy()
    distances = np.sum((alpha_channel_array - centroid) ** 2, axis = 1)
    closest_indices = np.argpartition(distances, neighbours)[:neighbours]
    neighbours_array_2 = alpha_channel_array[closest_indices]
    neighbours_center_2 = np.array([np.mean(neighbours_array_2[:, 0]), np.mean(neighbours_array_2[:, 1])])
    
    # Determine the line of action
    line_of_action_vector = neighbours_center_1 - neighbours_center_2
    line_of_action_vector = line_of_action_vector / np.linalg.norm(line_of_action_vector)
    #print(line_of_action_vector)
    angle_with_y_axis = angle_y_axis_two_points(neighbours_center_2, neighbours_center_1)
    
    # Determine normal and tangencial velocity
    velocity_1_normal = np.dot(np.array([object_1.x_velocity, object_1.y_velocity]), line_of_action_vector) 
    velocity_2_normal = np.dot(np.array([object_2.x_velocity, object_2.y_velocity]), line_of_action_vector)
    #print(velocity_1_normal)
    
    # Elastic collision new
    m1 = np.sum(object_1.alpha_channel_array)
    m2 = np.sum(object_2.alpha_channel_array)
    
    v1n_new = ((m1 - m2) * velocity_1_normal + 2 * m2 * velocity_2_normal) / (m1 + m2)
    v2n_new = ((m2 - m1) * velocity_2_normal + 2 * m1 * velocity_1_normal) / (m1 + m2)
    
    v1_new = np.array([object_1.x_velocity, object_1.y_velocity]) + (v1n_new - velocity_1_normal) * line_of_action_vector
    v2_new = np.array([object_2.x_velocity, object_2.y_velocity]) + (v2n_new - velocity_2_normal) * line_of_action_vector
    
    #print(v1_new)
    
    object_1.x_velocity = v1_new[0]
    object_1.y_velocity = v1_new[1]
    
    object_2.x_velocity = v2_new[0]
    object_2.y_velocity = v2_new[1]
    
    # Elastic collision
    #m1 = object_1.width * object_1.height  # Approximate mass based on size
    #m2 = object_2.width * object_2.height
    #v1x = object_1.x_velocity
    #v1y = object_1.y_velocity
    #v2x = object_2.x_velocity
    #v2y = object_2.y_velocity
    
    #v1x_new = ((m1 - m2) / (m1 + m2)) * v1x + ((2 * m2) / (m1 + m2)) * v2x
    #v1y_new = ((m1 - m2) / (m1 + m2)) * v1y + ((2 * m2) / (m1 + m2)) * v2y
    #v2x_new = ((2 * m1) / (m1 + m2)) * v1x + ((m2 - m1) / (m1 + m2)) * v2x
    #v2y_new = ((2 * m1) / (m1 + m2)) * v1y + ((m2 - m1) / (m1 + m2)) * v2y
    
    #object_1.x_velocity = v1x_new
    #object_1.y_velocity = v1y_new
    #object_2.x_velocity = v2x_new
    #object_2.y_velocity = v2y_new
    
    # Separate slightly to prevent sticking
    distance = np.sqrt((object_1.x_pos - object_2.x_pos)**2 + (object_1.y_pos - object_2.y_pos)**2)
    overlap = object_1.collision_radius + object_2.collision_radius - distance
    dx = (object_1.x_pos - object_2.x_pos) / distance
    dy = (object_1.y_pos - object_2.y_pos) / distance
    
    object_1.x_pos += overlap * dx / 2
    object_1.y_pos += overlap * dy / 2
    object_2.x_pos -= overlap * dx / 2
    object_2.y_pos -= overlap * dy / 2