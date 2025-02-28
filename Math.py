import numpy as np
import ctypes
from sdl3 import *

def pixel_alpha_channel_extraction(surface):
    pixels_pointer = surface.contents.pixels
    pitch = surface.contents.pitch
    height = surface.contents.h
    width = surface.contents.w
    
    raw_data = ctypes.string_at(pixels_pointer, pitch * height)
    pixel_array = np.frombuffer(raw_data, dtype = Uint32)
    pixel_array = pixel_array.reshape((height, width))
    alpha_channel_array = pixel_array & 0xFF > 0
    alpha_channel_array = np.where(alpha_channel_array)
    alpha_channel_array = np.column_stack((alpha_channel_array[1], alpha_channel_array[0]))
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
    return distance <= radius_1 + radius_2

def pixel_perfect_collision(object_1, object_2):
    # Get non-transparent pixel arrays
    pixel_array_1 = object_1.alpha_channel_array
    pixel_array_2 = object_2.alpha_channel_array
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
    
    return merged_array

def collision_velocity_update(object_1, object_2):
    # Elastic collision
    m1 = object_1.width * object_1.height  # Approximate mass based on size
    m2 = object_2.width * object_2.height
    v1x = object_1.x_velocity
    v1y = object_1.y_velocity
    v2x = object_2.x_velocity
    v2y = object_2.y_velocity
    
    v1x_new = ((m1 - m2) / (m1 + m2)) * v1x + ((2 * m2) / (m1 + m2)) * v2x
    v1y_new = ((m1 - m2) / (m1 + m2)) * v1y + ((2 * m2) / (m1 + m2)) * v2y
    v2x_new = ((2 * m1) / (m1 + m2)) * v1x + ((m2 - m1) / (m1 + m2)) * v2x
    v2y_new = ((2 * m1) / (m1 + m2)) * v1y + ((m2 - m1) / (m1 + m2)) * v2y
    
    object_1.x_velocity = v1x_new
    object_1.y_velocity = v1y_new
    object_2.x_velocity = v2x_new
    object_2.y_velocity = v2y_new
    
    # Separate slightly to prevent sticking
    distance = np.sqrt((object_1.x_pos - object_2.x_pos)**2 + (object_1.y_pos - object_2.y_pos)**2)
    overlap = object_1.collision_radius + object_2.collision_radius - distance
    dx = (object_1.x_pos - object_2.x_pos) / distance
    dy = (object_1.y_pos - object_2.y_pos) / distance
    
    object_1.x_pos += overlap * dx / 2
    object_1.y_pos += overlap * dy / 2
    object_2.x_pos -= overlap * dx / 2
    object_2.y_pos -= overlap * dy / 2