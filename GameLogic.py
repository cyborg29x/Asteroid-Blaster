import numpy as np
import ctypes
import time
from sdl3 import *
from Collision import *

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
    alpha_channel_array = pixel_array & 0xFF == 255
    alpha_channel_array = np.where(alpha_channel_array)
    alpha_channel_array = np.column_stack((alpha_channel_array[1], alpha_channel_array[0]))
    alpha_channel_array = alpha_channel_array.astype(np.int32)
    SDL_UnlockSurface(surface)
    #print(alpha_channel_array)
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

def pixel_boundary_collision(object_1, object_2):
    # Get non-transparent boundary pixels
    pixel_array_1 = object_1.boundary_pixels.copy()
    pixel_array_2 = object_2.boundary_pixels.copy()
    
    # Convert degrees to radians
    angle_1 = np.radians(object_1.angle)
    angle_2 = np.radians(object_2.angle)
    
    # Get object positions
    p1 = np.array([object_1.x_pos, object_1.y_pos])
    p2 = np.array([object_2.x_pos, object_2.y_pos])
    
    # Convert arrays to global coordinates
    pixel_array_1 = coordinate_conversion(pixel_array_1, 
                                          p1,
                                          object_1.angle)
    pixel_array_2 = coordinate_conversion(pixel_array_2,
                                          p2,
                                          object_2.angle)
    
    # Check if inside collision radius
    # Object 1
    p1_distances_1 = (pixel_array_1[:, 0] - p1[0]) ** 2 + (pixel_array_1[:, 1] - p1[1]) ** 2
    pixel_array_1 = pixel_array_1[p1_distances_1 < object_1.collision_radius ** 2]
    
    p2_distances_1 = (pixel_array_2[:, 0] - p1[0]) ** 2 + (pixel_array_2[:, 1] - p1[1]) ** 2
    pixel_array_2 = pixel_array_2[p2_distances_1 < object_1.collision_radius ** 2]
    
    # Object 2
    p1_distances_2 = (pixel_array_1[:, 0] - p2[0]) ** 2 + (pixel_array_1[:, 1] - p2[1]) ** 2
    pixel_array_1 = pixel_array_1[p1_distances_2 < object_1.collision_radius ** 2]
    
    p2_distances_2 = (pixel_array_2[:, 0] - p2[0]) ** 2 + (pixel_array_2[:, 1] - p2[1]) ** 2
    pixel_array_2 = pixel_array_2[p2_distances_2 < object_1.collision_radius ** 2]
    
    # Check if empty
    if (pixel_array_1.size == 0 or \
        pixel_array_2.size == 0):
        return 0
    
    #print(pixel_array_1, pixel_array_2)
    
    # Determine centroids
    centroid_1 = np.array([np.mean(pixel_array_1[:, 0]), np.mean(pixel_array_1[:, 1])])
    centroid_2 = np.array([np.mean(pixel_array_2[:, 0]), np.mean(pixel_array_2[:, 1])])
    
    
    return [centroid_1, centroid_2]
    

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
    merged_array = merged_array[distances_squared < object_1.collision_radius ** 2]
    
    # Repeat for object 2
    object_2_position = np.array([object_2.x_pos, object_2.y_pos])
    distances_squared = (merged_array[:, 0] - object_2_position[0]) ** 2 + (merged_array[:, 1] - object_2_position[1]) ** 2
    merged_array = merged_array[distances_squared < object_2.collision_radius ** 2]
    
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

def collision_update_v2(object_1, object_2):
    if (object_1.is_colliding and \
        object_2.is_colliding):
        return 0
    centroids = pixel_boundary_collision(object_1, object_2)
    #print(centroids)
    if centroids == 0:
        return 0
    centroid_1 = centroids[0]
    centroid_2 = centroids[1]
    
    p1 = np.array([object_1.x_pos, object_1.y_pos])
    v1 = np.array([object_1.x_velocity, object_1.y_velocity])
    
    p2 = np.array([object_2.x_pos, object_2.y_pos])
    v2 = np.array([object_2.x_velocity, object_2.y_velocity])
    
    #n = centroid_1 - centroid_2
    n = p1 - p2
    n = n / np.linalg.norm(n)
    #print(n)
    
    v1_n = np.dot(v1, n) * n
    v1_t = v1 - v1_n
    
    v2_n = np.dot(v2, n) * n
    v2_t = v2 - v2_n
    
    relative_velocity_n = np.dot(v1 - v2, n)
    #print(relative_velocity_n)
    if relative_velocity_n > 0:
        return 0
    
    # Velocity impulse
    m1 = np.sum(object_1.alpha_channel_array)
    m2 = np.sum(object_2.alpha_channel_array)
    e = 1
    v1 = v1 - (1 + e) * np.dot(v1, n) * n
    v2 = v2 - (1 + e) * np.dot(v2, n) * n
    
    #print(v1)
    object_1.x_velocity = v1[0]
    object_1.y_velocity = v1[1]
    #print(object_1.x_velocity, object_1.y_velocity)
    
    object_2.x_velocity = v2[0]
    object_2.y_velocity = v2[1]
   
def physics_update(objects_list):
    t = 0
    max_t = 20
    while t < max_t:
        # Initial moving of objects
        i = 0
        while i < len(objects_list):
            objects_list[i].x_pos += objects_list[i].x_velocity / max_t
            objects_list[i].y_pos += objects_list[i].y_velocity / max_t
            # Bounce objects off of screen borders
            if (objects_list[i].x_pos < 0
                or objects_list[i].x_pos > 1280):
                objects_list[i].x_velocity *= (-1)
            
            if (objects_list[i].y_pos < 0
                or objects_list[i].y_pos > 720):
                objects_list[i].y_velocity *= (-1)
            i += 1
        
        # Check for collisions
        i = 0
        while i < len(objects_list):
            j = i + 1
            while j < len(objects_list):
                # If there's a collision
                if (distance_between(objects_list[i], objects_list[j]) < (objects_list[i].collision_radius + objects_list[j].collision_radius) and \
                    pixel_boundary_collision(objects_list[i], objects_list[j]) != 0):
                    # Mark as colliding to prevent further velocity adjustment until clear
                    collision_update_v2(objects_list[i], objects_list[j])
                    objects_list[i].is_colliding = True
                    objects_list[j].is_colliding = True
                else:
                    # Mark as clear
                    objects_list[i].is_colliding = False
                    objects_list[j].is_colliding = False
                j += 1
            i +=1
        t +=1
    
    # Final update of draw rectangles
    i = 0
    while i < len(objects_list):
        objects_list[i].position_rect.x = objects_list[i].x_pos - objects_list[i].width / 2
        objects_list[i].position_rect.y = objects_list[i].y_pos - objects_list[i].height / 2
        i += 1
        
def distance_between(object_1, object_2):
    return np.sqrt((object_1.x_pos - object_2.x_pos) ** 2 + (object_1.y_pos - object_2.y_pos) ** 2)