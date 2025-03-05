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
    if (object_1.is_colliding or \
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

def collision_velocity_update(object_1, object_2, pixel_collision_array):
    # Make sure there is a collision
    if pixel_collision_array.size == 0:
        return 0
    
    # Ensure array is copied
    pixel_collision_array = pixel_collision_array.copy()
    
    # Determine the coordinates of the collision's centroid
    centroid_x_position = np.mean(pixel_collision_array[:, 0])
    centroid_y_position = np.mean(pixel_collision_array[:, 1])
    centroid = np.array([centroid_x_position, centroid_y_position])
    
    # Determine the 4 pixels nearest to the centroid
    # For object 1
    neighbours = 16
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
    
    # Separation impulse
    #push_direction_1 = np.array([object_1.x_pos, object_1.y_pos]) - centroid
    #push_direction_1 = np.linalg.norm(push_direction_1)
    #print(push_direction_1)
    
    #push_direction_2 = np.array([object_2.x_pos, object_2.y_pos]) - centroid
    #push_direction_2 = np.linalg.norm(push_direction_2)
    
    #overlap_distance = np.min(np.linalg.norm(pixel_collision_array - centroid, axis=1))
    #print(overlap_distance)
    
    #object_1.x_pos += push_direction_1[0] * overlap_distance * 0.5
    #object_1.y_pos += push_direction_1[1] * overlap_distance * 0.5
    
    #object_2.x_pos -= push_direction_2[0] * overlap_distance * 0.5
    #object_2.y_pos -= push_direction_2[1] * overlap_distance * 0.5
    
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
    #distance = np.sqrt((object_1.x_pos - object_2.x_pos)**2 + (object_1.y_pos - object_2.y_pos)**2)
    #overlap = object_1.collision_radius + object_2.collision_radius - distance
    #dx = (object_1.x_pos - object_2.x_pos) / distance
    #dy = (object_1.y_pos - object_2.y_pos) / distance
    
    #object_1.x_pos += overlap * dx / 2
    #object_1.y_pos += overlap * dy / 2
    #object_2.x_pos -= overlap * dx / 2
    #object_2.y_pos -= overlap * dy / 2

def physics_update(objects_list):
    i = 0
    while i < len(objects_list):
        j = 0
        original_x_pos = objects_list[i].x_pos
        original_y_pos = objects_list[i].y_pos
        original_position_rect = objects_list[i].position_rect
        objects_list[i].x_pos += objects_list[i].x_velocity
        objects_list[i].y_pos += objects_list[i].y_velocity
        objects_list[i].position_rect.x = objects_list[i].x_pos - objects_list[i].width / 2
        objects_list[i].position_rect.y = objects_list[i].y_pos - objects_list[i].height / 2
        while j < len(objects_list):
            if (j != i
                and distance_between(objects_list[i], objects_list[j]) <= (objects_list[i].collision_radius + objects_list[j].collision_radius)
                and pixel_perfect_collision(objects_list[i], objects_list[j]).size > 0):
                steps = 0
                increment = 0.5
                total_time = increment
                while steps < 10 or pixel_perfect_collision(objects_list[i], objects_list[j]).size == 0:
                    #print("collision", i, j)
                    if pixel_perfect_collision(objects_list[i], objects_list[j]).size > 0:
                        objects_list[i].x_pos -= objects_list[i].x_velocity * increment
                        objects_list[i].y_pos -= objects_list[i].y_velocity * increment
                        total_time -= increment
                    else:
                        objects_list[i].x_pos += objects_list[i].x_velocity * increment
                        objects_list[i].y_pos += objects_list[i].y_velocity * increment
                        total_time += increment
                    #objects_list[i].position_rect = original_position_rect
                    if steps < 10:
                        increment /= 2
                    steps += 1
                pixel_collision_array = pixel_perfect_collision(objects_list[i], objects_list[j])
                #print(pixel_collision_array)
                collision_velocity_update(objects_list[i], objects_list[j], pixel_collision_array)
                objects_list[i].x_pos += objects_list[i].x_velocity * (1 - total_time)
                objects_list[i].y_pos += objects_list[i].y_velocity * (1 - total_time)
                while pixel_perfect_collision(objects_list[i], objects_list[j]).size > 0:
                    objects_list[i].x_pos += objects_list[i].x_velocity * (1 - total_time)
                    objects_list[i].y_pos += objects_list[i].y_velocity * (1 - total_time)
                objects_list[i].position_rect.x = objects_list[i].x_pos - objects_list[i].width / 2
                objects_list[i].position_rect.y = objects_list[i].y_pos - objects_list[i].height / 2
                #print(pixel_perfect_collision(objects_list[i], objects_list[j]))
            j += 1
        if (objects_list[i].x_pos < objects_list[i].collision_radius
            or objects_list[i].x_pos > 1280 - objects_list[i].collision_radius):
            objects_list[i].x_velocity *= (-1)
            
        if (objects_list[i].y_pos < objects_list[i].collision_radius
            or objects_list[i].y_pos > 720 - objects_list[i].collision_radius):
            objects_list[i].y_velocity *= (-1)
            
        i += 1

def physics_update_v2(objects_list):
    #print("update v2")
    t = 0
    i = 0
    while t < 100:
        print(t)
        while i < len(objects_list):
            j = i
            #i_original_x = objects_list[i].x_pos
            #i_original_y = objects_list[i].y_pos        
            while j < len(objects_list):
                #j_original_x = objects_list[j].x_pos
                #j_original_y = objects_list[j].y_pos
                
                objects_list[i].x_pos += objects_list[i].x_velocity * 0.01 * 60
                objects_list[i].y_pos += objects_list[i].y_velocity * 0.01 * 60
                
                objects_list[j].x_pos += objects_list[j].x_velocity * 0.01 * 60
                objects_list[j].y_pos += objects_list[j].y_velocity * 0.01 * 60
                
                pixel_collision_array = pixel_perfect_collision(objects_list[i], objects_list[j])
                
                if (i != j
                    and distance_between(objects_list[i], objects_list[j]) < (objects_list[i].collision_radius + objects_list[j].collision_radius)
                    and pixel_collision_array.size > 0):
                    collision_velocity_update(objects_list[i], objects_list[j], pixel_collision_array)
                    
                if (objects_list[i].x_pos < objects_list[i].collision_radius
                or objects_list[i].x_pos > 1280 - objects_list[i].collision_radius):
                    objects_list[i].x_velocity *= (-1)
            
                if (objects_list[i].y_pos < objects_list[i].collision_radius
                or objects_list[i].y_pos > 720 - objects_list[i].collision_radius):
                    objects_list[i].y_velocity *= (-1)
                j += 1
            i += 1
        t += 1
    i = 0
    while i < len(objects_list):
        objects_list[i].position_rect.x = objects_list[i].x_pos - objects_list[i].width / 2
        objects_list[i].position_rect.y = objects_list[i].y_pos - objects_list[i].height / 2
        i += 1
        
def physics_update_v3(objects_list):
    t = 0
    max_t = 20
    while t < max_t:
        i = 0
        while i < len(objects_list):
            objects_list[i].x_pos += objects_list[i].x_velocity / max_t
            objects_list[i].y_pos += objects_list[i].y_velocity / max_t
            i += 1
        i = 0
        while i < len(objects_list):
            j = i
            while j < len(objects_list):
                pixel_collision_array = pixel_perfect_collision(objects_list[i], objects_list[j])
                if (i != j
                    and distance_between(objects_list[i], objects_list[j]) < (objects_list[i].collision_radius + objects_list[j].collision_radius)
                    and pixel_collision_array.size > 0):
                    collision_velocity_update(objects_list[i], objects_list[j], pixel_collision_array)
                j += 1
            i += 1
        i = 0
        while i < len(objects_list):
                if (objects_list[i].x_pos < objects_list[i].collision_radius
                    or objects_list[i].x_pos > 1280 - objects_list[i].collision_radius):
                    objects_list[i].x_velocity *= (-1)
            
                if (objects_list[i].y_pos < objects_list[i].collision_radius
                    or objects_list[i].y_pos > 720 - objects_list[i].collision_radius):
                    objects_list[i].y_velocity *= (-1)
                i += 1
        i = 0
        while i < len(objects_list):
            objects_list[i].position_rect.x = objects_list[i].x_pos - objects_list[i].width / 2
            objects_list[i].position_rect.y = objects_list[i].y_pos - objects_list[i].height / 2
            i += 1
        t += 1
        
def physics_update_v4(objects_list):
    t = 0
    max_t = 20
    while t < max_t:
        i = 0
        while i < len(objects_list):
            j = 0
            while j < len(objects_list):
                if (i != j
                    and distance_between(objects_list[i], objects_list[j]) < (objects_list[i].collision_radius + objects_list[j].collision_radius)
                    and pixel_boundary_collision(objects_list[i], objects_list[j]) != 0):
                    collision_update_v2(objects_list[i], objects_list[j])
                    objects_list[i].is_colliding = True
                    objects_list[j].is_colliding = True
                else:
                    objects_list[i].is_colliding = False
                    objects_list[j].is_colliding = False
                j += 1
            objects_list[i].x_pos += objects_list[i].x_velocity / max_t
            objects_list[i].y_pos += objects_list[i].y_velocity / max_t
            if (objects_list[i].x_pos < objects_list[i].collision_radius
                or objects_list[i].x_pos > 1280 - objects_list[i].collision_radius):
                objects_list[i].x_velocity *= (-1)
            
            if (objects_list[i].y_pos < objects_list[i].collision_radius
                or objects_list[i].y_pos > 720 - objects_list[i].collision_radius):
                objects_list[i].y_velocity *= (-1)
            objects_list[i].position_rect.x = objects_list[i].x_pos - objects_list[i].width / 2
            objects_list[i].position_rect.y = objects_list[i].y_pos - objects_list[i].height / 2
            i += 1
        t += 1
        
def physics_update_v5(objects_list):
    t = 0
    max_t = 20
    while t < max_t:
        i = 0
        while i < len(objects_list):
            j = 0
            while j < len(objects_list):
                # If there's collision
                if (i != j
                    and not objects_list[i].is_colliding
                    and not objects_list[j].is_colliding
                    and distance_between(objects_list[i], objects_list[j]) < (objects_list[i].collision_radius + objects_list[j].collision_radius)
                    and pixel_boundary_collision(objects_list[i], objects_list[j]) != 0):
                    sub_t = 0.5
                    
                    objects_list[i].x_pos -= objects_list[i].x_velocity / max_t / sub_t
                    objects_list[i].y_pos -= objects_list[i].y_velocity / max_t / sub_t
                    
                    objects_list[j].x_pos -= objects_list[j].x_velocity / max_t / sub_t
                    objects_list[j].y_pos -= objects_list[j].y_velocity / max_t / sub_t
                    
                    extra_t = 0.5
                    while sub_t > 0.01:
                        sub_t /= 2
                        if pixel_boundary_collision(objects_list[i], objects_list[j]) != 0:
                            objects_list[i].x_pos -= objects_list[i].x_velocity / max_t / sub_t
                            objects_list[i].y_pos -= objects_list[i].y_velocity / max_t / sub_t
                            
                            objects_list[j].x_pos -= objects_list[j].x_velocity / max_t / sub_t
                            objects_list[j].y_pos -= objects_list[j].y_velocity / max_t / sub_t
                            extra_t -= sub_t
                        else:
                            objects_list[i].x_pos += objects_list[i].x_velocity / max_t / sub_t
                            objects_list[i].y_pos += objects_list[i].y_velocity / max_t / sub_t
                            
                            objects_list[j].x_pos += objects_list[j].x_velocity / max_t / sub_t
                            objects_list[j].y_pos += objects_list[j].y_velocity / max_t / sub_t
                            extra_t += sub_t
                    while pixel_boundary_collision(objects_list[i], objects_list[j]) == 0:
                        objects_list[i].x_pos += objects_list[i].x_velocity / max_t / sub_t
                        objects_list[i].y_pos += objects_list[i].y_velocity / max_t / sub_t
                        objects_list[j].x_pos += objects_list[j].x_velocity / max_t / sub_t
                        objects_list[j].y_pos += objects_list[j].y_velocity / max_t / sub_t
                        extra_t += sub_t
                    collision_update_v2(objects_list[i], objects_list[j])
                    #print(extra_t)
                    if extra_t < 1:
                        objects_list[i].x_pos += objects_list[i].x_velocity / max_t / (1 - extra_t)
                        objects_list[i].y_pos += objects_list[i].y_velocity / max_t / (1 - extra_t)
                        objects_list[j].x_pos += objects_list[j].x_velocity / max_t / (1 - extra_t)
                        objects_list[j].y_pos += objects_list[j].y_velocity / max_t / (1 - extra_t)
                        extra_t = 1
                        if pixel_boundary_collision(objects_list[i], objects_list[j]) != 0:
                            objects_list[i].is_colliding = True
                            objects_list[j].is_colliding = True
                    print(sub_t, pixel_boundary_collision(objects_list[i], objects_list[j]))
                    # Handle cases where extra_t > 1
                else:
                    # If there's no collision
                    objects_list[i].is_colliding = False
                    objects_list[j].is_colliding = False
                    
                j += 1
            objects_list[i].x_pos += objects_list[i].x_velocity / max_t
            objects_list[i].y_pos += objects_list[i].y_velocity / max_t
            if (objects_list[i].x_pos < objects_list[i].collision_radius
                or objects_list[i].x_pos > 1280 - objects_list[i].collision_radius):
                objects_list[i].x_velocity *= (-1)
            
            if (objects_list[i].y_pos < objects_list[i].collision_radius
                or objects_list[i].y_pos > 720 - objects_list[i].collision_radius):
                objects_list[i].y_velocity *= (-1)
            objects_list[i].position_rect.x = objects_list[i].x_pos - objects_list[i].width / 2
            objects_list[i].position_rect.y = objects_list[i].y_pos - objects_list[i].height / 2
            i += 1
        t += 1
        
def distance_between(object_1, object_2):
    return np.sqrt((object_1.x_pos - object_2.x_pos) ** 2 + (object_1.y_pos - object_2.y_pos) ** 2)