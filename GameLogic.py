import numpy as np
import ctypes
import time
from sdl3 import *
from Collision import *

def pixel_alpha_channel_extraction(surface, renderer):
    SDL_LockSurface(surface)
    
    pixels_ptr = surface.contents.pixels
    pitch = surface.contents.pitch
    height = surface.contents.h
    width = surface.contents.w
    
    raw_data = ctypes.string_at(pixels_ptr, pitch * height)
    pixel_array = np.frombuffer(raw_data, dtype=np.uint8).copy()
    pixel_array = pixel_array.reshape((height, width, 4))
    
    # Identify non-transparent pixels
    alpha_channel_mask = pixel_array[:, :, 3] > 0
    
    # Extract coordinates in (Y, X) order, then swap to (X, Y)
    non_transparent_pixels = np.column_stack(np.where(alpha_channel_mask))[:, [1, 0]]  # Swap order
    
    non_transparent_pixels = non_transparent_pixels.astype(float)
    
    # Translate origin to center of surface
    non_transparent_pixels[:, 0] -= width / 2
    non_transparent_pixels[:, 1] -= height / 2
    
    SDL_UnlockSurface(surface)
    
    return non_transparent_pixels

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

def coordinate_conversion(coordinates_array, translation_coordinate, angle, renderer):
    # Relies on correctly converted input array (move origin to center from top left)
    # Make sure to copy instead of referencing
    coordinates_array = coordinates_array.astype(float).copy()
    translation_coordinate = translation_coordinate.astype(float).copy()
    
    #print(coordinates_array)
    # Transform angle from degrees to radians (clockwise positive)
    angle = np.radians(angle)
    #print(angle, np.cos(angle), np.sin(angle))
    
    # Store original values
    original_x = coordinates_array[:, 0]
    original_y = coordinates_array[:, 1]
    
    #print(coordinates_array.shape, original_x.shape, original_y.shape)
    
    # Apply rotation conversion
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    coordinates_array[:, :2] = (coordinates_array[:, :2] @ rotation_matrix.T + translation_coordinate)
    
    #SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255)
    #for pixel in coordinates_array:
    #    SDL_RenderPoint(renderer, pixel[0], pixel[1])
    #for pixel in pixel_array_2:
    #    SDL_RenderPoint(renderer, pixel[0], pixel[1])
    #SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255)
    
    # Apply final translation to global coordinates
    #coordinates_array[:, 0] += translation_coordinate[0]
    #coordinates_array[:, 1] += translation_coordinate[1]

    #SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255)
    #for pixel in coordinates_array:
    #    SDL_RenderPoint(renderer, pixel[0], pixel[1])
    #SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255)

    #print(coordinates_array)
    return coordinates_array

def pixel_boundary_collision(object_1, object_2, renderer):
    # Get all non-transparent pixels
    pixel_array_1 = object_1.alpha_channel_array.copy()
    pixel_array_2 = object_2.alpha_channel_array.copy()

    #SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255)
    #for pixel in pixel_array_1:
    #    SDL_RenderPoint(renderer, pixel[0], pixel[1])
    #for pixel in pixel_array_2:
    #    SDL_RenderPoint(renderer, pixel[0], pixel[1])
    #SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255)

    # Get object positions and collision radii squared
    p1, p2 = np.array([object_1.x_pos, object_1.y_pos]), np.array([object_2.x_pos, object_2.y_pos])
    r1_squared, r2_squared = object_1.collision_radius ** 2, object_2.collision_radius ** 2

    # Convert arrays to global coordinates
    pixel_array_1 = coordinate_conversion(pixel_array_1, p1, object_1.angle, renderer)
    pixel_array_2 = coordinate_conversion(pixel_array_2, p2, object_2.angle, renderer)
    #print(pixel_array_1, pixel_array_2)
    #SDL_Delay(10)
    
    #SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255)
    #for pixel in pixel_array_1:
    #    SDL_RenderPoint(renderer, pixel[0], pixel[1])
    #for pixel in pixel_array_2:
    #    SDL_RenderPoint(renderer, pixel[0], pixel[1])
    #SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255)
    
    # Filter pixels outside collision radii using boolean indexing
    def is_point_inside(point, center, radius_squared):
        return (point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2 < radius_squared

    mask1 = [is_point_inside(point, p1, r1_squared) and is_point_inside(point, p2, r2_squared) for point in pixel_array_1]
    mask2 = [is_point_inside(point, p1, r1_squared) and is_point_inside(point, p2, r2_squared) for point in pixel_array_2]

    pixel_array_1, pixel_array_2 = pixel_array_1[mask1], pixel_array_2[mask2]
    #print(pixel_array_1, pixel_array_2)
    
    #SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255)
    #for pixel in pixel_array_1:
    #    SDL_RenderPoint(renderer, pixel[0], pixel[1])
    #for pixel in pixel_array_2:
    #    SDL_RenderPoint(renderer, pixel[0], pixel[1])
    #SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255)
    
    # Check if either array is empty
    if pixel_array_1.size == 0 or pixel_array_2.size == 0:
        return np.array([])

    # Check if any pixels overlap
    overlapping_pixels = np.array([pixel for pixel in pixel_array_1 if any(np.linalg.norm(pixel - pixel_array_2, axis=1) < 1)])

    #SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255)
    #for pixel in overlapping_pixels:
    #    SDL_RenderPoint(renderer, pixel[0], pixel[1])
    #SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255)

    return overlapping_pixels

def collision_update_v2(object_1, object_2):
    #print(object_1.is_colliding, object_2.is_colliding)
    if (object_1.is_colliding or \
        object_2.is_colliding):
        return 0
    #centroids = pixel_boundary_collision(object_1, object_2)
    #if centroids == 0:
    #    return 0
    #centroid_1 = centroids[0]
    #centroid_2 = centroids[1]
    
    p1 = np.array([object_1.x_pos, object_1.y_pos])
    v1 = np.array([object_1.x_velocity, object_1.y_velocity])
    
    p2 = np.array([object_2.x_pos, object_2.y_pos])
    v2 = np.array([object_2.x_velocity, object_2.y_velocity])
    
    #n = centroid_1 - centroid_2
    n = p1 - p2
    n = n / np.linalg.norm(n)
    
    v1_n = np.dot(v1, n) * n
    v1_t = v1 - v1_n
    
    v2_n = np.dot(v2, n) * n
    v2_t = v2 - v2_n
    
    relative_velocity_n = np.dot(v1 - v2, n)
    if relative_velocity_n > 0:
        return 0
    
    # Velocity impulse
    m1 = np.sum(object_1.alpha_channel_array)
    m2 = np.sum(object_2.alpha_channel_array)
    e = 1
    v1_new = v1 - ((1 + e) * m2 / (m1 + m2)) * np.dot(v1 - v2, n) * n
    v2_new = v2 + ((1 + e) * m1 / (m1 + m2)) * np.dot(v1 - v2, n) * n
    
    object_1.x_velocity = v1_new[0]
    object_1.y_velocity = v1_new[1]
    
    object_2.x_velocity = v2_new[0]
    object_2.y_velocity = v2_new[1]
   
def physics_update(objects_list, renderer):
    max_t = 20
    for t in range(max_t):
        for obj in objects_list:
            obj.x_pos += obj.x_velocity / max_t
            obj.y_pos += obj.y_velocity / max_t

            if obj.x_pos < 0 or obj.x_pos > 1280:
                obj.x_velocity *= -1
            if obj.y_pos < 0 or obj.y_pos > 720:
                obj.y_velocity *= -1

        for i, obj1 in enumerate(objects_list):
            for obj2 in objects_list[i + 1:]:
                collision_distance = obj1.collision_radius + obj2.collision_radius
                if distance_between(obj1, obj2) < collision_distance:
                    colliding_pixels = pixel_boundary_collision(obj1, obj2, renderer)
                    #print(colliding_pixels)
                    if colliding_pixels.size > 0:
                        collision_update_v2(obj1, obj2)
                        obj1.is_colliding = obj2.is_colliding = True
                    else:
                        obj1.is_colliding = obj2.is_colliding = False

    for obj in objects_list:
        obj.position_rect.x = obj.x_pos - obj.width / 2
        obj.position_rect.y = obj.y_pos - obj.height / 2
        
def distance_between(object_1, object_2):
    return np.sqrt((object_1.x_pos - object_2.x_pos) ** 2 + (object_1.y_pos - object_2.y_pos) ** 2)