from sdl3 import *
import numpy as np

def get_boundary_pixels(surface):
    SDL_LockSurface(surface)
    pixels_pointer = surface.contents.pixels
    pitch = surface.contents.pitch
    height = surface.contents.h
    width = surface.contents.w
    
    raw_data = ctypes.string_at(pixels_pointer, pitch * height)
    pixel_array = np.frombuffer(raw_data, dtype = Uint32).copy()
    pixel_array = pixel_array.reshape((height, width))
    alpha_channel_array = pixel_array & 0xFF == 255
    opaque_pixels = np.column_stack(np.where(alpha_channel_array))
    
    # Find boundary pixels
    boundary_mask = np.zeros_like(alpha_channel_array, dtype=bool)

    for y, x in opaque_pixels:
        # Check 8-connected neighbors (up, down, left, right, diagonals)
        if (y > 0 and not alpha_channel_array[y - 1, x]) or \
           (y < height - 1 and not alpha_channel_array[y + 1, x]) or \
           (x > 0 and not alpha_channel_array[y, x - 1]) or \
           (x < width - 1 and not alpha_channel_array[y, x + 1]) or \
           (y > 0 and x > 0 and not alpha_channel_array[y - 1, x - 1]) or \
           (y > 0 and x < width - 1 and not alpha_channel_array[y - 1, x + 1]) or \
           (y < height - 1 and x > 0 and not alpha_channel_array[y + 1, x - 1]) or \
           (y < height - 1 and x < width - 1 and not alpha_channel_array[y + 1, x + 1]):
            boundary_mask[y, x] = True
    #print(boundary_mask)
    # Get coordinates of boundary pixels
    boundary_array = np.where(boundary_mask)
    boundary_array = np.column_stack(boundary_array)
    boundary_array = boundary_array[:, [1, 0]]
    
    # Convert to float for further processing
    boundary_array = boundary_array.astype(float)
    #print(boundary_array.size)
    
    SDL_UnlockSurface(surface)
    return boundary_array

def get_colliding_pixels(object_1, object_2):
    pass

def get_coarse_collision():
    pass

def get_fine_collision():
    pass