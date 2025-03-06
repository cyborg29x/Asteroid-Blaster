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
    alpha_channel_array = (pixel_array & 0xFF) > 0
    opaque_pixels = np.column_stack(np.where(alpha_channel_array))
    
    # Vectorized boundary detection using np.pad to check 8-connected neighbors
    padded = np.pad(alpha_channel_array, pad_width=1, mode='constant', constant_values=False)
    inner = padded[:-2, :-2] & padded[:-2, 1:-1] & padded[:-2, 2:] & \
            padded[1:-1, :-2] & padded[1:-1, 2:] & \
            padded[2:, :-2] & padded[2:, 1:-1] & padded[2:, 2:]
    boundary_mask = alpha_channel_array & (~inner)
    #print(boundary_mask)
    # Get coordinates of boundary pixels
    boundary_array = np.where(boundary_mask)
    boundary_array = np.column_stack(boundary_array)
    boundary_array = boundary_array[:, [1, 0]]
    
    # Convert to float for further processing
    boundary_array = boundary_array.astype(float)
    #print(boundary_array.size)
    
    # Translate origin to center of surface
    boundary_array[:, 0] -= width / 2
    boundary_array[:, 1] -= height / 2    
    
    SDL_UnlockSurface(surface)
    return boundary_array

def get_colliding_pixels(object_1, object_2):
    pass

def get_coarse_collision():
    pass

def get_fine_collision():
    pass