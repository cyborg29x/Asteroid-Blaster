from sdl3 import *
from GameLogic import *

def initialize(window_width, window_height):
    window = SDL_CreateWindow(b"Game Window", window_width, window_height, 0)
    renderer = SDL_CreateRenderer(window, None)
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255)
    SDL_SetRenderVSync(renderer, 1)
    return window, renderer

file_list = ["c:/Users/Mike/3D Objects/AI_Project/asteroid.png".encode("utf-8")]
def load_sprites(renderer):
    # Todo: Make it load dynamically with a for loop or something
    asteroid = GameObject("asteroid",
                          file_list[0],
                          renderer)
    return asteroid
        
class GameObject():
    def __init__(self, name, file_path, renderer):
        self.name = name
        temporary_surface = IMG_Load(file_path)
        temporary_surface = SDL_ConvertSurface(temporary_surface, SDL_PIXELFORMAT_RGBA8888)
        self.bitmask = pixel_alpha_channel_extraction(temporary_surface, renderer)
        self.texture = SDL_CreateTextureFromSurface(renderer, temporary_surface)
        SDL_DestroySurface(temporary_surface)