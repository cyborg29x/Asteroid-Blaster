from sdl3 import *

def initialize(window_width, window_height):
    window = SDL_CreateWindow(b"Game Window", window_width, window_height, 0)
    renderer = SDL_CreateRenderer(window, None)
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255)
    SDL_SetRenderVSync(renderer, 1)
    return window, renderer
    