import sys, time, numpy as np
from sdl3 import *
from GameObject import *
from Boot import *

# Create classes for the game objects

window_w = 1280
window_h = 720

window, renderer = initialize(window_w, window_h)
custom_cursor_surface = IMG_Load("c:/Users/Mike/3D Objects/AI_Project/crosshair.png".encode("utf-8"))
if not custom_cursor_surface:
    print("Error loading cursor image: ", SDL_GetError())
custom_cursor = SDL_CreateColorCursor(custom_cursor_surface, 16, 16)
SDL_SetCursor(custom_cursor)

spaceship_surface = IMG_Load("c:/Users/Mike/3D Objects/AI_Project/spaceship.png".encode("utf-8"))
if not spaceship_surface:
    print("Error loading spaceship image: ", SDL_GetError())
spaceship_position_w = window_w / 2
spaceship_position_h = window_h / 2
spaceship_width = spaceship_surface.contents.w
spaceship_height = spaceship_surface.contents.h
spaceship_position_rect = SDL_FRect(spaceship_position_w - spaceship_width / 2,
                                    spaceship_position_h - spaceship_height / 2,
                                    spaceship_surface.contents.w, 
                                    spaceship_surface.contents.h)
fps = 60
seconds_per_frame = 1 / fps

spaceship_texture = SDL_CreateTextureFromSurface(renderer, spaceship_surface)
if not spaceship_texture:
    print("Error creating spaceship texture: ", SDL_GetError())
angle = 0
spaceship_accel_magnitude = 1
ship_vel = [0, 0]
cursor_position = [window_w / 2, 0]
dx = 0
dy = 1
background_dot_list = []
for i in range(50):
    dot = BackgroundDot(ship_vel, window_w, window_h, renderer)
    background_dot_list.append(dot)

player_spaceship = Spaceship(window_w / 2, window_h / 2, renderer)

running = True
event = SDL_Event()
while running:
    frame_start_time = time.time()
    SDL_RenderClear(renderer)
    while SDL_PollEvent(event):
        if event.type == SDL_EVENT_QUIT:
            running = False
        if event.type == SDL_EVENT_MOUSE_MOTION:
            cursor_position[0] = event.motion.x
            cursor_position[1] = event.motion.y
        if event.type == SDL_EVENT_KEY_DOWN:
            if event.key.key == SDLK_W:
                ship_vel[0] += spaceship_accel_magnitude * np.sin(angle * np.pi / 180)
                ship_vel[1] -= spaceship_accel_magnitude * np.cos(angle * np.pi / 180)       
    delta_pos = [cursor_position[0] - spaceship_position_rect.x, cursor_position[1] - spaceship_position_rect.y]
    if delta_pos[0] > 0:
        angle = 180 - np.arccos(delta_pos[1] / np.sqrt(delta_pos[0] ** 2 + delta_pos[1] ** 2)) * 180 / np.pi
    else:
        angle = np.arccos(delta_pos[1] / np.sqrt(delta_pos[0] ** 2 + delta_pos[1] ** 2)) * 180 / np.pi - 180
    spaceship_position_rect.x += ship_vel[0]
    spaceship_position_rect.y += ship_vel[1]
    for i in background_dot_list:
        i.update(ship_vel)
        SDL_RenderTexture(renderer, i.texture, None, i.position_rect)
    SDL_RenderTextureRotated(renderer, spaceship_texture, None, spaceship_position_rect, angle, None, SDL_FLIP_NONE)
    SDL_RenderPresent(renderer)
    frame_time = time.time() - frame_start_time
    if frame_time < seconds_per_frame:
        time.sleep(seconds_per_frame - frame_time)
    
SDL_Quit()