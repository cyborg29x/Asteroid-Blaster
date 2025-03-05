import sys, time, numpy as np
import ctypes
import os
from sdl3 import *
from GameObject import *
from Boot import *

# Create classes for the game objects

window_w = 1280
window_h = 720
window, renderer = initialize(window_w, window_h)
asteroid_prototype = load_sprites(renderer)

cursor = Cursor()

fps = 60
seconds_per_frame = 1 / fps

player_spaceship = Spaceship(window_w / 2, window_h / 2, renderer)
spaceship_accel_magnitude = 0.1
background_dot_list = []
for i in range(50):
    dot = BackgroundDot([player_spaceship.x_velocity, player_spaceship.y_velocity], window_w, window_h, renderer)
    background_dot_list.append(dot)
asteroid_list = []
for i in range(10):
    asteroid = Asteroid(renderer)
    asteroid_list.append(asteroid)
missile_list = []
running = True
w_key_down = False
event = SDL_Event()
#print(asteroid_list[9].alpha_channel_array)
while running:
    frame_start_time = time.time()
    SDL_RenderClear(renderer)
    while SDL_PollEvent(event):
        if event.type == SDL_EVENT_QUIT:
            running = False
        if event.type == SDL_EVENT_MOUSE_MOTION:
            cursor.update(event.motion.x, event.motion.y)          
        if event.type == SDL_EVENT_KEY_DOWN:
            if event.key.key == SDLK_W:
                if not w_key_down:
                    w_key_down = True
        if event.type == SDL_EVENT_KEY_UP:
            if event.key.key == SDLK_W:
                w_key_down = False
        if event.type == SDL_EVENT_MOUSE_BUTTON_DOWN:
            if event.button.button == SDL_BUTTON_LEFT:
                missile_list.append(Missile(player_spaceship.x_pos, player_spaceship.y_pos, player_spaceship.x_velocity, player_spaceship.y_velocity, player_spaceship.angle, renderer))
    for i in missile_list:
        i.update()
        SDL_RenderTextureRotated(renderer, i.texture, None, i.position_rect, i.angle, None, SDL_FLIP_NONE)
        if i.is_out_of_bounds:
            missile_list.remove(i)
    if w_key_down:
        player_spaceship.update_velocity(spaceship_accel_magnitude) 
    player_spaceship.update_angle([cursor.x_pos, cursor.y_pos])
    player_spaceship.update_position()
    for i in background_dot_list:
        i.update([player_spaceship.x_velocity, player_spaceship.y_velocity])
        SDL_RenderTexture(renderer, i.texture, None, i.position_rect)
    physics_update_v5(asteroid_list)
    i = 0
    while i < len(asteroid_list):
        #j = i
        #while j < len(asteroid_list):
            #if i != j:
                #if circle_collision([asteroid_list[i].x_pos, asteroid_list[i].y_pos], asteroid_list[i].collision_radius, 
                #                    [asteroid_list[j].x_pos, asteroid_list[j].y_pos], asteroid_list[j].collision_radius):
                    # todo: Implement collision response
                    #pixel_collision_array = pixel_perfect_collision(asteroid_list[i], asteroid_list[j])
                    #print(pixel_collision_array)
                    #if pixel_collision_array.size > 0:
                        #collision_velocity_update(asteroid_list[i], asteroid_list[j], pixel_collision_array)
            #j += 1
        #os.system('cls')
        #print(i.alpha_channel_array)
        #asteroid_list[i].update()
        SDL_RenderTextureRotated(renderer, asteroid_prototype.texture, None, 
                                 asteroid_list[i].position_rect, asteroid_list[i].angle, None, SDL_FLIP_NONE)
        if asteroid_list[i].is_out_of_bounds:
            asteroid_list.remove(asteroid_list[i])
            asteroid_list.append(Asteroid(renderer))
        i += 1
    SDL_RenderTextureRotated(renderer, player_spaceship.texture, None, player_spaceship.position_rect, player_spaceship.angle, None, SDL_FLIP_NONE)
    SDL_RenderPresent(renderer)
    frame_time = time.time() - frame_start_time
    if frame_time < seconds_per_frame:
        time.sleep(seconds_per_frame - frame_time)
    #break
    
SDL_Quit()