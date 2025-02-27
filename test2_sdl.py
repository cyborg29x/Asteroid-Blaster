import sdl3, sys, time, numpy as np
import sdl3.SDL

# Create classes for the game objects

sdl3.SDL_Init(sdl3.SDL_INIT_VIDEO)
window_w = 640
window_h = 480


window = sdl3.SDL_CreateWindow(b"Game Window", window_w, window_h, sdl3.SDL_WINDOW_OPENGL)
if not window:
    print("Error creating window: ", sdl3.SDL_GetError())
renderer = sdl3.SDL_CreateRenderer(window, None)
if not renderer:
    print("Error creating renderer: ", sdl3.SDL_GetError())
sdl3.SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255)
if not renderer:
    print("Error setting draw color: ", sdl3.SDL_GetError())
custom_cursor_surface = sdl3.IMG_Load("c:/Users/Mike/3D Objects/AI_Project/crosshair.png".encode("utf-8"))
if not custom_cursor_surface:
    print("Error loading cursor image: ", sdl3.SDL_GetError())
custom_cursor = sdl3.SDL_CreateColorCursor(custom_cursor_surface, 16, 16)
sdl3.SDL_SetCursor(custom_cursor)

spaceship_surface = sdl3.IMG_Load("c:/Users/Mike/3D Objects/AI_Project/spaceship.png".encode("utf-8"))
if not spaceship_surface:
    print("Error loading spaceship image: ", sdl3.SDL_GetError())
spaceship_position_w = window_w / 2
spaceship_position_h = window_h / 2
print(spaceship_position_w, spaceship_position_h)
spaceship_width = spaceship_surface.contents.w
spaceship_height = spaceship_surface.contents.h
print(spaceship_width, spaceship_height)
spaceship_position_rect = sdl3.SDL_FRect(spaceship_position_w - spaceship_width / 2,
                                        spaceship_position_h - spaceship_height / 2,
                                        spaceship_surface.contents.w, 
                                        spaceship_surface.contents.h)
fps = 60
seconds_per_frame = 1 / fps

spaceship_texture = sdl3.SDL_CreateTextureFromSurface(renderer, spaceship_surface)
if not spaceship_texture:
    print("Error creating spaceship texture: ", sdl3.SDL_GetError())
angle = 0
spaceship_accel_magnitude = 1
ship_vel = [0, 0]
cursor_position = [window_w / 2, 0]
dx = 0
dy = 1


running = True
event = sdl3.SDL_Event()
while running:
    frame_start_time = time.time()
    sdl3.SDL_RenderClear(renderer)
    while sdl3.SDL_PollEvent(event):
        if event.type == sdl3.SDL_EVENT_QUIT:
            running = False
        if event.type == sdl3.SDL_EVENT_MOUSE_MOTION:
            cursor_position[0] = event.motion.x
            cursor_position[1] = event.motion.y
        if event.type == sdl3.SDL_EVENT_KEY_DOWN:
            if event.key.key == sdl3.SDLK_W:
                ship_vel[0] += spaceship_accel_magnitude * np.sin(angle * np.pi / 180)
                ship_vel[1] -= spaceship_accel_magnitude * np.cos(angle * np.pi / 180)       
    delta_pos = [cursor_position[0] - spaceship_position_rect.x, cursor_position[1] - spaceship_position_rect.y]
    if delta_pos[0] > 0:
        angle = 180 - np.arccos(delta_pos[1] / np.sqrt(delta_pos[0] ** 2 + delta_pos[1] ** 2)) * 180 / np.pi
    else:
        angle = np.arccos(delta_pos[1] / np.sqrt(delta_pos[0] ** 2 + delta_pos[1] ** 2)) * 180 / np.pi - 180
    spaceship_position_rect.x += ship_vel[0]
    spaceship_position_rect.y += ship_vel[1]
    sdl3.SDL_RenderTextureRotated(renderer, spaceship_texture, None, spaceship_position_rect, angle, None, sdl3.SDL_FLIP_NONE)
    sdl3.SDL_RenderPresent(renderer)
    frame_time = time.time() - frame_start_time
    if frame_time < seconds_per_frame:
        time.sleep(seconds_per_frame - frame_time) 
    
sdl3.SDL_DestroyWindow(window)
sdl3.SDL_Quit()