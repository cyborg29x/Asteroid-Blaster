import sdl3, sys, time, numpy as np

sdl3.SDL_Init(sdl3.SDL_INIT_VIDEO)
window_w = 640
window_h = 480

window = sdl3.SDL_CreateWindow(b"Test Window", window_w, window_h, 0)
window_surface = sdl3.SDL_GetWindowSurface(window)
#renderer = sdl3.SDL_CreateRenderer(window, -1, 0)

custom_cursor_surface = sdl3.IMG_Load("c:/Users/Mike/3D Objects/AI_Project/crosshair.png".encode("utf-8"))
if not custom_cursor_surface:
    print("Error loading cursor image: ", sdl3.SDL_GetError())
custom_cursor = sdl3.SDL_CreateColorCursor(custom_cursor_surface, 16, 16)
sdl3.SDL_SetCursor(custom_cursor)

spaceship_surface = sdl3.IMG_Load("c:/Users/Mike/3D Objects/AI_Project/spaceship.png".encode("utf-8"))
if not spaceship_surface:
    print("Error loading spaceship image: ", sdl3.SDL_GetError())
spaceship_position_w = int(window_w / 2)
spaceship_position_h = int(window_h / 2) #lazy conversion, should be done with ctypes
print(spaceship_position_w, spaceship_position_h)
spaceship_width = spaceship_surface.contents.w
spaceship_height = spaceship_surface.contents.h
print(spaceship_width, spaceship_height)
spaceship_position_rect = sdl3.SDL_Rect(int(spaceship_position_w - spaceship_width / 2) ,
                                        int(spaceship_position_h - spaceship_height / 2),
                                        spaceship_surface.contents.w, 
                                        spaceship_surface.contents.h)
sdl3.SDL_RotateSurface
fps = 60
seconds_per_frame = 1 / fps

running = True
event = sdl3.SDL_Event()
while running:
    frame_start_time = time.time()
    while sdl3.SDL_PollEvent(event):
        if event.type == sdl3.SDL_EVENT_QUIT:
            running = False
        #if event.type == sdl3.SDL_EVENT_MOUSE_MOTION:
            #print("Mouse moved to: ", event.motion.x, event.motion.y)
    sdl3.SDL_BlitSurface(spaceship_surface, None, window_surface, spaceship_position_rect)
    sdl3.SDL_UpdateWindowSurface(window)
    frame_time = time.time() - frame_start_time
    if frame_time < seconds_per_frame:
        time.sleep(seconds_per_frame - frame_time) 
    
sdl3.SDL_DestroyWindow(window)
sdl3.SDL_Quit()