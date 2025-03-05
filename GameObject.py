import numpy as np
import ctypes
from sdl3 import *
from GameLogic import *
from Collision import *

class Spaceship():
    def __init__(self, x, y, renderer):
        self.x_pos = x
        self.y_pos = y
        self.x_velocity = 0
        self.y_velocity = 0
        self.angle = 0
        self.file_path = "c:/Users/Mike/3D Objects/AI_Project/spaceship.png"
        self.surface = IMG_Load(self.file_path.encode("utf-8"))
        self.surface = SDL_ConvertSurface(self.surface, SDL_PIXELFORMAT_RGBA8888)
        self.width = self.surface.contents.w
        self.height = self.surface.contents.h
        self.position_rect = SDL_FRect(self.x_pos - self.width / 2, self.y_pos - self.height / 2, self.width, self.height)
        self.texture = SDL_CreateTextureFromSurface(renderer, self.surface)

    def update_velocity(self, ship_acceleration):
        self.x_velocity += ship_acceleration * np.sin(self.angle * np.pi / 180)
        self.y_velocity -= ship_acceleration * np.cos(self.angle * np.pi / 180)
        velocity_magnitude = np.sqrt(self.x_velocity ** 2 + self.y_velocity ** 2)
        if velocity_magnitude > 5:
            self.x_velocity *= 5 / velocity_magnitude
            self.y_velocity *= 5 / velocity_magnitude

    def update_position(self):
        self.x_pos += self.x_velocity
        self.y_pos += self.y_velocity
        self.position_rect.x = self.x_pos - self.width / 2
        self.position_rect.y = self.y_pos - self.height / 2
    
    def update_angle(self, cursor_position):
        self.angle = angle_y_axis_two_points([self.x_pos, self.y_pos], cursor_position)

class Missile():
    def __init__(self, x, y, x_velocity, y_velocity, angle, renderer):
        self.x_pos = x
        self.y_pos = y
        self.angle = angle
        self.acceleration = 0.3
        self.x_velocity = x_velocity
        self.y_velocity = y_velocity
        self.file_path = "c:/Users/Mike/3D Objects/AI_Project/missile.png"
        self.surface = IMG_Load(self.file_path.encode("utf-8"))
        self.scale = 0.25
        self.width = self.surface.contents.w * self.scale
        self.height = self.surface.contents.h * self.scale
        self.position_rect = SDL_FRect(self.x_pos - self.width / 2, self.y_pos - self.height / 2, self.width, self.height)
        self.texture = SDL_CreateTextureFromSurface(renderer, self.surface)
        self.is_out_of_bounds = False
        
    def update(self):
        self.x_velocity += self.acceleration * np.sin(self.angle * np.pi / 180)
        self.y_velocity -= self.acceleration * np.cos(self.angle * np.pi / 180)
        self.x_pos += self.x_velocity
        self.y_pos += self.y_velocity
        self.position_rect.x = self.x_pos - self.width / 2
        self.position_rect.y = self.y_pos - self.height / 2
        if self.x_pos < -self.width or self.x_pos > 1280 + self.width or self.y_pos < -self.height or self.y_pos > 720 + self.height:
            self.is_out_of_bounds = True

class Asteroid():
    def __init__(self, renderer):
        self.x_pos = np.random.randint(0, 1280)
        self.y_pos = np.random.randint(0, 720)
        self.pos = np.array([self.x_pos, self.y_pos])
        self.x_velocity = np.random.randint(-16, 16) / 10
        self.y_velocity = np.random.randint(-16, 16) / 10
        self.vel = np.array([self.x_velocity, self.y_velocity])
        self.angle = np.random.randint(0, 360)
        self.angular_velocity = np.random.randint(-10, 10) / 10
        self.file_path = "c:/Users/Mike/3D Objects/AI_Project/asteroid.png"
        self.surface = IMG_Load(self.file_path.encode("utf-8"))
        self.surface = SDL_ConvertSurface(self.surface, SDL_PIXELFORMAT_RGBA8888)
        self.alpha_channel_array = pixel_alpha_channel_extraction(self.surface).copy()
        #print(self.alpha_channel_array)
        self.width = self.surface.contents.w
        self.height = self.surface.contents.h
        self.position_rect = SDL_FRect(self.x_pos - self.width / 2, self.y_pos - self.height / 2, self.width, self.height)
        self.texture = SDL_CreateTextureFromSurface(renderer, self.surface)
        self.collision_radius = np.sqrt(self.width ** 2 + self.height ** 2) / 2
        self.is_out_of_bounds = False
        self.is_colliding = False
        self.boundary_pixels = get_boundary_pixels(self.surface).copy()
        #print(self.boundary_pixels)
        
    def update(self):
        self.x_pos += self.x_velocity
        self.y_pos += self.y_velocity
        self.position_rect.x = self.x_pos - self.width / 2
        self.position_rect.y = self.y_pos - self.height / 2
        #self.angle += self.angular_velocity
        if self.x_pos < -self.collision_radius or self.x_pos > 1280 + self.collision_radius or self.y_pos < -self.collision_radius or self.y_pos > 720 + self.collision_radius:
            self.is_out_of_bounds = True

class Cursor():
    def __init__(self):
        self.x_pos = 0
        self.y_pos = 0
        self.file_path = "c:/Users/Mike/3D Objects/AI_Project/crosshair.png"
        self.surface = IMG_Load(self.file_path.encode("utf-8"))
        self.cursor = SDL_CreateColorCursor(self.surface, 16, 16)
        SDL_SetCursor(self.cursor)

    def update(self, x, y):
        self.x_pos = x
        self.y_pos = y
    
class BackgroundDot:
    def __init__(self, vel_vector, screen_width, screen_height, renderer):
        self.vel_vector = vel_vector
        self.file_path = "c:/Users/Mike/3D Objects/AI_Project/dot.png"
        self.x_pos = np.random.randint(0, screen_width)
        self.y_pos = np.random.randint(0, screen_height)
        self.surface = IMG_Load(self.file_path.encode("utf-8"))
        self.width = self.surface.contents.w
        self.height = self.surface.contents.h
        self.position_rect = SDL_FRect(self.x_pos, self.y_pos, self.width, self.height)
        self.texture = SDL_CreateTextureFromSurface(renderer, self.surface)
        self.parallax_scale = np.random.randint(100, 500) / 1000
        
    def update(self, vel_vector):
        self.vel_vector = vel_vector
        self.x_pos -= self.vel_vector[0] * self.parallax_scale
        self.y_pos -= self.vel_vector[1] * self.parallax_scale
        self.position_rect.x = self.x_pos
        self.position_rect.y = self.y_pos