import numpy as np

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

def pixel_perfect_collision():
    
    pass

def collision_velocity_update(object_1, object_2):
    # Elastic collision
    m1 = object_1.width * object_1.height  # Approximate mass based on size
    m2 = object_2.width * object_2.height
    v1x = object_1.x_velocity
    v1y = object_1.y_velocity
    v2x = object_2.x_velocity
    v2y = object_2.y_velocity
    
    v1x_new = ((m1 - m2) / (m1 + m2)) * v1x + ((2 * m2) / (m1 + m2)) * v2x
    v1y_new = ((m1 - m2) / (m1 + m2)) * v1y + ((2 * m2) / (m1 + m2)) * v2y
    v2x_new = ((2 * m1) / (m1 + m2)) * v1x + ((m2 - m1) / (m1 + m2)) * v2x
    v2y_new = ((2 * m1) / (m1 + m2)) * v1y + ((m2 - m1) / (m1 + m2)) * v2y
    
    object_1.x_velocity = v1x_new
    object_1.y_velocity = v1y_new
    object_2.x_velocity = v2x_new
    object_2.y_velocity = v2y_new
    
    # Separate slightly to prevent sticking
    distance = np.sqrt((object_1.x_pos - object_2.x_pos)**2 + (object_1.y_pos - object_2.y_pos)**2)
    overlap = object_1.collision_radius + object_2.collision_radius - distance
    dx = (object_1.x_pos - object_2.x_pos) / distance
    dy = (object_1.y_pos - object_2.y_pos) / distance
    
    object_1.x_pos += overlap * dx / 2
    object_1.y_pos += overlap * dy / 2
    object_2.x_pos -= overlap * dx / 2
    object_2.y_pos -= overlap * dy / 2