#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import carla
import numpy as np
import random
import torch
from sklearn.cluster import DBSCAN

# Function to spawn actors in the CARLA world
def spawn_actor(world, blueprint, spawn_point, color=None):
    """Attempts to spawn the given blueprint at the spawn point with a specified color."""
    if color and blueprint.has_attribute('color'):
        blueprint.set_attribute('color', color)
    actor = world.try_spawn_actor(blueprint, spawn_point)
    if actor is None:
        print("Spawning failed at:", spawn_point.location)
    else:
        print(f"Spawned {blueprint.id} at {spawn_point.location.x}, {spawn_point.location.y}")
    return actor

# Function to process LiDAR data and draw bounding boxes around detected objects
def process_lidar_data(data, world, vehicle):
    """Processes LiDAR data to find objects and draw bounding boxes around them."""
    points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 4), 4))[:, :3]  # Extracting x, y, z coordinates

    if points.size == 0:
        print("No points detected by LiDAR this frame.")
        return  # No points to process

    clustering = DBSCAN(eps=0.6, min_samples=10).fit(points)
    labels = clustering.labels_

    unique_labels = set(labels)
    print(f"Detected {len(unique_labels) - 1 if -1 in unique_labels else len(unique_labels)} objects.")
    for label in unique_labels:
        if label != -1:
            object_points = points[labels == label]
            z_min = np.min(object_points[:, 2])
            z_max = np.max(object_points[:, 2])
            x_min, y_min = np.min(object_points, axis=0)[:2]
            x_max, y_max = np.max(object_points, axis=0)[:2]
            bbox = carla.BoundingBox(
                carla.Location(x=(x_min + x_max) / 2, y=(y_min + y_max) / 2, z=(z_min + z_max) / 2),
                carla.Vector3D(x=(x_max - x_min) / 2, y=(y_max - y_min) / 2, z=(z_max - z_min) / 2)
            )
            world.debug.draw_box(bbox, carla.Rotation(), 0.1, carla.Color(0, 255, 0, 0), 0.05)

# Function to setup vehicles and bikes in the world
def setup_vehicles_and_bikes(number_of_vehicles, number_of_bikes, world, blueprint_library, spawn_points):
    vehicles = []
    bikes = []
    vehicle_bp = blueprint_library.filter("vehicle.audi.a2")[0]
    bike_bp = blueprint_library.filter("vehicle.bh.crossbike")[0]

    chosen_points = random.sample(spawn_points, number_of_vehicles + number_of_bikes)

    for i in range(number_of_vehicles):
        color = random.choice(['255,0,0', '0,255,0', '0,0,255', '255,255,0', '0,255,255'])
        vehicle = spawn_actor(world, vehicle_bp, chosen_points[i], color)
        vehicles.append(vehicle)

    for i in range(number_of_vehicles, number_of_vehicles + number_of_bikes):
        bike = spawn_actor(world, bike_bp, chosen_points[i])
        bikes.append(bike)

    return vehicles, bikes

# Load the model correctly and check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('Mo-Li3DeTr_Rainy_Model.pth', map_location=device)  # Correct model name
model = model.to(device)
model.eval()

# CARLA setup
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Set the weather to Moderate rain
world.set_weather(carla.WeatherParameters(
    cloudiness=80.0,  # Considerable cloud cover, but not completely overcast
    precipitation=70.0,  # Substantial precipitation, but not at maximum
    precipitation_deposits=70.0,  # Significant wetness on surfaces, but not flooded
    wind_intensity=30.0,  # Moderate wind, less than the stormy conditions
    fog_density=50.0,  # Reduced fog density for better visibility than maximum fog
    wetness=70.0,  # Noticeably wet surfaces, contributing to slippery conditions
    sun_azimuth_angle=0.0,  # Standard positioning for the sun
    sun_altitude_angle=-20.0  # Sun is low, simulating either dawn or dusk, not complete darkness
))

blueprint_library = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

number_of_vehicles = 30
number_of_bikes = 15

vehicles, bikes = setup_vehicles_and_bikes(number_of_vehicles, number_of_bikes, world, blueprint_library, spawn_points)

for v in vehicles + bikes:
    v.set_autopilot(True)  # Enable autopilot for each vehicle and bike

# Attach a LiDAR sensor to a randomly selected vehicle
lidar_vehicle = random.choice(vehicles)
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))  # Mounted on the roof
lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=lidar_vehicle)
print(f"LiDAR sensor attached to vehicle at {lidar_vehicle.get_location().x}, {lidar_vehicle.get_location().y}.")
lidar.listen(lambda data: process_lidar_data(data, world, lidar_vehicle))

try:
    while True:
        world.wait_for_tick()
finally:
    # Clean up all actors
    lidar.stop()
    lidar.destroy()
    for v in vehicles + bikes:
        v.destroy()
    print("Simulation actors have been cleaned up.")

