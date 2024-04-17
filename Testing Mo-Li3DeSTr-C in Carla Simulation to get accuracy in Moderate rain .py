#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import carla
import numpy as np
import torch
from sklearn.cluster import DBSCAN
import time

def spawn_actor(world, blueprint, spawn_point, color=None):
    """Spawns an actor in the CARLA world."""
    if color and blueprint.has_attribute('color'):
        blueprint.set_attribute('color', color)
    actor = world.try_spawn_actor(blueprint, spawn_point)
    if actor is None:
        print("Spawning failed at:", spawn_point.location)
    else:
        print(f"Spawned {blueprint.id} at {spawn_point.location.x}, {spawn_point.location.y}")
    return actor

def process_lidar_data(data, model, device):
    """Processes LiDAR data to detect objects, predict their classes, including intensity in the output."""
    points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
    points = points.reshape((int(points.shape[0] / 4), 4))

    if points.size == 0:
        return []

    clustering = DBSCAN(eps=0.6, min_samples=10).fit(points[:, :3])
    labels = clustering.labels_
    unique_labels = set(labels)
    
    detected_objects = []
    for label in unique_labels:
        if label == -1:
            continue
        object_points = points[labels == label]
        bbox = calculate_bbox(object_points)
        intensity_mean = np.mean(object_points[:, 3])
        center = np.array([(bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2, (bbox[4] + bbox[5]) / 2, intensity_mean])
        dimensions = np.array([(bbox[1] - bbox[0]) / 2, (bbox[3] - bbox[2]) / 2, (bbox[5] - bbox[4]) / 2])
        tensor_input = torch.tensor([np.concatenate((center, dimensions))], device=device)
        
        with torch.no_grad():
            predicted = model(tensor_input).argmax(1).item()
        
        detected_objects.append({'bbox': bbox, 'label': predicted, 'intensity': intensity_mean})
    
    return detected_objects

def calculate_bbox(points):
    """Calculates bounding box from point cloud data."""
    x_min, y_min = np.min(points[:, :2], axis=0)
    x_max, y_max = np.max(points[:, :2], axis=0)
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])
    return [x_min, x_max, y_min, y_max, z_min, z_max]

def get_ground_truths(world):
    """Retrieves ground truth bounding boxes and labels from the simulation."""
    ground_truths = []
    for actor in world.get_actors():
        if actor.type_id.startswith('vehicle.') or actor.type_id.startswith('walker.pedestrian'):
            bbox = actor.bounding_box.extent
            loc = actor.get_location()
            x_min = loc.x - bbox.x
            x_max = loc.x + bbox.x
            y_min = loc.y - bbox.y
            y_max = loc.y + bbox.y
            z_min = loc.z - bbox.z
            z_max = loc.z + bbox.z
            ground_truths.append({'bbox': [x_min, x_max, y_min, y_max, z_min, z_max], 'label': actor.type_id})
    return ground_truths

# Initialize CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Set the weather to Moderate rain
world.set_weather(carla.WeatherParameters(precipitation=60.0))

# Load model and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('Mo-Li3DeTr_Rainy_Model.pth', map_location=device)
model.to(device)
model.eval()

# Attach LiDAR to a vehicle
vehicle_bp = world.get_blueprint_library().filter('model3')[0]
spawn_point = random.choice(world.get_map().get_spawn_points())
vehicle = spawn_actor(world, vehicle_bp, spawn_point)
lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('range', '5000')
lidar_location = carla.Location(0, 0, 2)
lidar_transform = carla.Transform(lidar_location)
lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

# Simulation parameters
simulation_duration = 10 * 60  # 10 minutes
frame_rate = int(1.0 / world.get_settings().fixed_delta_seconds)
total_frames = simulation_duration * frame_rate

predictions = []
start_time = time.time()

# Run simulation
while time.time() - start_time < simulation_duration:
    world.tick()
    lidar_data = lidar_sensor.listen(lambda data: process_lidar_data(data, model, device))
    predictions.extend(lidar_data)
    if len(predictions) >= total_frames:
        break

# Retrieve and evaluate ground truths
ground_truths = get_ground_truths(world)
precision, recall, f1_score = evaluate_model(predictions, ground_truths)
print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}')

# Clean up
lidar_sensor.stop()
lidar_sensor.destroy()
vehicle.destroy()
print("Simulation actors have been cleaned up.")

