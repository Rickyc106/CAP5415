#!/usr/bin/env python3

import cv2
import carla
import argparse
import random
import time
import queue
import re
import os
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class World(object):
    def __init__(self, args):
        self.host = args.host
        self.port = args.port
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.actor_list = []
        self._weather_presets = self._find_weather_presets()
        
    def destroy(self):
        # Need to destroy actors since they persist in the CARLA server
        # even after this client terminates
        for actor in reversed(self.actor_list):
            actor.destroy()

    def connect_to_server(self, carla_map=None):
        # Connect to CARLA server at specified host IP / port
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(20.0) # seconds

        # Load a random world map unless provided
        map_name = random.choice(self.client.get_available_maps()) if carla_map is None else carla_map
        logger.info('Loading world: %s...' % map_name)
        self.world = self.client.load_world(map_name)

        # Apply random weather preset
        weather = random.choice(self._weather_presets)
        logger.info('Applying weather preset: %s...' % weather[1])
        self.world.set_weather(weather[0])

    def spawn_vehicle(self, bp=None):
        # Pick a random spawn point for hero
        spawn_point = random.choice(self.world.get_map().get_spawn_points())

        # Spawn a random vehicle at spawn point
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = random.choice(blueprint_library.filter('vehicle')) if bp is None else bp
        logger.info('Spawning vehicle...')
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.vehicle.set_autopilot(True)
        self.actor_list.append(self.vehicle)
        logger.info('Created %s' % self.vehicle.type_id)

    def spawn_camera(self):
        # Spawn an RGB camera and attach to vehicle
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        logger.info('Spawning camera...')
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        logger.info('Created %s' % self.camera.type_id)

    def _find_weather_presets(self):
        rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
        name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
        presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
        return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

# ==============================================================================
# -- SynchronousDatasetCollector -----------------------------------------------
# ==============================================================================

class SynchronousDatasetCollector(object):
    def __init__(self, world, fps):
        assert fps >= 10, 'Cannot sync lesser than 10.0 FPS'
        self.world = world.world
        self.camera = world.camera
        self.delta_seconds = 1.0 / fps
        self._settings = None
        self._queues = []

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))
        self._queues.append(queue.Queue())
        self.world.on_tick(self._queues[-1].put)
        self._queues.append(queue.Queue())
        self.camera.listen(self._queues[-1].put)
        return self
    
    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data
    
    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

# ==============================================================================
# -- process_image() -----------------------------------------------------------
# ==============================================================================

def process_image(image):
    image.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3] # Remove alpha channel
    array = array[:, :, ::-1]
    return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

# ==============================================================================
# -- compute_transformation_matrix() -------------------------------------------
# ==============================================================================

def compute_transformation_matrix(tf1, tf2):
    logger.debug(tf1)
    logger.debug(tf2)

    # Rotation about X-axis
    roll = np.radians(tf2.rotation.roll - tf1.rotation.roll)
    # Rotation about Y-axis
    pitch = np.radians(tf2.rotation.pitch - tf1.rotation.pitch)
    # Rotation about Z-axis
    yaw = np.radians(tf2.rotation.yaw - tf1.rotation.yaw)
    
    yaw_matrix = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw),  math.cos(yaw), 0],
        [0, 0, 1]
    ])
    pitch_matrix = np.array([
        [math.cos(pitch), 0, -math.sin(pitch)],
        [0, 1, 0],
        [math.sin(pitch), 0,  math.cos(pitch)]
    ])
    roll_matrix = np.array([
        [1, 0, 0],
        [0,  math.cos(roll), math.sin(roll)],
        [0, -math.sin(roll), math.cos(roll)]
    ])

    # Unreal Engine default rotation order for Euler angles is ZYX
    rotation_matrix = yaw_matrix * pitch_matrix * roll_matrix
    translation_matrix = np.array([
        [tf2.location.x - tf1.location.x],
        [tf2.location.y - tf1.location.y],
        [tf2.location.z - tf1.location.z]
    ])

    logger.debug(rotation_matrix)
    logger.debug(translation_matrix)

    # Concatenate rotation and translation matrices into one transformation matrix
    transformation_matrix = np.concatenate((rotation_matrix, translation_matrix), axis=1)

    return transformation_matrix

# ==============================================================================
# -- simulate_trajectory() -----------------------------------------------------
# ==============================================================================

def simulate_trajectory(args, cv2_out):
    try:
        # Initialize new world and actors
        world = World(args)
        world.connect_to_server()
        world.spawn_vehicle()
        world.spawn_camera()
        
        logger.info('Setup Complete! Data collection in progress...')

        # For odometry ground truth we need to compare all future poses to the
        # initial vehicle pose. This includes position (location) and orientation (rotation).
        initial_pose = world.vehicle.get_transform()
        logger.info(f'Initial pose: {initial_pose}')

        # Ensure we only capture the exact number of frames requested
        frame_count = 0
        max_num_of_frames = args.fps * args.duration
        start_time = time.time()

        # Save all pose transformation matrices for plotting
        ground_truths = []

        with SynchronousDatasetCollector(world, fps=args.fps) as collector:
            while True:
                # Tick client and retrieve world snapshot / image frame
                snapshot, image = collector.tick(timeout=2.0)
                image = process_image(image)

                # Compare vehicle pose compared against initial pose as ground truth odometry
                tf = world.vehicle.get_transform()
                mat = compute_transformation_matrix(tf1=initial_pose, tf2=tf)
                ground_truths.append(mat)

                # Write frame to output video
                cv2_out.write(image)

                # Only render if requested
                if args.render:
                    fps = round(1.0 / snapshot.timestamp.delta_seconds)
                    cv2.putText(image, f'{fps} FPS (simulated)', org=(10, 50),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(255, 255, 255), thickness=2)
                    vel = world.vehicle.get_velocity()
                    cv2.putText(image, f'Speed: {3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2):.2f} km/h',
                        org=(10, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 255, 255), thickness=1)
                    cv2.imshow('Video', image)
                    if cv2.waitKey(1) == ord('q'):
                        break

                # Stop collecting data once requested duration has elapsed
                frame_count += 1
                if frame_count >= max_num_of_frames:
                    break

        end_time = time.time()
        logger.info(f'Collected {frame_count} frames in: {end_time - start_time:.2f} seconds')

        # Write raw data to separate log file to avoid race conditions
        filename = logger.handlers[0].baseFilename.replace('header', 'data')
        with open(filename, 'w') as outfile:
            for mat in ground_truths:
                np.savetxt(outfile, mat, fmt='%-7.2f')

        # Load array from file and ensure it matches
        ground_truths = np.array(ground_truths)
        load_data = np.loadtxt(filename)
        load_data = load_data.reshape((frame_count, 3,4))
        assert np.all(load_data == ground_truths.round(2))

    finally:
        # Clean up actors
        logger.info('Destroying actors')
        world.destroy()
        logger.info('Done')

        # Clean up cv2
        cv2_out.release()
        cv2.destroyAllWindows()

# ==============================================================================
# -- run_main() ----------------------------------------------------------------
# ==============================================================================

def run_main(args):
    # Modify global logger since we are writing to multiple log files
    global logger
    for iteration in range(args.trajectories):
        # Setup logging at specified directory per dataset
        cwd = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(cwd, args.logs)
        os.makedirs(log_dir, exist_ok=True)
        logname = os.path.join(log_dir, f'header-{iteration}.log')
        handler = logging.FileHandler(filename=logname, mode='w')
        formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Set up OpenCV video writer object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        videoname = os.path.join(log_dir, f'output-{iteration}.avi')
        out = cv2.VideoWriter(videoname, fourcc, args.fps, (800, 600))

        # Run data collection for a single iteration
        simulate_trajectory(args, out)

        # Clean up logger
        logger.removeHandler(handler)

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

if __name__ == '__main__':
    try:
        argparser = argparse.ArgumentParser(
            description='Connects to CARLA server and generates visual odometry datasets.')
        argparser.add_argument(
            '--host',
            metavar='H',
            default='127.0.0.1',
            help='IP of the host server (default: 127.0.0.1)')
        argparser.add_argument(
            '-p', '--port',
            metavar='P',
            default=2000,
            type=int,
            help='TCP port to listen to (default: 2000)')
        argparser.add_argument(
            '--trajectories',
            default=10,
            type=int,
            help='Total number of trajectories to collect (default: 10)')
        argparser.add_argument(
            '--duration',
            default=30.0,
            type=float,
            help='Elasped duration in seconds per epoch cycle (default: 30.0)')
        argparser.add_argument(
            '--fps',
            default=30.0,
            type=float,
            help='Simulated frames-per-second (default: 30.0)')
        argparser.add_argument(
            '--render',
            action='store_true',
            help='Show live camera feed (default: False)')
        argparser.add_argument(
            '--logs',
            default='data',
            help='Directory to log datasets to (default: ./data)')
        args = argparser.parse_args()
        run_main(args)

    except KeyboardInterrupt:
        print('Cancelled by user')