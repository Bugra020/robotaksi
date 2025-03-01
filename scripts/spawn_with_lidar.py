import carla
import time

def main():

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    blueprint_library = world.get_blueprint_library()
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')

    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('points_per_second', '100000')
    lidar_bp.set_attribute('rotation_frequency', '20')
    lidar_bp.set_attribute('range', '50')

    vehicle = None
    lidar_sensor = None

    try:
        vehicle = world.spawn_actor(blueprint_library.filter('model3')[0], world.get_map().get_spawn_points()[1])
        lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=2.5))
        lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

        lidar_sensor.listen(lambda data: print(f"{len(data)} bytes of data retrieved!\n{vehicle.get_location().x}, {vehicle.get_location().z}"))

        vehicle.set_autopilot(True)

        while True:
          world.tick()     
          time.sleep(1)
    except KeyboardInterrupt:
         print("\n stopping the script")
    finally:
        if lidar_sensor is not None:
             lidar_sensor.destroy()
             print("lidar destroyed")
        if vehicle is not None:
             vehicle.destroy()
             print("vehicle destoyed")

if __name__ == '__main__':
    main()