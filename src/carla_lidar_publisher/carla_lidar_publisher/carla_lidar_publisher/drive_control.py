import heapq
import math
import time
import typing

import carla
import matplotlib.pyplot as plt
from geopy.distance import distance

# parameters
origin_lat = -0.00025120269920364535
origin_lon = -0.000604158081644657
origin_grid_x = 48
origin_grid_y = 58

grid_width = 0
grid_height = 0
map_grid_array = []


# grid and map handling
def load_map(path: str):
    global map_grid_array, grid_width, grid_height
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
        grid_height = len(lines)
        grid_width = len(lines[0].split())
        map_grid_array = [int(val) for line in lines for val in line.split()]
    print(f"Map loaded: {grid_width}x{grid_height}")  # Debug print


def to_index(x: int, y: int) -> int:
    return y * grid_width + x


def from_index(index: int) -> tuple[int, int]:
    return index % grid_width, index // grid_width


def grid_to_gps(x: int, y: int) -> tuple[float, float]:
    dx = x - origin_grid_x
    dy = y - origin_grid_y
    lat_offset = distance(meters=-dy).destination((origin_lat, origin_lon), 0).latitude
    lon_offset = distance(meters=dx).destination((origin_lat, origin_lon), 90).longitude
    return lat_offset, lon_offset


def gps_to_grid(lat: float, lon: float) -> tuple[int, int]:
    north_m = distance((lat, origin_lon), (origin_lat, origin_lon)).meters
    east_m = distance((origin_lat, lon), (origin_lat, origin_lon)).meters
    if lat < origin_lat:
        north_m *= -1
    if lon < origin_lon:
        east_m *= -1
    x = int(round(origin_grid_x + east_m))
    y = int(round(origin_grid_y - north_m))
    return x, y


def gps_to_index(lat: float, lon: float) -> int:
    x, y = gps_to_grid(lat, lon)
    return to_index(x, y)


# path planning
def astar(start_idx: int, goal_idx: int, grid: typing.List[int]) -> typing.List[int]:
    def neighbors(idx):
        x, y = from_index(idx)
        candidates = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_width and 0 <= ny < grid_height:
                nidx = to_index(nx, ny)
                if grid[nidx] == 0:
                    candidates.append(nidx)
        return candidates

    def heuristic(a: int, b: int) -> float:
        ax, ay = from_index(a)
        bx, by = from_index(b)
        return abs(ax - bx) + abs(ay - by)

    open_set = []
    heapq.heappush(open_set, (0, start_idx))
    came_from = {}
    g_score = {start_idx: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal_idx:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_idx)
            path.reverse()
            return path

        for neighbor in neighbors(current):
            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal_idx)
                heapq.heappush(open_set, (f_score, neighbor))
    return []


def drive_to(
    target_gps: tuple[float, float], current_gps: tuple[float, float]
) -> typing.List[int]:
    start_idx = gps_to_index(*current_gps)
    goal_idx = gps_to_index(*target_gps)
    path = astar(start_idx, goal_idx, map_grid_array)
    print(f"Path length: {len(path)}")
    return path


def gps_to_carla_location(lat, lon, z=0.5):
    dx = gps_to_grid(lat, lon)[0] - origin_grid_x
    dy = gps_to_grid(lat, lon)[1] - origin_grid_y
    return carla.Location(x=dx, y=-dy, z=z)


def move_vehicle_along_path(path: typing.List[int], world, vehicle):
    path_world = []
    for index in path:
        x, y = from_index(index)
        lat, lon = grid_to_gps(x, y)
        location = gps_to_carla_location(lat, lon)
        path_world.append(location)

    i = 0
    while i < len(path_world):
        target = path_world[i]
        print(f"Moving to waypoint {i + 1}/{len(path_world)}: {target}")
        reached = drive_to_location(vehicle, target)
        if reached:
            print(f"Reached waypoint {i + 1}")
            i += 1
        time.sleep(0.1)


def drive_to_location(vehicle, target: carla.Location) -> bool:
    transform = vehicle.get_transform()
    current = transform.location
    dx = target.x - current.x
    dy = target.y - current.y
    distance_to_target = math.sqrt(dx**2 + dy**2)

    if distance_to_target < 0.5:
        return True

    yaw = math.radians(transform.rotation.yaw)
    heading_vector = carla.Vector3D(x=math.cos(yaw), y=math.sin(yaw))
    target_vector = carla.Vector3D(x=dx, y=dy)

    dot = heading_vector.x * target_vector.x + heading_vector.y * target_vector.y
    det = heading_vector.x * target_vector.y - heading_vector.y * target_vector.x
    angle = math.atan2(det, dot)

    control = carla.VehicleControl()
    control.throttle = 0.3
    control.steer = max(min(angle, 1.0), -1.0)
    control.brake = 0.0
    vehicle.apply_control(control)

    print(
        f"Driving | Current: ({current.x:.2f}, {current.y:.2f}) | "
        f"Target: ({target.x:.2f}, {target.y:.2f}) | "
        f"Steer: {control.steer:.2f}, Throttle: {control.throttle:.2f}"
    )

    return False


if __name__ == "__main__":
    load_map("/home/bugrakaya/robotaksi/scripts/occupancy_grid.txt")
    current_gps = (origin_lat, origin_lon)
    target_gps = grid_to_gps(
        origin_grid_x + 20, origin_grid_y - 15 - 15
    )  # 15 north, 20 east
    path = drive_to(target_gps, current_gps)

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    vehicle = world.get_actors().filter("vehicle.*")[0]  # Use first existing vehicle
    time.sleep(2)

    move_vehicle_along_path(path, world, vehicle)

    print("Navigation complete.")
