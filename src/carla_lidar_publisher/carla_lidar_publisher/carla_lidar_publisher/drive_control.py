import heapq
import typing

from geopy.distance import distance

GRID_WIDTH = 100

origin_lat = -0.00025120269920364535
origin_lon = -0.000604158081644657

map_grid_array = [0] * (GRID_WIDTH * GRID_WIDTH)


def load_map(path: str):
    with open(path, "r") as f:
        values = f.read().split()
        for i in range(len(values)):
            map_grid_array[i] = int(values[i])


def to_index(x: int, y: int) -> int:
    return y * GRID_WIDTH + x


def from_index(index: int) -> tuple[int, int]:
    return index % GRID_WIDTH, index // GRID_WIDTH


def grid_to_gps(x: int, y: int) -> tuple[float, float]:
    lat_offset = distance(meters=y).destination((origin_lat, origin_lon), 180).latitude
    lon_offset = distance(meters=x).destination((origin_lat, origin_lon), 90).longitude
    return lat_offset, lon_offset


def gps_to_grid(lat: float, lon: float) -> tuple[int, int]:
    north_m = distance((lat, origin_lon), (origin_lat, origin_lon)).meters
    east_m = distance((origin_lat, lon), (origin_lat, origin_lon)).meters
    if lat < origin_lat:
        north_m *= -1
    if lon < origin_lon:
        east_m *= -1
    return int(round(east_m)), int(round(north_m))


def gps_to_index(lat: float, lon: float) -> int:
    x, y = gps_to_grid(lat, lon)
    return to_index(x, y)


def astar(start_idx: int, goal_idx: int, grid: list[int]) -> list[int]:
    def h(i1, i2):
        x1, y1 = from_index(i1)
        x2, y2 = from_index(i2)
        return abs(x1 - x2) + abs(y1 - y2)

    open_set = []
    heapq.heappush(open_set, (h(start_idx, goal_idx), 0, start_idx, []))
    visited = set()

    while open_set:
        _, cost, current, path = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        if current == goal_idx:
            return path + [current]

        x, y = from_index(current)
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_WIDTH:
                neighbor = to_index(nx, ny)
                if grid[neighbor] == 0:
                    heapq.heappush(
                        open_set,
                        (
                            cost + 1 + h(neighbor, goal_idx),
                            cost + 1,
                            neighbor,
                            path + [current],
                        ),
                    )
    return []


def drive_to(
    target_gps: typing.Tuple[float, float], current_gps: typing.Tuple[float, float]
) -> list[tuple[float, float]]:
    start = gps_to_index(*current_gps)
    goal = gps_to_index(*target_gps)
    path = astar(start, goal, map_grid_array)
    return [grid_to_gps(*from_index(idx)) for idx in path]
