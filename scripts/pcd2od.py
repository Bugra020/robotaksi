import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def read_pcd_ascii(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    header_ended = False
    data = []
    for line in lines:
        if line.startswith("DATA"):
            header_ended = True
            continue
        if header_ended:
            parts = line.strip().split()
            if len(parts) >= 3:
                x, y, z = map(float, parts[:3])
                data.append((x, y, z))
    return np.array(data)


def pointcloud_to_occupancy_grid(points, resolution=1):
    points = points[(points[:, 2] >= 0.8) & (points[:, 2] <= 2.0)]

    x_coords, y_coords = points[:, 0], points[:, 1]

    x_min, y_min = np.min(x_coords), np.min(y_coords)
    x_coords -= x_min
    y_coords -= y_min

    grid_x = (x_coords / resolution).astype(int)
    grid_y = (y_coords / resolution).astype(int)

    grid_width = grid_x.max() + 1
    grid_height = grid_y.max() + 1

    grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
    grid[grid_y, grid_x] = 255

    return grid


def main():
    pcd_file = "map.pcd"
    points = read_pcd_ascii(pcd_file)
    occupancy_grid = pointcloud_to_occupancy_grid(points, resolution=1)

    plt.imshow(occupancy_grid, cmap="gray", origin="lower")
    plt.title("occupancy grid map")
    plt.show()

    Image.fromarray(occupancy_grid).save("occupancy_map.png")


if __name__ == "__main__":
    main()
