import sys

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


def pointcloud_to_occupancy_grid(
    points, resolution=1, z_min=0.2, z_max=2.0, min_points=3
):
    filtered_points = points[(points[:, 2] >= z_min) & (points[:, 2] <= z_max)]
    x, y, z = filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2]

    x_min, y_min = np.min(x), np.min(y)
    x -= x_min
    y -= y_min

    grid_x = (x / resolution).astype(int)
    grid_y = (y / resolution).astype(int)
    grid_w, grid_h = grid_x.max() + 1, grid_y.max() + 1

    z_sum = np.zeros((grid_h, grid_w), dtype=np.float32)
    count = np.zeros((grid_h, grid_w), dtype=np.uint16)

    for gx, gy, gz in zip(grid_x, grid_y, z):
        z_sum[gy, gx] += gz
        count[gy, gx] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        avg_z = np.where(count >= min_points, z_sum / count, 0)

    avg_z_clipped = np.clip(avg_z, z_min, z_max)
    normalized = ((avg_z_clipped - z_min) / (z_max - z_min) * 255).astype(np.uint8)
    normalized[:, :] = 255
    normalized[count < min_points] = 0

    return normalized


def main():
    res_arg = float(sys.argv[1])
    min_points_arg = float(sys.argv[2])
    z_min_arg, z_max_arg = float(sys.argv[3]), float(sys.argv[4])

    pcd_file = "map.pcd"
    points = read_pcd_ascii(pcd_file)
    occupancy_grid = pointcloud_to_occupancy_grid(
        points,
        resolution=res_arg,
        min_points=min_points_arg,
        z_min=z_min_arg,
        z_max=z_max_arg,
    )

    plt.imshow(occupancy_grid, cmap="gray", origin="lower")
    plt.title(
        f"res={res_arg}, grid_filtering={min_points_arg}, {z_min_arg}<z<{z_max_arg}"
    )
    plt.show()

    Image.fromarray(occupancy_grid).save("occupancy_map.png")


if __name__ == "__main__":
    main()
