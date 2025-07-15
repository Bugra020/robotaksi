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
    points, resolution=1.0, z_min=0.2, z_max=2.0, min_points=3
):
    filtered_points = points[(points[:, 2] >= z_min) & (points[:, 2] <= z_max)]
    x, y, z = filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2]

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    width = int(np.ceil((x_max - x_min) / resolution))
    height = int(np.ceil((y_max - y_min) / resolution))

    print(f"Map dimensions: {width} x {height} (meters and pixels)")
    print(f"X range: {x_min:.2f} to {x_max:.2f}")
    print(f"Y range: {y_min:.2f} to {y_max:.2f}")

    grid_x = ((x - x_min) / resolution).astype(int)
    grid_y = ((y - y_min) / resolution).astype(int)

    z_sum = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.uint16)

    for gx, gy, gz in zip(grid_x, grid_y, z):
        if 0 <= gy < height and 0 <= gx < width:
            z_sum[gy, gx] += gz
            count[gy, gx] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        avg_z = np.where(count >= min_points, z_sum / count, 0)

    # Normalize and threshold
    normalized = ((avg_z - z_min) / (z_max - z_min) * 255).astype(np.uint8)
    normalized[:, :] = 255  # Set all valid cells to white
    normalized[count < min_points] = 0  # Obstacle cells to black

    return normalized


def main():
    if len(sys.argv) < 5:
        print("Usage: python script.py <res> <min_points> <z_min> <z_max>")
        return

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

    # Show and save image
    plt.imshow(occupancy_grid, cmap="gray", origin="lower")
    plt.title(f"res={res_arg}, min_pts={min_points_arg}, {z_min_arg}<z<{z_max_arg}")
    plt.show()

    Image.fromarray(occupancy_grid).save("occupancy_map.png")
    print("Saved as occupancy_map.png")


if __name__ == "__main__":
    main()
