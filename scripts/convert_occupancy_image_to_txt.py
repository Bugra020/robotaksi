import numpy as np
from PIL import Image


def convert_image_to_txt_proper(
    image_path: str, output_path: str, threshold: int = 128
):
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)

    binary_grid = (img_array > threshold).astype(int)

    with open(output_path, "w") as f:
        for row in binary_grid:
            f.write(" ".join(str(val) for val in row) + "\n")

    print(
        f"Saved occupancy grid to {output_path}, size {binary_grid.shape[0]}x{binary_grid.shape[1]}"
    )


if __name__ == "__main__":
    convert_image_to_txt_proper("occupancy_map.png", "occupancy_grid.txt")
