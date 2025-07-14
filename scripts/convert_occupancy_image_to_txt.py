import numpy as np
from PIL import Image


def convert_image_to_txt_reverse(
    image_path: str, output_path: str, threshold: int = 128
):
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)

    binary_grid = (img_array > threshold).astype(int)

    with open(output_path, "w") as f:
        for val in binary_grid.flatten():
            f.write(f"{val}\n")

    print(
        f"Saved reversed occupancy grid to {output_path}, total {binary_grid.size} entries."
    )


if __name__ == "__main__":
    convert_image_to_txt_reverse("occupancy_map.png", "occupancy_grid.txt")
