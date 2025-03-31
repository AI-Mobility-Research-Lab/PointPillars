import numpy as np
import argparse
import os


def load_kitti_bin(file_path):
    """Load a KITTI .bin file into a NumPy array."""
    return np.fromfile(file_path, dtype=np.float32).reshape(
        -1, 4
    )  # KITTI format: x, y, z, intensity


def save_kitti_bin(file_path, point_cloud):
    """Save a NumPy array as a KITTI .bin file."""
    point_cloud.astype(np.float32).tofile(file_path)


def remove_half_beams(points):
    """
    Remove every alternate beam based on the z-values.
    Assuming the dataset has structured beams.
    """
    # Sort points by z-axis (height)
    sorted_indices = np.argsort(points[:, 2])
    sorted_points = points[sorted_indices]

    # Select only alternate beams
    filtered_points = sorted_points[::2]

    return filtered_points


def process_kitti_bin(input_file, output_file):
    """Process KITTI .bin file to remove half of the beams."""
    points = load_kitti_bin(input_file)
    filtered_points = remove_half_beams(points)
    save_kitti_bin(output_file, filtered_points)
    print(f"Processed file saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove half the beams from a KITTI .bin point cloud."
    )
    parser.add_argument("input", help="Path to the input KITTI .bin file")
    parser.add_argument("output", help="Path to the output KITTI .bin file")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
    else:
        process_kitti_bin(args.input, args.output)
