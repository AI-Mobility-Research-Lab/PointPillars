import numpy as np
import open3d as o3d
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


def remove_half_beams(points, num_bins=64):
    """
    Remove every alternate beam using vertical angle binning.
    For LiDAR point clouds, beams are better approximated by their vertical angle rather than the z coordinate directly.
    """
    # Compute the vertical angle (in radians) for each point
    angles = np.arctan2(points[:, 2], np.sqrt(points[:, 0]**2 + points[:, 1]**2))
    # Create equally spaced bins in the angle space corresponding to the number of beams
    bins = np.linspace(angles.min(), angles.max(), num_bins + 1)

    filtered_points = []
    for i in range(num_bins):
        if i % 2 == 0:  # Retain every alternate beam
            # Use <= for the last bin to include the max angle value
            if i == num_bins - 1:
                mask = (angles >= bins[i]) & (angles <= bins[i + 1])
            else:
                mask = (angles >= bins[i]) & (angles < bins[i + 1])
            filtered_points.append(points[mask])

    return np.vstack(filtered_points) if filtered_points else points


def visualize_point_clouds(original, modified):
    """Visualize original and modified point clouds side by side."""
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(original[:, :3])

    modified_pcd = o3d.geometry.PointCloud()
    modified_pcd.points = o3d.utility.Vector3dVector(modified[:, :3])

    # Assign colors for differentiation
    original_pcd.paint_uniform_color([1, 0, 0])  # Red for original
    modified_pcd.paint_uniform_color([0, 1, 0])  # Green for modified

    # Compute translation offset based on the bounding box width of the original point cloud
    bbox = original_pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()  # [width, height, depth]
    translation = np.array([extent[0] + 1, 0, 0])  # Shift modified point cloud to the right by the width plus a 1-unit margin
    modified_pcd.translate(translation)

    o3d.visualization.draw_geometries(
        [original_pcd, modified_pcd],
        window_name="Original (Red) vs Modified (Green) Point Cloud",
    )


def process_kitti_bin(input_file, output_file):
    """Process KITTI .bin file to remove half of the beams and visualize."""
    points = load_kitti_bin(input_file)
    filtered_points = remove_half_beams(points, num_bins=64)

    save_kitti_bin(output_file, filtered_points)
    print(f"Processed file saved to: {output_file}")

    visualize_point_clouds(points, filtered_points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove half the beams from a KITTI .bin point cloud and visualize."
    )
    parser.add_argument("input", help="Path to the input KITTI .bin file")
    parser.add_argument("output", help="Path to the output KITTI .bin file")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
    else:
        process_kitti_bin(args.input, args.output)
